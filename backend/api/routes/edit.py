from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ...services.approval_registry import get_approval_registry
from ...services.edit_service import get_edit_service
from ...tools.file_tools import PathTraversalError

router = APIRouter()


# ── request / response models ─────────────────────────────────────────────────

class WritePreviewRequest(BaseModel):
    root: str
    path: str
    content: str


class ReplacePreviewRequest(BaseModel):
    root: str
    path: str
    old_text: str
    new_text: str


class PreviewResponse(BaseModel):
    id: str
    path: str
    diff: str
    original: str
    proposed: str
    is_new_file: bool


class ApplyRequest(BaseModel):
    id: str


class ApplyResponse(BaseModel):
    applied: bool
    path: str


class RejectRequest(BaseModel):
    id: str


class RejectResponse(BaseModel):
    rejected: bool
    path: str


# ── endpoints ─────────────────────────────────────────────────────────────────

def _to_preview(edit) -> PreviewResponse:
    return PreviewResponse(
        id=edit.id,
        path=edit.path,
        diff=edit.diff(),
        original=edit.original,
        proposed=edit.proposed,
        is_new_file=edit.is_new_file,
    )


@router.post("/preview/write", response_model=PreviewResponse)
async def preview_write(req: WritePreviewRequest) -> PreviewResponse:
    """Stage a write_file — returns diff without touching the filesystem."""
    svc = get_edit_service()
    try:
        edit = svc.preview_write(req.root, req.path, req.content)
    except PathTraversalError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _to_preview(edit)


@router.post("/preview/replace", response_model=PreviewResponse)
async def preview_replace(req: ReplacePreviewRequest) -> PreviewResponse:
    """Stage a replace_in_file — returns diff without touching the filesystem."""
    svc = get_edit_service()
    try:
        edit = svc.preview_replace(req.root, req.path, req.old_text, req.new_text)
    except PathTraversalError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _to_preview(edit)


@router.post("/apply", response_model=ApplyResponse)
async def apply_edit(req: ApplyRequest) -> ApplyResponse:
    """Apply a pending edit — writes to disk and removes from the pending store."""
    svc = get_edit_service()
    try:
        edit = svc.apply(req.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except TimeoutError as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    # Unblock any waiting agent coroutine
    get_approval_registry().resolve(req.id, {"applied": True, "path": edit.path})
    return ApplyResponse(applied=True, path=edit.path)


@router.post("/reject", response_model=RejectResponse)
async def reject_edit(req: RejectRequest) -> RejectResponse:
    """Discard a pending edit without modifying any files."""
    svc = get_edit_service()
    try:
        edit = svc.reject(req.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    # Unblock any waiting agent coroutine
    get_approval_registry().resolve(req.id, {"applied": False, "path": edit.path})
    return RejectResponse(rejected=True, path=edit.path)


@router.get("/pending/{edit_id}", response_model=PreviewResponse)
async def get_pending_edit(edit_id: str) -> PreviewResponse:
    """Retrieve a pending edit by ID (for polling or re-display)."""
    svc = get_edit_service()
    edit = svc.get(edit_id)
    if edit is None:
        raise HTTPException(status_code=404, detail=f"Pending edit not found: {edit_id!r}")
    return _to_preview(edit)
