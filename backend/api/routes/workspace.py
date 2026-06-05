import time
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from ...services.workspace import list_files
from ...tools.file_tools import read_file, PathTraversalError
from ...tools.search_tools import search_code, SearchMatch

router = APIRouter()


class FileListResponse(BaseModel):
    root: str
    files: List[str]
    total: int


class FileContentResponse(BaseModel):
    path: str
    content: str
    size: int


class SearchRequest(BaseModel):
    root: str
    query: str
    case_sensitive: bool = False
    use_regex: bool = False
    file_pattern: Optional[str] = None
    max_results: int = 200


class MatchItem(BaseModel):
    file: str
    line: int
    content: str


class SearchResponse(BaseModel):
    query: str
    matches: List[MatchItem]
    total: int
    engine: str       # "ripgrep" | "python"
    elapsed_ms: int


@router.get("/files", response_model=FileListResponse)
async def get_workspace_files(
    root: str = Query(..., description="Absolute path to workspace root"),
) -> FileListResponse:
    """Return a list of all non-ignored file paths relative to root."""
    try:
        files = list_files(root)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return FileListResponse(root=root, files=files, total=len(files))


@router.get("/file", response_model=FileContentResponse)
async def get_file_content(
    root: str = Query(..., description="Absolute path to workspace root"),
    path: str = Query(..., description="File path relative to workspace root"),
) -> FileContentResponse:
    """Return the text content of a single file."""
    try:
        content = read_file(root, path)
    except PathTraversalError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (IsADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return FileContentResponse(path=path, content=content, size=len(content))


@router.post("/search", response_model=SearchResponse)
async def search_workspace(req: SearchRequest) -> SearchResponse:
    """Full-text search across all workspace files."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    if req.max_results < 1 or req.max_results > 1000:
        raise HTTPException(status_code=422, detail="max_results must be between 1 and 1000")

    t0 = time.monotonic()
    try:
        matches, engine = await search_code(
            root=req.root,
            query=req.query,
            case_sensitive=req.case_sensitive,
            use_regex=req.use_regex,
            file_pattern=req.file_pattern,
            max_results=req.max_results,
        )
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    elapsed = int((time.monotonic() - t0) * 1000)

    return SearchResponse(
        query=req.query,
        matches=[MatchItem(file=m.file, line=m.line, content=m.content) for m in matches],
        total=len(matches),
        engine=engine,
        elapsed_ms=elapsed,
    )
