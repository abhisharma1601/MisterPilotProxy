import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ...logging_config import log_pii_findings
from ...services.approval_registry import get_approval_registry
from ...services.pii_service import get_pii_pipeline
from ...services.terminal_service import get_terminal_service

log = logging.getLogger("terminal")

router = APIRouter()


# ── models ─────────────────────────────────────────────────────────────────────

class StageRequest(BaseModel):
    command: str
    workspace_root: str


class StageResponse(BaseModel):
    id: str
    command: str
    workspace_root: str


class ExecuteRequest(BaseModel):
    id: str
    approved: bool
    timeout: int = 30


class ExecuteResponse(BaseModel):
    id: str
    command: str
    approved: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool
    error: Optional[str] = None


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("/stage", response_model=StageResponse)
async def stage_command(req: StageRequest) -> StageResponse:
    """
    Register a terminal command for user approval.
    Does NOT execute anything — the command only runs after /execute is called
    with approved=true.
    """
    pipeline = get_pii_pipeline()
    sanitized_command, findings = pipeline.redact(req.command)
    if findings:
        log_pii_findings(log, "/terminal/stage", findings)

    svc = get_terminal_service()
    try:
        cmd = svc.stage(sanitized_command, req.workspace_root)
    except ValueError as exc:
        log.warning("stage_command: invalid input — %s", exc)
        raise HTTPException(status_code=422, detail="Invalid input")
    except NotADirectoryError as exc:
        log.warning("stage_command: not a directory — %s", exc)
        raise HTTPException(status_code=400, detail="Not a directory")

    return StageResponse(id=cmd.id, command=cmd.command, workspace_root=cmd.workspace_root)


@router.post("/execute", response_model=ExecuteResponse)
async def execute_command(req: ExecuteRequest) -> ExecuteResponse:
    """
    Execute (or deny) a staged command.
    approved=true  → run the command and return captured output.
    approved=false → discard without any execution.
    """
    if req.timeout < 1 or req.timeout > 120:
        raise HTTPException(
            status_code=422, detail="timeout must be between 1 and 120 seconds"
        )

    svc = get_terminal_service()
    try:
        result = await svc.execute(req.id, req.approved, req.timeout)
    except KeyError as exc:
        log.warning("execute_command: command not found — %s", exc)
        raise HTTPException(status_code=404, detail="Command not found")
    except TimeoutError as exc:
        log.warning("execute_command: command expired — %s", exc)
        raise HTTPException(status_code=410, detail="Command expired")

    pipeline = get_pii_pipeline()
    stdout, stdout_findings = pipeline.redact(result.stdout)
    stderr, stderr_findings = pipeline.redact(result.stderr)
    all_findings = stdout_findings + stderr_findings
    if all_findings:
        log_pii_findings(log, "/terminal/execute (output)", all_findings)

    # Unblock any waiting agent coroutine
    get_approval_registry().resolve(
        req.id,
        {
            "approved": result.approved,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
        },
    )

    return ExecuteResponse(
        id=result.id,
        command=result.command,
        approved=result.approved,
        stdout=stdout,
        stderr=stderr,
        exit_code=result.exit_code,
        duration_ms=result.duration_ms,
        timed_out=result.timed_out,
        error=result.error,
    )


@router.get("/pending/{cmd_id}", response_model=StageResponse)
async def get_pending_command(cmd_id: str) -> StageResponse:
    """Retrieve a staged command (useful for extension polling)."""
    svc = get_terminal_service()
    cmd = svc.get(cmd_id)
    if cmd is None:
        raise HTTPException(status_code=404, detail=f"Pending command not found: {cmd_id!r}")
    return StageResponse(id=cmd.id, command=cmd.command, workspace_root=cmd.workspace_root)
