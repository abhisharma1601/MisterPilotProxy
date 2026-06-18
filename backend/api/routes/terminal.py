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


@router.post("/execute", response_model=ExecuteResponse)
async def execute_command(req: ExecuteRequest) -> ExecuteResponse:
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
