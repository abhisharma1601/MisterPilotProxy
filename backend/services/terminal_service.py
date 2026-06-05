import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_PENDING_TTL_SECONDS = 120   # approval window: 2 minutes
_MAX_OUTPUT_BYTES = 50_000   # cap stdout/stderr at 50 KB each
_DEFAULT_TIMEOUT = 30        # command execution timeout in seconds


@dataclass
class PendingCommand:
    id: str
    command: str
    workspace_root: str
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > _PENDING_TTL_SECONDS


@dataclass
class ExecutionResult:
    id: str
    command: str
    approved: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_ms: int = 0
    timed_out: bool = False
    error: Optional[str] = None


async def _run(command: str, cwd: str, timeout: int) -> tuple[str, str, int, bool]:
    """
    Run command in cwd. Returns (stdout, stderr, exit_code, timed_out).
    Kills the process on timeout; caps output at _MAX_OUTPUT_BYTES each.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        timed_out = False
    except asyncio.TimeoutError:
        proc.kill()
        raw_out, raw_err = await proc.communicate()
        timed_out = True

    stdout = raw_out.decode(errors="replace")[:_MAX_OUTPUT_BYTES]
    stderr = raw_err.decode(errors="replace")[:_MAX_OUTPUT_BYTES]
    return stdout, stderr, proc.returncode or 0, timed_out


class TerminalService:
    def __init__(self) -> None:
        self._pending: Dict[str, PendingCommand] = {}

    # ── internal ──────────────────────────────────────────────────────────────

    def _prune_expired(self) -> None:
        for k in [k for k, v in self._pending.items() if v.is_expired()]:
            del self._pending[k]

    # ── stage ─────────────────────────────────────────────────────────────────

    def stage(self, command: str, workspace_root: str) -> PendingCommand:
        """
        Register a command for user approval. Does NOT execute anything.

        Raises:
            ValueError           — command is empty
            NotADirectoryError   — workspace_root does not exist
        """
        command = command.strip()
        if not command:
            raise ValueError("Command must not be empty")

        if not Path(workspace_root).is_dir():
            raise NotADirectoryError(
                f"Workspace root is not a directory: {workspace_root!r}"
            )

        self._prune_expired()

        cmd = PendingCommand(
            id=str(uuid.uuid4()),
            command=command,
            workspace_root=workspace_root,
        )
        self._pending[cmd.id] = cmd
        return cmd

    def get(self, cmd_id: str) -> Optional[PendingCommand]:
        return self._pending.get(cmd_id)

    # ── execute ───────────────────────────────────────────────────────────────

    async def execute(
        self,
        cmd_id: str,
        approved: bool,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> ExecutionResult:
        """
        Consume a pending command.

        If approved=True: run it and return captured output.
        If approved=False: discard without any execution.

        Raises:
            KeyError    — cmd_id not found or already consumed
            TimeoutError — pending command expired before approval
        """
        cmd = self._pending.pop(cmd_id, None)
        if cmd is None:
            raise KeyError(f"Pending command not found: {cmd_id!r}")

        if cmd.is_expired():
            raise TimeoutError(f"Pending command expired: {cmd_id!r}")

        if not approved:
            return ExecutionResult(
                id=cmd.id,
                command=cmd.command,
                approved=False,
            )

        t0 = time.monotonic()
        try:
            stdout, stderr, exit_code, timed_out = await _run(
                cmd.command, cmd.workspace_root, timeout
            )
        except OSError as exc:
            return ExecutionResult(
                id=cmd.id,
                command=cmd.command,
                approved=True,
                error=f"Failed to start process: {exc}",
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

        result = ExecutionResult(
            id=cmd.id,
            command=cmd.command,
            approved=True,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=int((time.monotonic() - t0) * 1000),
            timed_out=timed_out,
        )

        if timed_out:
            result.stderr = (
                f"[MisterPilot] Command timed out after {timeout}s\n" + result.stderr
            ).strip()

        return result


_instance: Optional[TerminalService] = None


def get_terminal_service() -> TerminalService:
    global _instance
    if _instance is None:
        _instance = TerminalService()
    return _instance
