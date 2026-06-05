import pytest
from pathlib import Path

from backend.services.terminal_service import TerminalService


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def svc() -> TerminalService:
    return TerminalService()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.py").write_text("print('hello')\n")
    return tmp_path


# ── stage ─────────────────────────────────────────────────────────────────────

def test_stage_creates_pending(svc: TerminalService, workspace: Path):
    cmd = svc.stage("echo hi", str(workspace))
    assert cmd.id
    assert cmd.command == "echo hi"
    assert cmd.workspace_root == str(workspace)
    assert svc.get(cmd.id) is not None


def test_stage_empty_command_raises(svc: TerminalService, workspace: Path):
    with pytest.raises(ValueError, match="empty"):
        svc.stage("   ", str(workspace))


def test_stage_bad_root_raises(svc: TerminalService):
    with pytest.raises(NotADirectoryError):
        svc.stage("echo hi", "/nonexistent/path/xyz")


def test_stage_strips_whitespace(svc: TerminalService, workspace: Path):
    cmd = svc.stage("  echo hi  ", str(workspace))
    assert cmd.command == "echo hi"


# ── execute — denied ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_denied_returns_empty(svc: TerminalService, workspace: Path):
    cmd = svc.stage("echo hi", str(workspace))
    result = await svc.execute(cmd.id, approved=False)
    assert result.approved is False
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.exit_code == -1


@pytest.mark.asyncio
async def test_execute_denied_removes_pending(svc: TerminalService, workspace: Path):
    cmd = svc.stage("echo hi", str(workspace))
    await svc.execute(cmd.id, approved=False)
    assert svc.get(cmd.id) is None


# ── execute — approved ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_approved_runs_command(svc: TerminalService, workspace: Path):
    cmd = svc.stage("echo hello_world", str(workspace))
    result = await svc.execute(cmd.id, approved=True)
    assert result.approved is True
    assert "hello_world" in result.stdout
    assert result.exit_code == 0
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_execute_captures_exit_code(svc: TerminalService, workspace: Path):
    cmd = svc.stage("exit 42", str(workspace))
    result = await svc.execute(cmd.id, approved=True)
    assert result.exit_code == 42


@pytest.mark.asyncio
async def test_execute_captures_stderr(svc: TerminalService, workspace: Path):
    cmd = svc.stage("echo error_text >&2", str(workspace))
    result = await svc.execute(cmd.id, approved=True)
    assert "error_text" in result.stderr


@pytest.mark.asyncio
async def test_execute_uses_workspace_cwd(svc: TerminalService, workspace: Path):
    cmd = svc.stage("pwd", str(workspace))
    result = await svc.execute(cmd.id, approved=True)
    # pwd output should be the workspace directory
    assert str(workspace) in result.stdout.strip()


@pytest.mark.asyncio
async def test_execute_not_found_raises(svc: TerminalService):
    with pytest.raises(KeyError):
        await svc.execute("nonexistent-id", approved=True)


@pytest.mark.asyncio
async def test_execute_consumes_pending(svc: TerminalService, workspace: Path):
    cmd = svc.stage("echo once", str(workspace))
    await svc.execute(cmd.id, approved=True)
    with pytest.raises(KeyError):
        await svc.execute(cmd.id, approved=True)   # already consumed


@pytest.mark.asyncio
async def test_execute_timeout(svc: TerminalService, workspace: Path):
    cmd = svc.stage("sleep 10", str(workspace))
    result = await svc.execute(cmd.id, approved=True, timeout=1)
    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()


@pytest.mark.asyncio
async def test_output_capped(svc: TerminalService, workspace: Path):
    # Generate > 50KB of output
    big_cmd = "python3 -c \"print('x' * 60000)\""
    cmd = svc.stage(big_cmd, str(workspace))
    result = await svc.execute(cmd.id, approved=True)
    assert len(result.stdout) <= 50_000


# ── HTTP endpoint tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/terminal/stage",
            json={"command": "echo hi", "workspace_root": str(workspace)},
        )

    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["command"] == "echo hi"


@pytest.mark.asyncio
async def test_stage_empty_command_422(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/terminal/stage",
            json={"command": "", "workspace_root": str(workspace)},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_execute_endpoint_denied(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    svc = TerminalService()
    cmd = svc.stage("echo hi", str(workspace))

    import backend.services.terminal_service as _m
    _orig = _m._instance
    _m._instance = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/terminal/execute", json={"id": cmd.id, "approved": False})
    finally:
        _m._instance = _orig

    assert r.status_code == 200
    assert r.json()["approved"] is False


@pytest.mark.asyncio
async def test_execute_endpoint_approved(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    svc = TerminalService()
    cmd = svc.stage("echo integration_test", str(workspace))

    import backend.services.terminal_service as _m
    _orig = _m._instance
    _m._instance = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/terminal/execute", json={"id": cmd.id, "approved": True})
    finally:
        _m._instance = _orig

    assert r.status_code == 200
    body = r.json()
    assert "integration_test" in body["stdout"]
    assert body["exit_code"] == 0


@pytest.mark.asyncio
async def test_execute_not_found_404():
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/terminal/execute", json={"id": "ghost", "approved": True})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_pending_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    svc = TerminalService()
    cmd = svc.stage("echo pending", str(workspace))

    import backend.services.terminal_service as _m
    _orig = _m._instance
    _m._instance = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(f"/terminal/pending/{cmd.id}")
    finally:
        _m._instance = _orig

    assert r.status_code == 200
    assert r.json()["id"] == cmd.id
