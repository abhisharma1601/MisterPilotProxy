import time
import pytest
from pathlib import Path

from backend.services.edit_service import EditService, get_edit_service
from backend.tools.file_tools import PathTraversalError


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def svc() -> EditService:
    return EditService()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "util.py").write_text("X = 1\n")
    return tmp_path


# ── EditService unit tests ────────────────────────────────────────────────────

def test_preview_write_new_file(svc: EditService, workspace: Path):
    edit = svc.preview_write(str(workspace), "new_file.py", "print('new')\n")
    assert edit.is_new_file is True
    assert edit.original == ""
    assert edit.proposed == "print('new')\n"
    assert edit.path == "new_file.py"
    assert "+" in edit.diff()


def test_preview_write_existing_file(svc: EditService, workspace: Path):
    new_content = "def main():\n    print('hello world')\n"
    edit = svc.preview_write(str(workspace), "main.py", new_content)
    assert edit.is_new_file is False
    assert "hello world" in edit.diff()
    assert "-    print('hello')" in edit.diff()


def test_preview_write_identical_raises(svc: EditService, workspace: Path):
    original = (workspace / "main.py").read_text()
    with pytest.raises(ValueError, match="No changes detected"):
        svc.preview_write(str(workspace), "main.py", original)


def test_preview_write_traversal_rejected(svc: EditService, workspace: Path):
    with pytest.raises(PathTraversalError):
        svc.preview_write(str(workspace), "../../etc/shadow", "bad")


def test_preview_replace_ok(svc: EditService, workspace: Path):
    edit = svc.preview_replace(str(workspace), "main.py", "hello", "hello world")
    assert "hello world" in edit.proposed
    assert edit.is_new_file is False


def test_preview_replace_not_found(svc: EditService, workspace: Path):
    with pytest.raises(ValueError, match="not found"):
        svc.preview_replace(str(workspace), "main.py", "DOES NOT EXIST", "x")


def test_preview_replace_file_not_found(svc: EditService, workspace: Path):
    with pytest.raises(FileNotFoundError):
        svc.preview_replace(str(workspace), "ghost.py", "x", "y")


def test_preview_replace_traversal_rejected(svc: EditService, workspace: Path):
    with pytest.raises(PathTraversalError):
        svc.preview_replace(str(workspace), "../../secret", "a", "b")


def test_apply_writes_to_disk(svc: EditService, workspace: Path):
    edit = svc.preview_write(str(workspace), "main.py", "# rewritten\n")
    svc.apply(edit.id)
    assert (workspace / "main.py").read_text() == "# rewritten\n"


def test_apply_removes_from_pending(svc: EditService, workspace: Path):
    edit = svc.preview_write(str(workspace), "main.py", "# new\n")
    svc.apply(edit.id)
    assert svc.get(edit.id) is None


def test_apply_not_found_raises(svc: EditService):
    with pytest.raises(KeyError):
        svc.apply("nonexistent-id")


def test_reject_does_not_write(svc: EditService, workspace: Path):
    original = (workspace / "main.py").read_text()
    edit = svc.preview_write(str(workspace), "main.py", "# changed\n")
    svc.reject(edit.id)
    assert (workspace / "main.py").read_text() == original


def test_reject_removes_from_pending(svc: EditService, workspace: Path):
    edit = svc.preview_write(str(workspace), "main.py", "# changed\n")
    svc.reject(edit.id)
    assert svc.get(edit.id) is None


def test_reject_not_found_raises(svc: EditService):
    with pytest.raises(KeyError):
        svc.reject("nonexistent-id")


def test_diff_format_unified(svc: EditService, workspace: Path):
    edit = svc.preview_write(str(workspace), "main.py", "def main():\n    print('hi')\n")
    diff = edit.diff()
    assert diff.startswith("---")
    assert "+++" in diff
    assert "@@" in diff


def test_replace_only_first_occurrence(svc: EditService, workspace: Path):
    (workspace / "repeat.py").write_text("aaa\naaa\n")
    edit = svc.preview_replace(str(workspace), "repeat.py", "aaa", "bbb")
    lines = edit.proposed.splitlines()
    assert lines[0] == "bbb"
    assert lines[1] == "aaa"


# ── HTTP endpoint integration tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_preview_write_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/edit/preview/write",
            json={"root": str(workspace), "path": "main.py", "content": "# new\n"},
        )

    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert "diff" in body
    assert body["is_new_file"] is False


@pytest.mark.asyncio
async def test_preview_write_new_file_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/edit/preview/write",
            json={"root": str(workspace), "path": "brand_new.py", "content": "x = 1\n"},
        )

    assert r.status_code == 200
    assert r.json()["is_new_file"] is True


@pytest.mark.asyncio
async def test_preview_write_identical_422(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    original = (workspace / "main.py").read_text()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/edit/preview/write",
            json={"root": str(workspace), "path": "main.py", "content": original},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_preview_replace_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/edit/preview/replace",
            json={"root": str(workspace), "path": "main.py", "old_text": "hello", "new_text": "hi"},
        )

    assert r.status_code == 200
    body = r.json()
    assert "hi" in body["proposed"]


@pytest.mark.asyncio
async def test_preview_traversal_403(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/edit/preview/write",
            json={"root": str(workspace), "path": "../../etc/passwd", "content": "bad"},
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_apply_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    svc = EditService()
    edit = svc.preview_write(str(workspace), "main.py", "# applied\n")

    # Patch singleton so the HTTP app uses our service instance with the staged edit
    import backend.services.edit_service as _m
    _orig = _m._instance
    _m._instance = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/edit/apply", json={"id": edit.id})
    finally:
        _m._instance = _orig

    assert r.status_code == 200
    assert r.json()["applied"] is True
    assert (workspace / "main.py").read_text() == "# applied\n"


@pytest.mark.asyncio
async def test_apply_unknown_id_404():
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/edit/apply", json={"id": "does-not-exist"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_reject_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    svc = EditService()
    edit = svc.preview_write(str(workspace), "main.py", "# rejected\n")

    import backend.services.edit_service as _m
    _orig = _m._instance
    _m._instance = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/edit/reject", json={"id": edit.id})
    finally:
        _m._instance = _orig

    assert r.status_code == 200
    assert r.json()["rejected"] is True
    # File unchanged
    assert "hello" in (workspace / "main.py").read_text()


@pytest.mark.asyncio
async def test_get_pending_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    svc = EditService()
    edit = svc.preview_write(str(workspace), "main.py", "# peeking\n")

    import backend.services.edit_service as _m
    _orig = _m._instance
    _m._instance = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(f"/edit/pending/{edit.id}")
    finally:
        _m._instance = _orig

    assert r.status_code == 200
    assert r.json()["id"] == edit.id
