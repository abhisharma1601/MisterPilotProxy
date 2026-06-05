import os
import pytest
from pathlib import Path

from backend.services.workspace import list_files


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A small synthetic workspace tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass")
    (tmp_path / "README.md").write_text("# Project")
    (tmp_path / "package.json").write_text("{}")
    return tmp_path


def test_lists_source_files(workspace: Path):
    files = list_files(str(workspace))
    assert "src/main.py" in files
    assert "src/utils.py" in files
    assert "README.md" in files
    assert "package.json" in files


def test_result_is_sorted(workspace: Path):
    files = list_files(str(workspace))
    assert files == sorted(files)


def test_ignores_node_modules(workspace: Path):
    nm = workspace / "node_modules" / "lodash"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}")

    files = list_files(str(workspace))
    assert not any("node_modules" in f for f in files)


def test_ignores_dot_git(workspace: Path):
    git = workspace / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main")

    files = list_files(str(workspace))
    assert not any(".git" in f for f in files)


def test_ignores_dist_and_build(workspace: Path):
    for dirname in ("dist", "build", "target"):
        d = workspace / dirname
        d.mkdir()
        (d / "bundle.js").write_text("(function(){})()")

    files = list_files(str(workspace))
    assert not any(f.startswith(("dist/", "build/", "target/")) for f in files)


def test_ignores_binary_extensions(workspace: Path):
    (workspace / "icon.png").write_bytes(b"\x89PNG\r\n")
    (workspace / "font.woff2").write_bytes(b"wOFF")
    (workspace / "archive.zip").write_bytes(b"PK")

    files = list_files(str(workspace))
    assert "icon.png" not in files
    assert "font.woff2" not in files
    assert "archive.zip" not in files


def test_ignores_pycache(workspace: Path):
    pc = workspace / "__pycache__"
    pc.mkdir()
    (pc / "main.cpython-312.pyc").write_bytes(b"\x00")

    files = list_files(str(workspace))
    assert not any("__pycache__" in f for f in files)


def test_raises_on_missing_root():
    with pytest.raises(NotADirectoryError):
        list_files("/nonexistent/path/that/does/not/exist")


def test_returns_paths_relative_to_root(workspace: Path):
    files = list_files(str(workspace))
    for f in files:
        assert not os.path.isabs(f), f"Expected relative path, got: {f}"


@pytest.mark.asyncio
async def test_workspace_files_endpoint(workspace: Path):
    """HTTP integration: GET /workspace/files returns expected structure."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/workspace/files", params={"root": str(workspace)})

    assert r.status_code == 200
    body = r.json()
    assert "files" in body
    assert "total" in body
    assert body["total"] == len(body["files"])
    assert "src/main.py" in body["files"]


@pytest.mark.asyncio
async def test_workspace_files_bad_root():
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/workspace/files", params={"root": "/no/such/directory"})

    assert r.status_code == 400
