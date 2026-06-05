import pytest
from pathlib import Path

from backend.tools.file_tools import read_file, write_file, replace_in_file, PathTraversalError


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.py").write_text("print('hello')")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested content")
    return tmp_path


# ── read_file ─────────────────────────────────────────────────────────────────


def test_read_file_simple(workspace: Path):
    content = read_file(str(workspace), "hello.py")
    assert content == "print('hello')"


def test_read_file_nested(workspace: Path):
    content = read_file(str(workspace), "sub/nested.txt")
    assert content == "nested content"


def test_read_file_not_found(workspace: Path):
    with pytest.raises(FileNotFoundError):
        read_file(str(workspace), "does_not_exist.py")


def test_read_file_directory(workspace: Path):
    with pytest.raises(IsADirectoryError):
        read_file(str(workspace), "sub")


def test_read_file_rejects_dotdot(workspace: Path):
    with pytest.raises(PathTraversalError):
        read_file(str(workspace), "../etc/passwd")


def test_read_file_rejects_dotdot_in_middle(workspace: Path):
    with pytest.raises(PathTraversalError):
        read_file(str(workspace), "sub/../../etc/passwd")


def test_read_file_rejects_absolute_escape(workspace: Path):
    """An absolute path that resolves outside root must be rejected."""
    with pytest.raises(PathTraversalError):
        read_file(str(workspace), "/etc/passwd")


def test_read_file_too_large(workspace: Path):
    big = workspace / "big.bin"
    big.write_bytes(b"x" * (1_000_001))
    with pytest.raises(ValueError, match="too large"):
        read_file(str(workspace), "big.bin")


def test_read_file_binary_replaced(workspace: Path):
    """Binary content should not crash; replacement chars are returned."""
    (workspace / "binary.bin").write_bytes(b"\xff\xfe binary data")
    content = read_file(str(workspace), "binary.bin")
    assert isinstance(content, str)


# ── write_file ────────────────────────────────────────────────────────────────


def test_write_file_creates_new(workspace: Path):
    write_file(str(workspace), "new_file.txt", "created")
    assert (workspace / "new_file.txt").read_text() == "created"


def test_write_file_creates_parents(workspace: Path):
    write_file(str(workspace), "deep/dir/file.txt", "deep")
    assert (workspace / "deep" / "dir" / "file.txt").read_text() == "deep"


def test_write_file_rejects_traversal(workspace: Path):
    with pytest.raises(PathTraversalError):
        write_file(str(workspace), "../outside.txt", "bad")


# ── replace_in_file ───────────────────────────────────────────────────────────


def test_replace_in_file(workspace: Path):
    result = replace_in_file(str(workspace), "hello.py", "hello", "world")
    assert "world" in result
    assert (workspace / "hello.py").read_text() == "print('world')"


def test_replace_in_file_not_found_text(workspace: Path):
    with pytest.raises(ValueError, match="not found"):
        replace_in_file(str(workspace), "hello.py", "NONEXISTENT", "x")


def test_replace_only_first_occurrence(workspace: Path):
    (workspace / "repeat.txt").write_text("aaa")
    replace_in_file(str(workspace), "repeat.txt", "a", "b")
    assert (workspace / "repeat.txt").read_text() == "baa"


# ── HTTP endpoint tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_file_endpoint(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/workspace/file",
            params={"root": str(workspace), "path": "hello.py"},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "print('hello')"
    assert body["path"] == "hello.py"
    assert body["size"] > 0


@pytest.mark.asyncio
async def test_get_file_traversal_blocked(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/workspace/file",
            params={"root": str(workspace), "path": "../../etc/passwd"},
        )

    assert r.status_code == 403


@pytest.mark.asyncio
async def test_get_file_not_found(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/workspace/file",
            params={"root": str(workspace), "path": "ghost.py"},
        )

    assert r.status_code == 404
