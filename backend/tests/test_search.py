import shutil
import pytest
from pathlib import Path

from backend.tools.search_tools import (
    search_code,
    _python_search_sync,
    _rg_search,
    SearchMatch,
)

RIPGREP = shutil.which("rg") is not None


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    print('hello world')\n\nmain()\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def helper(x):\n    return x * 2\n\nHELPER_CONST = 42\n"
    )
    (tmp_path / "README.md").write_text(
        "# My Project\n\nThis project says hello world.\n"
    )
    (tmp_path / "config.json").write_text('{"debug": true, "port": 8000}')
    # Binary file — should not cause crashes
    (tmp_path / "icon.pyc").write_bytes(b"\x00\x01\x02")
    return tmp_path


# ── Python fallback unit tests ────────────────────────────────────────────────

def test_python_finds_literal_match(workspace: Path):
    results = _python_search_sync(str(workspace), "hello world", False, False, None, 100)
    files = {r.file for r in results}
    assert "src/main.py" in files
    assert "README.md" in files


def test_python_case_insensitive(workspace: Path):
    results = _python_search_sync(str(workspace), "HELPER", False, False, None, 100)
    files = {r.file for r in results}
    assert "src/utils.py" in files


def test_python_case_sensitive(workspace: Path):
    results = _python_search_sync(str(workspace), "HELPER", True, False, None, 100)
    # Only the HELPER_CONST line should match with case-sensitive, not "helper"
    for r in results:
        assert "HELPER" in r.content


def test_python_returns_correct_line_numbers(workspace: Path):
    results = _python_search_sync(str(workspace), "print", False, False, None, 100)
    assert any(r.file == "src/main.py" and r.line == 2 for r in results)


def test_python_file_pattern_filter(workspace: Path):
    results = _python_search_sync(str(workspace), "hello", False, False, "*.py", 100)
    for r in results:
        assert r.file.endswith(".py"), f"Expected .py file, got {r.file}"
    files = {r.file for r in results}
    assert "README.md" not in files


def test_python_max_results_respected(workspace: Path):
    results = _python_search_sync(str(workspace), "e", False, False, None, 3)
    assert len(results) <= 3


def test_python_regex_mode(workspace: Path):
    results = _python_search_sync(str(workspace), r"def \w+\(", False, True, None, 100)
    assert any(r.file == "src/main.py" for r in results)
    assert any(r.file == "src/utils.py" for r in results)


def test_python_invalid_regex_falls_back_to_literal(workspace: Path):
    # Should not raise even with a broken regex pattern
    results = _python_search_sync(str(workspace), "[invalid(", False, True, None, 100)
    assert isinstance(results, list)


def test_python_no_match_returns_empty(workspace: Path):
    results = _python_search_sync(str(workspace), "xyzzy_no_match_ever", False, False, None, 100)
    assert results == []


def test_python_binary_file_skipped(workspace: Path):
    # .pyc is in IGNORE_EXTENSIONS so it won't be in list_files at all — no crash
    results = _python_search_sync(str(workspace), "main", False, False, None, 100)
    assert not any(r.file.endswith(".pyc") for r in results)


# ── async search_code (engine selection) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_search_code_returns_matches(workspace: Path):
    matches, engine = await search_code(str(workspace), "hello world")
    assert len(matches) >= 2   # src/main.py and README.md
    assert engine in ("ripgrep", "python")


@pytest.mark.asyncio
async def test_search_code_empty_workspace(tmp_path: Path):
    matches, _ = await search_code(str(tmp_path), "anything")
    assert matches == []


# ── ripgrep-specific tests (skip if not installed) ────────────────────────────

@pytest.mark.skipif(not RIPGREP, reason="ripgrep not installed")
@pytest.mark.asyncio
async def test_rg_finds_matches(workspace: Path):
    results = await _rg_search(str(workspace), "hello world", False, False, None, 100)
    files = {r.file for r in results}
    assert "src/main.py" in files


@pytest.mark.skipif(not RIPGREP, reason="ripgrep not installed")
@pytest.mark.asyncio
async def test_rg_file_pattern(workspace: Path):
    results = await _rg_search(str(workspace), "hello", False, False, "*.py", 100)
    for r in results:
        assert r.file.endswith(".py")


@pytest.mark.skipif(not RIPGREP, reason="ripgrep not installed")
@pytest.mark.asyncio
async def test_rg_returns_correct_line_number(workspace: Path):
    results = await _rg_search(str(workspace), "print", False, False, None, 100)
    hit = next((r for r in results if r.file == "src/main.py"), None)
    assert hit is not None
    assert hit.line == 2


# ── HTTP endpoint tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_endpoint_ok(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/workspace/search",
            json={"root": str(workspace), "query": "hello world"},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    assert body["engine"] in ("ripgrep", "python")
    assert body["elapsed_ms"] >= 0
    assert all(k in body["matches"][0] for k in ("file", "line", "content"))


@pytest.mark.asyncio
async def test_search_endpoint_empty_query(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/workspace/search",
            json={"root": str(workspace), "query": "   "},
        )

    assert r.status_code == 422


@pytest.mark.asyncio
async def test_search_endpoint_no_match(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/workspace/search",
            json={"root": str(workspace), "query": "xyzzy_no_match"},
        )

    assert r.status_code == 200
    assert r.json()["total"] == 0


@pytest.mark.asyncio
async def test_search_endpoint_file_pattern(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/workspace/search",
            json={"root": str(workspace), "query": "hello", "file_pattern": "*.py"},
        )

    body = r.json()
    assert r.status_code == 200
    for match in body["matches"]:
        assert match["file"].endswith(".py")


@pytest.mark.asyncio
async def test_search_endpoint_max_results(workspace: Path):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/workspace/search",
            json={"root": str(workspace), "query": "e", "max_results": 2},
        )

    body = r.json()
    assert r.status_code == 200
    assert body["total"] <= 2
