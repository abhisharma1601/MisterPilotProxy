import asyncio
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..services.workspace import list_files
from ..tools.file_tools import MAX_READ_BYTES


@dataclass
class SearchMatch:
    file: str     # path relative to workspace root
    line: int     # 1-based line number
    content: str  # the matching line (rstripped)


def _ripgrep_available() -> bool:
    return shutil.which("rg") is not None


async def search_code(
    root: str,
    query: str,
    case_sensitive: bool = False,
    use_regex: bool = False,
    file_pattern: Optional[str] = None,
    max_results: int = 200,
) -> tuple[List[SearchMatch], str]:
    """
    Search workspace files for query.
    Returns (matches, engine) where engine is "ripgrep" or "python".
    Tries ripgrep first; falls back to Python if rg is not installed.
    """
    if _ripgrep_available():
        try:
            matches = await _rg_search(root, query, case_sensitive, use_regex, file_pattern, max_results)
            return matches, "ripgrep"
        except Exception:
            pass  # fall through to Python

    matches = await asyncio.get_event_loop().run_in_executor(
        None,
        _python_search_sync,
        root, query, case_sensitive, use_regex, file_pattern, max_results,
    )
    return matches, "python"


# ── ripgrep backend ────────────────────────────────────────────────────────────

async def _rg_search(
    root: str,
    query: str,
    case_sensitive: bool,
    use_regex: bool,
    file_pattern: Optional[str],
    max_results: int,
) -> List[SearchMatch]:
    cmd = ["rg", "--json", "--line-number"]

    if not case_sensitive:
        cmd.append("--ignore-case")
    if not use_regex:
        cmd.append("--fixed-strings")   # literal search
    if file_pattern:
        cmd.extend(["--glob", file_pattern])

    cmd += ["--", query, root]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()

    root_path = Path(root).resolve()
    matches: List[SearchMatch] = []

    for raw in stdout.decode(errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if obj.get("type") != "match":
            continue

        data = obj["data"]
        abs_path = data["path"]["text"]

        try:
            rel = str(Path(abs_path).relative_to(root_path))
        except ValueError:
            rel = abs_path

        matches.append(
            SearchMatch(
                file=rel,
                line=data["line_number"],
                content=data["lines"]["text"].rstrip("\n\r"),
            )
        )

        if len(matches) >= max_results:
            break

    return matches


# ── pure-Python fallback ───────────────────────────────────────────────────────

def _glob_matches(rel_path: str, pattern: str) -> bool:
    """True if the filename or full relative path matches a glob pattern."""
    p = Path(rel_path)
    return p.match(pattern) or p.name == pattern


def _python_search_sync(
    root: str,
    query: str,
    case_sensitive: bool,
    use_regex: bool,
    file_pattern: Optional[str],
    max_results: int,
) -> List[SearchMatch]:
    flags = 0 if case_sensitive else re.IGNORECASE

    if use_regex:
        try:
            pattern = re.compile(query, flags)
        except re.error:
            # Invalid regex — fall back to literal
            pattern = re.compile(re.escape(query), flags)
    else:
        pattern = re.compile(re.escape(query), flags)

    root_path = Path(root).resolve()
    files = list_files(root)
    matches: List[SearchMatch] = []

    for rel_file in files:
        if file_pattern and not _glob_matches(rel_file, file_pattern):
            continue

        full = root_path / rel_file

        try:
            if full.stat().st_size > MAX_READ_BYTES:
                continue
            text = full.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        for line_num, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(
                    SearchMatch(file=rel_file, line=line_num, content=line.rstrip())
                )
                if len(matches) >= max_results:
                    return matches

    return matches
