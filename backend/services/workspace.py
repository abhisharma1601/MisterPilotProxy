import os
from pathlib import Path
from typing import List

# Directories to skip entirely during traversal
_IGNORE_DIRS: frozenset[str] = frozenset([
    "node_modules", "target", "dist", "build",
    ".git", ".idea", ".vscode",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env",
    "coverage", ".tox", ".eggs", "*.egg-info",
])

# File extensions that are never useful to read as text
_IGNORE_EXTENSIONS: frozenset[str] = frozenset([
    ".pyc", ".pyo", ".pyd",
    ".so", ".dll", ".dylib", ".exe", ".obj", ".class", ".jar",
    ".bin", ".dat",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pdf",
])


def _should_skip_dir(name: str) -> bool:
    return name in _IGNORE_DIRS or (name.startswith(".") and name not in (".",))


def _should_skip_file(name: str) -> bool:
    _, ext = os.path.splitext(name)
    return ext.lower() in _IGNORE_EXTENSIONS


def list_files(root: str) -> List[str]:
    """
    Walk root recursively and return a sorted list of file paths
    relative to root, skipping ignored directories and extensions.
    """
    root_path = Path(root).resolve()

    if not root_path.is_dir():
        raise NotADirectoryError(f"Workspace root is not a directory: {root}")

    results: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        dirnames[:] = sorted(d for d in dirnames if not _should_skip_dir(d))

        for name in sorted(filenames):
            if _should_skip_file(name):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(root_path)
            results.append(str(rel))

    return results
