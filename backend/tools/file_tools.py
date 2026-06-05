from pathlib import Path

# 1 MB — refuse to read larger files to protect context window and memory
MAX_READ_BYTES = 1_000_000


class PathTraversalError(PermissionError):
    """Raised when a requested path escapes the workspace root."""


def _safe_resolve(root: str, path: str) -> Path:
    """
    Resolve `path` relative to `root` and guarantee the result stays
    inside `root`. Raises PathTraversalError otherwise.
    """
    root_path = Path(root).resolve()

    # Reject absolute paths — they would bypass workspace containment
    if Path(path).is_absolute():
        raise PathTraversalError(f"Absolute paths are not allowed: {path!r}")

    # Reject '..' components before any resolution attempt
    clean = Path(path)
    if ".." in clean.parts:
        raise PathTraversalError(f"Path traversal not allowed: {path!r}")

    resolved = (root_path / clean.as_posix().lstrip("/\\")).resolve()

    try:
        resolved.relative_to(root_path)
    except ValueError:
        raise PathTraversalError(
            f"Path {path!r} resolves outside workspace root {root!r}"
        )

    return resolved


def read_file(root: str, path: str) -> str:
    """
    Read and return the text content of `path` (relative to `root`).

    Raises:
        PathTraversalError  — path escapes root
        FileNotFoundError   — file does not exist
        IsADirectoryError   — path points to a directory
        ValueError          — file exceeds MAX_READ_BYTES
    """
    resolved = _safe_resolve(root, path)

    if not resolved.exists():
        raise FileNotFoundError(f"No such file: {path!r}")

    if resolved.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path!r}")

    size = resolved.stat().st_size
    if size > MAX_READ_BYTES:
        raise ValueError(
            f"File too large to read ({size:,} bytes > {MAX_READ_BYTES:,} limit): {path!r}"
        )

    # errors='replace' so binary-ish files never crash the endpoint
    return resolved.read_text(encoding="utf-8", errors="replace")


def write_file(root: str, path: str, content: str) -> None:
    """
    Write content to path (relative to root).
    Parent directories are created if they do not exist.
    Kept here for Phase 4 — not exposed via HTTP until then.
    """
    resolved = _safe_resolve(root, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")


def replace_in_file(root: str, path: str, old_text: str, new_text: str) -> str:
    """
    Replace the first occurrence of old_text with new_text in the file.
    Returns the new full content.
    Kept here for Phase 4 — not exposed via HTTP until then.
    """
    current = read_file(root, path)
    if old_text not in current:
        raise ValueError(f"Text to replace not found in {path!r}")
    updated = current.replace(old_text, new_text, 1)
    write_file(root, path, updated)
    return updated
