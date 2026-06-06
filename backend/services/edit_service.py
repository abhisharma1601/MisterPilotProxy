import difflib
import time
import uuid
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Dict, Optional

from ..tools.file_tools import (
    _safe_resolve,
    read_file,
    write_file,
    BLOCKED_FILENAME_PATTERNS,
    BlockedFileError,
    PathTraversalError,
)

_PENDING_TTL_SECONDS = 300  # pending edits expire after 5 minutes


@dataclass
class PendingEdit:
    id: str
    root: str
    path: str
    original: str      # current content; empty string for new files
    proposed: str      # content after the edit
    is_new_file: bool
    created_at: float = field(default_factory=time.monotonic)

    def diff(self) -> str:
        orig = self.original.splitlines(keepends=True)
        prop = self.proposed.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                orig,
                prop,
                fromfile=f"a/{self.path}" if not self.is_new_file else "/dev/null",
                tofile=f"b/{self.path}",
                lineterm="",
            )
        )

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > _PENDING_TTL_SECONDS


class EditService:
    def __init__(self) -> None:
        self._pending: Dict[str, PendingEdit] = {}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _prune_expired(self) -> None:
        for k in [k for k, v in self._pending.items() if v.is_expired()]:
            del self._pending[k]

    def _store(self, edit: PendingEdit) -> PendingEdit:
        self._prune_expired()
        self._pending[edit.id] = edit
        return edit

    # ── preview operations ────────────────────────────────────────────────────

    def preview_write(self, root: str, path: str, content: str) -> PendingEdit:
        """
        Stage a write_file operation. Returns a PendingEdit with diff.
        Does NOT touch the filesystem.

        Raises:
            PathTraversalError  — path escapes root
            ValueError          — content is identical to current file
        """
        resolved = _safe_resolve(root, path)  # raises PathTraversalError if unsafe

        name_lower = resolved.name.lower()
        if any(fnmatch(name_lower, pat.lower()) for pat in BLOCKED_FILENAME_PATTERNS):
            raise BlockedFileError(
                f"Writing {path!r} is blocked — this file may contain secrets."
            )

        try:
            original = read_file(root, path)
            is_new = False
        except FileNotFoundError:
            original = ""
            is_new = True

        if not is_new and original == content:
            raise ValueError(f"No changes detected in {path!r}")

        return self._store(
            PendingEdit(
                id=str(uuid.uuid4()),
                root=root,
                path=path,
                original=original,
                proposed=content,
                is_new_file=is_new,
            )
        )

    def preview_replace(
        self, root: str, path: str, old_text: str, new_text: str
    ) -> PendingEdit:
        """
        Stage a replace_in_file operation. Returns a PendingEdit with diff.
        Does NOT touch the filesystem.

        Raises:
            PathTraversalError  — path escapes root
            FileNotFoundError   — file does not exist
            ValueError          — old_text not found in file
        """
        resolved = _safe_resolve(root, path)

        name_lower = resolved.name.lower()
        if any(fnmatch(name_lower, pat.lower()) for pat in BLOCKED_FILENAME_PATTERNS):
            raise BlockedFileError(
                f"Writing {path!r} is blocked — this file may contain secrets."
            )

        original = read_file(root, path)  # may raise FileNotFoundError

        if old_text not in original:
            raise ValueError(f"Text to replace not found in {path!r}")

        proposed = original.replace(old_text, new_text, 1)

        return self._store(
            PendingEdit(
                id=str(uuid.uuid4()),
                root=root,
                path=path,
                original=original,
                proposed=proposed,
                is_new_file=False,
            )
        )

    # ── apply / reject ────────────────────────────────────────────────────────

    def apply(self, edit_id: str) -> PendingEdit:
        """Write the proposed content to disk and remove from pending store."""
        edit = self._pending.get(edit_id)
        if edit is None:
            raise KeyError(f"Pending edit not found: {edit_id!r}")
        if edit.is_expired():
            del self._pending[edit_id]
            raise TimeoutError(f"Pending edit expired: {edit_id!r}")

        write_file(edit.root, edit.path, edit.proposed)
        del self._pending[edit_id]
        return edit

    def reject(self, edit_id: str) -> PendingEdit:
        """Discard a pending edit without touching the filesystem."""
        edit = self._pending.pop(edit_id, None)
        if edit is None:
            raise KeyError(f"Pending edit not found: {edit_id!r}")
        return edit

    def get(self, edit_id: str) -> Optional[PendingEdit]:
        return self._pending.get(edit_id)


_instance: Optional[EditService] = None


def get_edit_service() -> EditService:
    global _instance
    if _instance is None:
        _instance = EditService()
    return _instance
