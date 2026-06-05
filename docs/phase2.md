# Phase 2 — Workspace Context & File Reading

## What was built

```
GET /workspace/files?root=<path>   →  { files: [...], total: N, root }
GET /workspace/file?root=<path>&path=<rel>  →  { content, path, size }

POST /chat/stream  (updated)
  body: { messages, workspace_root }
  →  system prompt now includes workspace path + file listing
```

Path-traversal protection on every file operation.

---

## New files

| File | Role |
|------|------|
| `backend/services/workspace.py` | `list_files(root)` — recursive walk with ignore rules |
| `backend/tools/file_tools.py` | `read_file`, `write_file`, `replace_in_file` — all path-safe |
| `backend/api/routes/workspace.py` | HTTP endpoints for `GET /workspace/files` and `GET /workspace/file` |
| `backend/tests/test_workspace.py` | 10 tests for workspace listing and HTTP endpoint |
| `backend/tests/test_file_tools.py` | 14 tests covering reads, writes, replacements, traversal rejection |

---

## Ignore rules (`services/workspace.py`)

**Directories skipped:**
`node_modules`, `target`, `dist`, `build`, `.git`, `.idea`, `.vscode`,
`__pycache__`, `.pytest_cache`, `.mypy_cache`, `venv`, `.venv`, `env`, `.env`,
`coverage`, `.tox`, `.eggs`

**File extensions skipped:**
`.pyc`, `.pyo`, `.so`, `.dll`, `.exe`, `.bin`, `.png`, `.jpg`, `.gif`,
`.ico`, `.woff`, `.woff2`, `.ttf`, `.zip`, `.tar`, `.gz`, `.mp3`, `.mp4`,
`.pdf`, and more (see source for full list)

---

## Security: path traversal protection (`tools/file_tools.py`)

Every file operation goes through `_safe_resolve(root, path)`:

1. Reject any path containing `..` components before resolution
2. Strip leading `/` or `\` so joinpath never treats input as absolute
3. `Path.resolve()` to canonicalise symlinks
4. Call `.relative_to(root_path)` — raises `PathTraversalError` if outside root

Result: `../../etc/passwd`, `/etc/passwd`, `sub/../../../secret` all return HTTP 403.

---

## Workspace context in chat

When the extension sends `workspace_root` in the chat request body, the backend:

1. Builds a system message including the workspace path
2. Lists workspace files (capped at 500 entries to stay within context limits)
3. Prepends this as the `system` role message so the LLM knows what files exist

The LLM can then say "I can see you have `src/utils.py`…" without any extra work.

---

## How to run & test

```bash
# From MisterPilot/ with .venv active:
cd backend && pytest -v

# Smoke test the new endpoints:
ROOT=$(pwd)   # or any project directory
curl "http://localhost:8000/workspace/files?root=$ROOT"
curl "http://localhost:8000/workspace/file?root=$ROOT&path=README.md"
```

---

## Running tests

```
backend/tests/test_workspace.py     10 tests
backend/tests/test_file_tools.py    14 tests
backend/tests/test_chat.py          4 tests  (unchanged from Phase 1)
```

---

## Next phase

**Phase 3** adds:
- `search_code(query)` tool using ripgrep
- `POST /workspace/search` endpoint
- Returns: file, line number, matched content
- Falls back to pure-Python grep if ripgrep is not installed
