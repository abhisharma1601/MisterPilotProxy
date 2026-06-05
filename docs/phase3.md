# Phase 3 — Search Tool

## What was built

```
POST /workspace/search
Body: { root, query, case_sensitive?, use_regex?, file_pattern?, max_results? }
Response: { query, matches:[{file, line, content}], total, engine, elapsed_ms }
```

Tries **ripgrep** first; falls back to **pure Python** if `rg` is not in PATH.

---

## New files

| File | Role |
|------|------|
| `backend/tools/search_tools.py` | `search_code()` + ripgrep and Python backends |
| `backend/tests/test_search.py` | 21 tests covering both engines and the HTTP endpoint |

## Modified files

| File | Change |
|------|--------|
| `backend/api/routes/workspace.py` | Added `POST /workspace/search` |

---

## Architecture

### `search_code(root, query, …)` — engine selection

```
search_code()
  ├── shutil.which("rg") found?
  │     └── YES → _rg_search()   (asyncio.create_subprocess_exec)
  │                  ↓ on exception
  │              _python_search_sync() in executor
  └── NO → _python_search_sync() in executor
```

Both return `List[SearchMatch]` and a string `"ripgrep"` or `"python"` so the
caller knows which engine ran (visible in the API response).

### ripgrep backend (`_rg_search`)

Command built:
```
rg --json --line-number [--ignore-case] [--fixed-strings] [--glob <pat>] -- <query> <root>
```

- `--json` → NDJSON output, one object per line
- `--fixed-strings` (default, `use_regex=False`) → literal string match, safe from regex injection
- `--ignore-case` (default, `case_sensitive=False`)

Output filtered to `type == "match"` objects; paths made relative to root.

### Python fallback (`_python_search_sync`)

1. `list_files(root)` — reuses Phase 2 workspace walker (ignore rules apply)
2. Apply optional `file_pattern` via `pathlib.Path.match()`
3. Skip files > `MAX_READ_BYTES` (1 MB)
4. `re.compile(re.escape(query))` for literal, or `re.compile(query)` for regex
5. Invalid regex silently downgrades to literal match
6. Runs in `asyncio.run_in_executor` so the event loop stays unblocked

### Request/Response schema

```json
// Request
{
  "root": "/abs/path/to/workspace",
  "query": "def main",
  "case_sensitive": false,
  "use_regex": false,
  "file_pattern": "*.py",
  "max_results": 200
}

// Response
{
  "query": "def main",
  "matches": [
    { "file": "src/main.py", "line": 1, "content": "def main():" }
  ],
  "total": 1,
  "engine": "ripgrep",
  "elapsed_ms": 3
}
```

---

## Security notes

- `use_regex=false` (default) passes `--fixed-strings` to ripgrep and `re.escape()` in Python — the query is treated as a literal string, preventing regex injection DoS via catastrophic backtracking
- When `use_regex=true`, an invalid regex pattern silently falls back to literal search in Python (and ripgrep handles its own validation)
- No file content is written; this endpoint is read-only

---

## How to run & test

```bash
# Install ripgrep (optional but recommended):
# Ubuntu/Debian:  sudo apt install ripgrep
# macOS:          brew install ripgrep

cd MisterPilot && source .venv/bin/activate
cd backend && pytest -v tests/test_search.py

# Smoke test:
curl -s -X POST http://localhost:8000/workspace/search \
  -H "Content-Type: application/json" \
  -d '{"root":"'$(pwd)'","query":"def ","file_pattern":"*.py"}' | python3 -m json.tool
```

Tests automatically skip ripgrep-specific cases if `rg` is not installed.

---

## Test count

| Test file | Count |
|-----------|-------|
| `test_search.py` | 21 |
| `test_workspace.py` | 10 |
| `test_file_tools.py` | 14 |
| `test_chat.py` | 4 |
| **Total** | **49** |

---

## Next phase

**Phase 4** adds:
- `write_file(path, content)` — creates or overwrites a file
- `replace_in_file(path, old_text, new_text)` — targeted edit
- Diff preview before any write (unified diff format)
- Confirmation required: the extension shows a VS Code diff editor before applying
