# Phase 4 — File Editing with Confirmation

## What was built

```
POST /edit/preview/write    →  { id, diff, original, proposed, is_new_file }
POST /edit/preview/replace  →  { id, diff, original, proposed, is_new_file }
POST /edit/apply            →  { applied, path }   (writes to disk)
POST /edit/reject           →  { rejected, path }  (discards, no write)
GET  /edit/pending/{id}     →  preview payload (for polling/re-display)
```

Nothing is written until the user clicks **Apply** in the webview DiffCard or calls `/edit/apply`.

---

## New files

| File | Role |
|------|------|
| `backend/services/edit_service.py` | `EditService` — stage, diff, apply, reject + 5-min TTL |
| `backend/api/routes/edit.py` | All five `/edit/*` HTTP endpoints |
| `backend/tests/test_edit.py` | 24 tests (unit + HTTP) |
| `extension/webview/src/components/DiffCard.tsx` | Diff viewer + Apply/Reject/Full-diff buttons |

## Modified files

| File | Change |
|------|--------|
| `backend/main.py` | Mounts `/edit` router |
| `extension/src/types.ts` | New message types for applyEdit, rejectEdit, viewDiff, pendingEdit, editApplied, editRejected, editError |
| `extension/src/sidebarProvider.ts` | Handles applyEdit/rejectEdit/viewDiff from webview; pending_edit SSE events from agent |
| `extension/webview/src/types.ts` | `PendingEditData`, updated `ChatMessage`, new message union types |
| `extension/webview/src/components/ChatPanel.tsx` | Edit action handlers; filters edit cards from LLM context |
| `extension/webview/src/components/MessageList.tsx` | Renders DiffCard when message has pendingEdit |
| `extension/webview/src/styles.css` | DiffCard, diff line coloring, resolved card states |

---

## Architecture

### Pending edit lifecycle

```
preview_write / preview_replace
  └─► EditService._store() → { id: uuid, original, proposed, created_at }
        │
        ├─► apply(id)  → write_file() → del _pending[id]
        └─► reject(id) → del _pending[id]   (no I/O)

Pending edits expire after 5 minutes (TimeoutError → HTTP 410).
```

### Diff generation

Uses Python's built-in `difflib.unified_diff`:
- New files: `fromfile="/dev/null"` so the diff clearly shows all lines added
- Replacements: first occurrence only (`str.replace(old, new, 1)`)
- Identical content → `ValueError` (HTTP 422) before a pending edit is created

### Security

- All paths go through `_safe_resolve()` (Phase 2) before staging
- Nothing is written at preview time — filesystem is only touched on `/edit/apply`
- Expired pending edits are pruned on each call; applying an expired edit returns HTTP 410

### Webview DiffCard

```
┌─────────────────────────────────────────┐
│ 📝 src/main.py                    [edit]│
├─────────────────────────────────────────┤
│ --- a/src/main.py                       │  ← grey header
│ +++ b/src/main.py                       │
│ @@ -1,2 +1,2 @@                         │  ← cyan hunk
│  def main():                            │  ← context
│ -    print('hello')                     │  ← red remove
│ +    print('hello world')               │  ← green add
├─────────────────────────────────────────┤
│ [✓ Apply] [✗ Reject]        [⊞ Full diff]│
└─────────────────────────────────────────┘
```

After resolution:
- Applied → green card `✅ Applied changes to src/main.py`
- Rejected → grey card `✗ Rejected edit to src/main.py`

**⊞ Full diff** opens VS Code's native diff editor (`vscode.diff` command) with the original and proposed as untitled documents — for full-screen review.

Edit cards are excluded from the LLM message history sent on the next chat turn.

---

## How to run & test

```bash
cd MisterPilot && source .venv/bin/activate
cd backend && pytest -v tests/test_edit.py

# Smoke test via curl (with backend running):
ROOT=$(pwd)

# 1. Stage a write
RESULT=$(curl -s -X POST http://localhost:8000/edit/preview/write \
  -H "Content-Type: application/json" \
  -d "{\"root\":\"$ROOT\",\"path\":\"test_file.py\",\"content\":\"x = 1\\n\"}")
echo $RESULT | python3 -m json.tool

ID=$(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 2. Apply it
curl -s -X POST http://localhost:8000/edit/apply \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"$ID\"}" | python3 -m json.tool
```

---

## Test count

| File | Count |
|------|-------|
| `test_edit.py` | 24 |
| `test_search.py` | 21 |
| `test_workspace.py` | 10 |
| `test_file_tools.py` | 14 |
| `test_chat.py` | 4 |
| **Total** | **73** |

---

## Next phase

**Phase 5** adds:
- `execute_terminal(command)` tool
- VS Code `showInformationMessage` approval popup before any execution
- `POST /terminal/run` endpoint (guarded by a one-time token from the extension)
- Captures `stdout`, `stderr`, `exit_code`
- Execution strictly sandboxed to the workspace directory
