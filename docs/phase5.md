# Phase 5 — Terminal Execution

## What was built

```
POST /terminal/stage    →  { id, command, workspace_root }  (no execution)
POST /terminal/execute  →  { stdout, stderr, exit_code, duration_ms, timed_out, error }
GET  /terminal/pending/{id}  →  staged command details
```

Commands only run after the user clicks **Allow** in a VS Code modal dialog.

---

## New files

| File | Role |
|------|------|
| `backend/services/terminal_service.py` | `TerminalService` — stage, execute, 2-min TTL |
| `backend/api/routes/terminal.py` | Three `/terminal/*` endpoints |
| `backend/tests/test_terminal.py` | 21 tests |
| `extension/webview/src/components/TerminalCard.tsx` | Terminal output card (pending/done/denied/error) |

## Modified files

| File | Change |
|------|--------|
| `backend/main.py` | Mounts `/terminal` router |
| `extension/src/types.ts` | Three new terminal message types |
| `extension/src/sidebarProvider.ts` | `_handleTerminalApproval()` — modal dialog + execute call |
| `extension/webview/src/types.ts` | `PendingTerminalData`, updated `ChatMessage` |
| `extension/webview/src/components/ChatPanel.tsx` | Handles `terminalPending/terminalResult/terminalError` |
| `extension/webview/src/components/MessageList.tsx` | Renders `TerminalCard` |
| `extension/webview/src/styles.css` | Terminal card styles |

---

## Architecture

### Approval flow

```
Agent (Phase 6) emits SSE: { "type": "pending_terminal", "id": "…", "command": "…" }
  │
  ▼
extension/_dispatchSsePayload  →  void _handleTerminalApproval(id, command, webview)
  │
  ├── webview.postMessage({ type: 'terminalPending' })   ← card appears as ⏳
  │
  ├── vscode.window.showWarningMessage(modal: true)       ← BLOCKS until user acts
  │       "MisterPilot wants to run:  $ npm install"
  │       [Allow]  [Deny]
  │
  ├── POST /terminal/execute  { id, approved: true/false }
  │
  └── webview.postMessage({ type: 'terminalResult' })    ← card shows output
```

The `{ modal: true }` flag makes the VS Code dialog block all editor interaction until dismissed — there is no way for the agent to bypass it.

### TerminalCard states

```
⏳ pending   → "Waiting for approval in VS Code…"

$ npm test                                     ✓
added 42 packages in 3.2s
exit 0 · 3200ms

$ bad_command                                  ✗
bash: bad_command: command not found
exit 127 · 18ms

$ rm -rf /                                    🚫
Command denied by user
```

### Security

| Mechanism | What it prevents |
|-----------|-----------------|
| `{ modal: true }` dialog | Silent background execution — user must see and approve every command |
| `cwd = workspace_root` | Commands start in the project directory (not `/` or `$HOME`) |
| `asyncio.wait_for(timeout=30)` | Hung commands; process is killed on timeout |
| `stdout/stderr` capped at 50 KB each | Memory exhaustion from runaway output |
| Two-phase stage/execute | The agent cannot execute without a round-trip through the extension |
| Pending TTL = 2 min | Stale approvals cannot be replayed |

---

## How to run & test

```bash
cd MisterPilot && source .venv/bin/activate
cd backend && pytest -v tests/test_terminal.py

# Smoke test (backend running, terminal in workspace):
ROOT=$(pwd)

ID=$(curl -s -X POST http://localhost:8000/terminal/stage \
  -H "Content-Type: application/json" \
  -d "{\"command\":\"echo hello\",\"workspace_root\":\"$ROOT\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -s -X POST http://localhost:8000/terminal/execute \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"$ID\",\"approved\":true}" | python3 -m json.tool
```

---

## Test count

| File | Count |
|------|-------|
| `test_terminal.py` | 21 |
| `test_edit.py` | 24 |
| `test_search.py` | 21 |
| `test_workspace.py` | 10 |
| `test_file_tools.py` | 14 |
| `test_chat.py` | 4 |
| **Total** | **94** |

---

## Next phase

**Phase 6** — ReAct agent loop:
- Tool registry: `read_file`, `search_code`, `write_file`, `replace_in_file`, `execute_terminal`
- Agent runs tool calls automatically, injects results, continues until final answer
- File edits and terminal commands emit `pending_edit` / `pending_terminal` SSE events mid-stream
- Extension intercepts and shows approval UI before execution
