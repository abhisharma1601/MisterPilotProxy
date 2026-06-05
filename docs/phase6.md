# Phase 6 — ReAct Agent Loop

## What was built

A full ReAct (Reason + Act) agent that orchestrates all five tools in a streaming loop
until it reaches a final answer, with inline approval for any destructive operations.

```
POST /agent/stream   →   SSE stream of chunks, tool events, and approval cards
```

---

## New files

| File | Role |
|------|------|
| `backend/services/approval_registry.py` | `ApprovalRegistry` — asyncio.Event bridge that lets agent coroutines pause and wait for out-of-band user approvals |
| `backend/agents/react_agent.py` | ReAct loop — six tool definitions, `_execute_tool()` generator, `run()` entry point |
| `backend/api/routes/agent.py` | `POST /agent/stream` SSE endpoint |
| `backend/tests/test_agent.py` | 21 tests covering all tools, approval flows, and guard rails |

## Modified files

| File | Change |
|------|--------|
| `backend/llm/deepseek_client.py` | Added `complete_with_tools()` — non-streaming call with `tool_choice=auto` |
| `backend/main.py` | Mounts `/agent` router |
| `backend/api/routes/edit.py` | `apply` and `reject` call `registry.resolve()` to unblock waiting agents |
| `backend/api/routes/terminal.py` | `execute` calls `registry.resolve()` with the full result dict |
| `backend/tools/file_tools.py` | `_safe_resolve()` now explicitly rejects absolute paths (pre-existing gap) |
| `extension/src/sidebarProvider.ts` | Uses `/agent/stream` instead of `/chat/stream`; forwards `tool_call`/`tool_result` events |
| `extension/webview/src/types.ts` | `ToolCallData`, `toolCall` field on `ChatMessage`, two new `ExtensionMessage` variants |
| `extension/webview/src/components/ChatPanel.tsx` | Handles `toolCall`/`toolResult` events; filters tool cards from LLM history |
| `extension/webview/src/components/MessageList.tsx` | Renders `ToolCallCard` with spinner / checkmark |
| `extension/webview/src/styles.css` | Tool call card styles (spinning border, done checkmark) |

---

## Architecture

### Agent loop

```
User sends message
       │
       ▼
POST /agent/stream
       │
       ▼
react_agent.run(messages, workspace_root)
       │
   for iteration in range(15):
       │
       ├── complete_with_tools(context, TOOLS)  ← non-streaming LLM call
       │
       ├── choice.finish_reason == "stop"
       │       └── yield chunk(content) → yield done   [return]
       │
       └── finish_reason == "tool_calls"
               │
               for each tool_call:
                 yield tool_call event
                 _execute_tool(name, args, workspace_root)
                   ├── read_file / list_files / search_code  → immediate result
                   ├── write_file / replace_in_file
                   │     ├── edit_service.preview_write/replace()
                   │     ├── registry.register(edit.id)
                   │     ├── yield pending_edit   ← extension shows DiffCard
                   │     └── await registry.wait_for(edit.id, timeout=300s)
                   │               ↑
                   │    /edit/apply or /edit/reject calls registry.resolve()
                   │
                   └── execute_terminal
                         ├── terminal_service.stage()
                         ├── registry.register(cmd.id)
                         ├── yield pending_terminal   ← extension shows modal
                         └── await registry.wait_for(cmd.id, timeout=300s)
                                       ↑
                              /terminal/execute calls registry.resolve()
                 yield tool_result event
                 append tool message to context
       │
   yield error("max iterations reached") → yield done
```

### ApprovalRegistry

The key mechanism that lets the SSE stream stay open while the agent waits for
user approval from a separate HTTP endpoint:

```python
# Agent side (inside the SSE generator):
registry.register(edit.id)       # 1. create asyncio.Event
yield {"type": "pending_edit"}   # 2. extension shows DiffCard
result = await registry.wait_for(edit.id, timeout=300)  # 3. SUSPEND

# Extension side (separate HTTP call, same process):
GET /edit/apply  →  svc.apply()  →  registry.resolve(id, {"applied": True})
                                                      # 4. RESUME agent
```

This works because both the SSE handler and the apply/execute handlers run in the
same asyncio event loop. The `asyncio.Event.set()` call from the HTTP handler
unblocks the `Event.wait()` in the agent coroutine.

### Tool call display in webview

```
MisterPilot
📄 read_file  ⏳       ← tool_call event (spinner)
📄 read_file  ✓        ← tool_result event (checkmark, faded)
🔍 search_code  ⏳
🔍 search_code  ✓
[streaming final answer...]
```

Tool call messages are filtered out of the LLM history on the next turn (same as
edit cards and terminal cards).

---

## Available tools

| Tool | Approval? | Description |
|------|-----------|-------------|
| `read_file` | No | Read file contents (first 150 lines shown) |
| `list_files` | No | List all workspace files (first 300) |
| `search_code` | No | ripgrep / Python fallback, max 50 matches |
| `write_file` | Yes (DiffCard) | Show unified diff, write on approval |
| `replace_in_file` | Yes (DiffCard) | Replace exact text, write on approval |
| `execute_terminal` | Yes (modal) | Run in workspace cwd, 30s timeout |

---

## How to run & test

```bash
cd MisterPilot && source .venv/bin/activate

# Phase 6 tests only
python -m pytest backend/tests/test_agent.py -v

# Full suite
python -m pytest backend/tests/ -v

# Smoke test — agent with a file read
ROOT=$(pwd)
curl -sN -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{\"role\": \"user\", \"content\": \"List the Python files in this project.\"}],
    \"workspace_root\": \"$ROOT\"
  }"
```

---

## Test count

| File | Count |
|------|-------|
| `test_agent.py` | 21 |
| `test_terminal.py` | 21 |
| `test_edit.py` | 24 |
| `test_search.py` | 21 |
| `test_workspace.py` | 10 |
| `test_file_tools.py` | 14 |
| `test_chat.py` | 4 |
| **Total** | **115 + 1 fixed = 116** |

(3 skipped: ripgrep-specific search tests that auto-skip when `rg` is not installed)
