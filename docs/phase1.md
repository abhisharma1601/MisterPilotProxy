# Phase 1 — VS Code Sidebar Chat

## What was built

A working end-to-end chat loop:

```
VS Code Sidebar (React/Vite webview)
  └─► extension/src/sidebarProvider.ts  (fetch SSE stream)
        └─► FastAPI  /chat/stream        (SSE endpoint)
              └─► DeepSeek API           (OpenAI-compatible, streaming)
```

---

## Architecture

### Backend (`backend/`)

| File | Role |
|------|------|
| `main.py` | FastAPI app factory, CORS, route mounting |
| `config.py` | Pydantic models loaded from `config.yaml` |
| `llm/deepseek_client.py` | Async OpenAI client wrapper with retry logic |
| `api/routes/chat.py` | `POST /chat/stream` → SSE, `GET /chat/health` |

The chat endpoint returns a **Server-Sent Events** stream. Each event carries a JSON payload:

```
data: {"type": "chunk", "content": "Hello"}
data: {"type": "done"}
```

On error:
```
data: {"type": "error", "message": "connection refused"}
data: {"type": "done"}
```

### Extension (`extension/`)

| File | Role |
|------|------|
| `src/extension.ts` | Activation — registers the sidebar provider |
| `src/sidebarProvider.ts` | Resolves the webview, bridges messages to/from backend |
| `src/types.ts` | Shared message types |

The extension reads the SSE stream with `fetch()` + `ReadableStream`. Each chunk is forwarded to the webview via `webview.postMessage()`.

### Webview (`extension/webview/`)

Built with **React 18 + Vite**. Compiled to a single IIFE bundle (`webview-dist/assets/index.js`) so VS Code's Content Security Policy nonce mechanism works without code splitting.

| Component | Role |
|-----------|------|
| `ChatPanel` | Owns all state: messages, streaming flag, workspace root |
| `MessageList` | Renders messages; markdown via `marked` + DOMPurify sanitization |
| `InputBar` | Auto-resizing textarea; Enter to send, Shift+Enter for newline |

---

## Configuration

Edit `backend/config.yaml`:

```yaml
deepseek:
  api_key: "YOUR_DEEPSEEK_API_KEY"
  base_url: "http://127.0.0.1:8666/v1"   # or https://api.deepseek.com/v1
  model: "deepseek-v4-pro"
  timeout: 60
  max_retries: 3
```

The `api_key` can also be set via environment variable by overriding `config.yaml` or extending `config.py`.

The extension reads `misterpilot.backendUrl` from VS Code settings (default `http://localhost:8000`).

---

## How to run

### 1 — Backend

```bash
cd MisterPilot

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Set your API key
#   Option A: edit backend/config.yaml
#   Option B: export DEEPSEEK_API_KEY=sk-...  (add env loading if needed)

# Start the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/chat/health
# {"status":"ok"}
```

Quick smoke test:
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}]}'
```

### 2 — Webview (build once before launching the extension)

```bash
cd MisterPilot/extension/webview
npm install
npm run build
# → ../webview-dist/assets/index.js  +  index.css
```

For live webview development (hot-reload in browser, not in VS Code):
```bash
npm run dev
# Opens http://localhost:5173 — messages won't reach the backend
# because acquireVsCodeApi() is absent; useful for UI iteration only.
```

### 3 — Extension

```bash
cd MisterPilot/extension
npm install
npm run build:ext          # compiles src/ → out/

# Open the extension folder in VS Code, then press F5
# (or: Run → Start Debugging)
# A new Extension Development Host window opens.
# Click the stacked-layers icon in the Activity Bar → MisterPilot panel appears.
```

One-liner to build everything and launch:
```bash
cd MisterPilot/extension
npm run build:all && code --extensionDevelopmentPath="$(pwd)"
```

---

## Running tests

```bash
cd MisterPilot
source .venv/bin/activate
cd backend
pytest -v
```

Four tests are included:
- `test_health_returns_ok` — sanity check the health endpoint
- `test_chat_stream_returns_sse` — verifies SSE format and token assembly
- `test_chat_stream_propagates_error` — error path emits an error event then done
- `test_chat_stream_injects_system_prompt` — system_prompt is prepended correctly

---

## Next phase

**Phase 2** will add:
- `GET /workspace/files` — workspace file tree (with ignore rules)
- `read_file(path)` tool — reads any file within the workspace
- The extension passes the workspace root to the backend on connect
