"""
ReAct agent loop.

Each iteration:
  1. Call LLM (streaming, tool_choice=auto)
  2. Yield any model prose as chunks
  3. If finish_reason=tool_calls:
       - MCP tools  → executed directly by the backend via MCPClientManager
       - Local tools → delegated to the VS Code extension via SSE + ApprovalRegistry
  4. If finish_reason=stop → done
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..config import get_config
from ..llm.deepseek_client import get_deepseek_client
from ..services.approval_registry import get_approval_registry
from ..services.pii_service import get_pii_pipeline

try:
    from ..mcp.client import get_mcp_manager
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

log = logging.getLogger(__name__)


AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a file in the workspace. "
                "Use this to understand existing code before suggesting changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all tracked files in the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for text or code patterns across all workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text or pattern to search for.",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob to restrict search scope, e.g. '*.py'.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching is case-sensitive (default false).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write full content to a file. "
                "Shows the user a unified diff and requires explicit approval before anything is written."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new content for the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "Replace a specific block of text inside an existing file. "
                "Shows a diff and requires approval before writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace (must appear exactly once in the file).",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_terminal",
            "description": (
                "Run a shell command inside the workspace directory. "
                "Shows the command to the user and requires approval before execution. "
                "Use ONLY for: running builds, tests, installs, git operations, or commands "
                "that cannot be done with the other tools. "
                "NEVER use this to read files (use read_file) or search code (use search_code)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def _build_system(
    workspace_root: Optional[str],
    mode: str = "agent",
    mcp_server_names: Optional[List[str]] = None,
) -> str:
    identity = [
        "You are MisterPilot, a sharp and efficient AI coding assistant embedded in VS Code.",
        "",
    ]

    if mode == "ask":
        parts = identity + [
            "You are in Ask mode: answer questions, explain concepts, and read files when helpful.",
            "You MUST NOT create, modify, or delete any files.",
            "You MUST NOT run any terminal commands.",
            "",
            "Available tools (read-only):",
            "  read_file, list_files, search_code — use freely to look up code",
            "",
            "If the user asks you to make changes, explain clearly what changes would be needed "
            "but do not attempt to write or modify anything.",
        ]
    else:
        parts = identity + [
            "You answer questions about code, explain concepts, and help implement changes.",
            "",
            "## Local tools",
            "  read_file        — read a file. Use this to inspect any file.",
            "  list_files       — list workspace files.",
            "  search_code      — search for text or patterns across the workspace.",
            "  write_file       — write full file content (shows diff, requires approval).",
            "  replace_in_file  — replace a specific block in a file (shows diff, requires approval).",
            "  execute_terminal — run a shell command (requires approval).",
            "",
            "## Local tool rules",
            "  - Always use read_file to read files. NEVER use execute_terminal to cat, head, or tail a file.",
            "  - Always use search_code to search. NEVER use execute_terminal to grep or find.",
            "  - Only use execute_terminal for: builds, tests, package installs, git operations,",
            "    or tasks that genuinely require a shell command.",
            "  - Always read relevant files before making changes.",
            "  - Prefer replace_in_file over write_file when editing an existing file.",
        ]

    if mcp_server_names:
        parts += [
            "",
            "## Connected MCP servers",
        ]
        for name in mcp_server_names:
            parts.append(f"  {name} — tools prefixed mcp__{name}__")
        parts += [
            "",
            "## MCP usage rules (READ CAREFULLY)",
            "  - When the user's request involves an external service (GitHub, web fetch, etc.),",
            "    ALWAYS use the corresponding MCP tool. Never guess, assume, or hallucinate data",
            "    that an MCP tool can retrieve — call the tool instead.",
            "  - GitHub operations (list repos, search code, create issues, get PRs, etc.) MUST",
            "    use the github MCP tool. NEVER assume or guess the user's GitHub username or repo",
            "    names — the MCP server is authenticated and will return the correct data.",
            "  - When the user asks for THEIR OWN repos and you don't know their GitHub username,",
            "    run `git remote get-url origin` via execute_terminal to extract it from the",
            "    workspace. Parse the owner from the URL (e.g. github.com/<owner>/<repo>),",
            "    then use search_repositories with 'user:<owner>'. NEVER guess from folder names.",
            "  - Pick the ONE most relevant MCP server for the task. Do not call multiple MCP servers",
            "    unless the request genuinely requires data from more than one.",
            "  - For pure coding tasks (read files, write code, run tests) always prefer local tools.",
            "    MCP servers are for external services, not local workspace operations.",
            "  - NEVER call an MCP server speculatively or 'just to check'. Only call it when needed.",
        ]

    if workspace_root:
        parts.append(f"\nWorkspace root: {workspace_root}")
    else:
        parts.append(
            "\nNo workspace is open. You can still answer general coding questions, "
            "but file and terminal tools are unavailable."
        )
    return "\n".join(parts)


async def _execute_tool(
    name: str,
    args: Dict[str, Any],
    workspace_root: Optional[str],
    call_id: str,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Delegate all tool execution to the VS Code extension via SSE.

    The extension has local filesystem and terminal access; the backend only
    does LLM inference. Flow:
      1. Register call_id in ApprovalRegistry
      2. Yield tool_call_request → extension executes locally
      3. Extension calls POST /agent/tool_result to unblock the registry
      4. Yield the result content back to the agent loop
    """
    if workspace_root is None and name in (
        "read_file", "list_files", "search_code",
        "write_file", "replace_in_file", "execute_terminal",
    ):
        yield {"type": "_result", "content": "No workspace is open — file tools are unavailable."}
        return

    registry = get_approval_registry()
    registry.register(call_id)
    yield {"type": "tool_call_request", "call_id": call_id, "tool": name, "args": args}
    result = await registry.wait_for(call_id, timeout=300.0)
    content = result.get("content", "") if result else f"Tool call timed out after 5 minutes: {name}"
    yield {"type": "_result", "content": content}


# Read-only subset of tools used in Ask mode
ASK_TOOLS: List[Dict[str, Any]] = [
    t for t in AGENT_TOOLS
    if t["function"]["name"] in ("read_file", "list_files", "search_code")
]


def _sanitize_err(msg: str, api_key: Optional[str]) -> str:
    """Strip the API key from error messages before they reach the UI."""
    if api_key and len(api_key) > 8 and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED]")
    return msg


async def run(
    messages: List[Dict[str, Any]],
    workspace_root: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    mode: str = "agent",
    mcp_servers: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Main ReAct loop.  Yields SSE-compatible event dicts:

        { type: "chunk",            content: str }      — model prose (tool reasoning or final answer)
        { type: "tool_call",        id: str, tool: str, args: dict }
        { type: "tool_result",      id: str, tool: str, content: str }
        { type: "pending_edit",     id, path, diff, original, proposed, is_new_file }
        { type: "pending_terminal", id, command, workspace_root }
        { type: "error",            message: str }
        { type: "done" }
    """
    llm = get_deepseek_client(api_key)
    cfg = get_config()
    pipeline = get_pii_pipeline()

    # Redact PII from user messages before they reach the LLM
    clean_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            sanitized, findings = pipeline.redact(msg["content"])
            if findings:
                log.info(
                    "PII redacted: %d finding(s) — types: %s",
                    len(findings),
                    ", ".join(f.entity_type for f in findings),
                )
            clean_messages.append({**msg, "content": sanitized})
        else:
            clean_messages.append(msg)

    base_tools = AGENT_TOOLS if mode == "agent" else ASK_TOOLS

    # Fetch MCP tools and merge them into the tool list
    mcp_tool_list: List[Dict[str, Any]] = []
    mcp_manager = None
    if mcp_servers and _MCP_AVAILABLE:
        mcp_manager = get_mcp_manager()
        try:
            mcp_tool_list = await mcp_manager.get_tools(mcp_servers)
        except Exception as exc:
            log.error("Failed to load MCP tools: %s", exc)

    tools = base_tools + mcp_tool_list
    mcp_server_names = list(mcp_servers.keys()) if mcp_servers else None

    context: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system(workspace_root, mode, mcp_server_names)},
        *clean_messages,
    ]

    resolved_model = model or cfg.deepseek.model

    for _iteration in range(50):
        # ── LLM call (streaming) ─────────────────────────────────────────────
        message_content: Optional[str] = None
        tool_calls: Optional[List[Dict[str, Any]]] = None
        finish_reason: Optional[str] = None

        try:
            async for event in llm.stream_with_tools(
                context,
                tools=tools,
                model=resolved_model,
                temperature=0.3,
                max_tokens=8192,
            ):
                if event["type"] == "content_chunk":
                    yield {"type": "chunk", "content": event["content"]}
                elif event["type"] == "complete":
                    message_content = event["content"]
                    tool_calls = event["tool_calls"]
                    finish_reason = event["finish_reason"]
                    if event["usage"]:
                        u = event["usage"]
                        cached = 0
                        if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
                            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
                        turn_cache_hit = cached
                        turn_cache_miss = max(0, u.prompt_tokens - cached)
                        turn_output = u.completion_tokens
                        # Emit usage immediately so cost is tracked even if user stops
                        yield {
                            "type": "usage",
                            "model": resolved_model,
                            "output_tokens": turn_output,
                            "cache_hit_tokens": turn_cache_hit,
                            "cache_miss_tokens": turn_cache_miss,
                        }
        except Exception as exc:  # noqa: BLE001
            log.error("LLM error in agent loop: %s", exc)
            yield {"type": "error", "message": _sanitize_err("LLM request failed — please try again", api_key)}
            yield {"type": "done"}
            return

        # ── Final answer ──────────────────────────────────────────────────────
        if finish_reason == "stop" or not tool_calls:
            yield {"type": "done"}
            return

        # ── Tool calls ────────────────────────────────────────────────────────
        context.append(
            {
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls,
            }
        )

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            yield {"type": "tool_call", "id": tc["id"], "tool": tool_name, "args": tool_args}

            tool_result_content = ""
            if mcp_manager and mcp_servers and mcp_manager.is_mcp_tool(tool_name):
                # Execute MCP tools directly in the backend — no extension round-trip needed
                tool_result_content = await mcp_manager.call_tool(tool_name, tool_args, mcp_servers)
            else:
                async for event in _execute_tool(tool_name, tool_args, workspace_root, tc["id"]):
                    if event["type"] == "_result":
                        tool_result_content = event["content"]
                    else:
                        yield event  # forward pending_edit / pending_terminal to extension

            # Cap tool result at 8 KB to avoid blowing out the context window
            capped = tool_result_content[:8192]
            if len(tool_result_content) > 8192:
                capped += "\n... (truncated)"

            # Redact PII from tool results before they reach the LLM
            sanitized_tool_result, tool_findings = pipeline.redact(capped)
            if tool_findings:
                log.info(
                    "PII redacted from tool result (%s): %d finding(s) — types: %s",
                    tool_name,
                    len(tool_findings),
                    ", ".join(f.entity_type for f in tool_findings),
                )

            context.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": sanitized_tool_result,
                }
            )

            yield {
                "type": "tool_result",
                "id": tc["id"],
                "tool": tool_name,
                "content": sanitized_tool_result[:4096],
            }

    yield {"type": "chunk", "content": "\n\n⚠️ Reached maximum iteration limit. Task may be incomplete."}
    yield {"type": "done"}
