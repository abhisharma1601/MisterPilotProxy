"""
ReAct agent loop.

Each iteration:
  1. Call LLM (non-streaming, tool_choice=auto)
  2. Yield any model prose as chunks
  3. If finish_reason=tool_calls  → execute each tool, loop
  4. If finish_reason=stop        → done

Approval-required tools (write_file, replace_in_file, execute_terminal) register
with ApprovalRegistry before yielding their SSE event, then await the registry
until the extension calls /edit/apply, /edit/reject, or /terminal/execute.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..config import get_config
from ..llm.deepseek_client import get_deepseek_client
from ..services.approval_registry import get_approval_registry
from ..services.edit_service import get_edit_service
from ..services.pii_service import get_pii_pipeline
from ..services.terminal_service import get_terminal_service
from ..services.workspace import list_files
from ..tools.file_tools import read_file, BlockedFileError, PathTraversalError
from ..tools.search_tools import search_code

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
                "Shows the command to the user and requires approval before execution."
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


def _build_system(workspace_root: Optional[str], mode: str = "agent") -> str:
    if mode == "ask":
        parts = [
            "You are MisterPilot, an AI coding assistant running inside VS Code.",
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
        parts = [
            "You are MisterPilot, an AI coding assistant running inside VS Code.",
            "You answer questions about code, explain concepts, and help implement changes.",
            "",
            "Available tools:",
            "  read_file, list_files, search_code — read freely, no approval needed",
            "  write_file, replace_in_file        — shows a diff, requires user approval",
            "  execute_terminal                   — shows the command, requires user approval",
            "",
            "Always read relevant files before suggesting or making changes.",
            "Explain what you are doing before each tool call.",
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
    Async generator that executes one tool call.

    Yields zero or more SSE-style events (pending_edit, pending_terminal),
    then exactly one {"type": "_result", "content": str} as the last item.
    """
    if workspace_root is None and name in (
        "read_file", "list_files", "search_code",
        "write_file", "replace_in_file", "execute_terminal",
    ):
        yield {"type": "_result", "content": "No workspace is open — file tools are unavailable."}
        return

    try:
        if name == "read_file":
            path = args.get("path", "")
            try:
                raw = read_file(workspace_root, path)  # type: ignore[arg-type]
            except BlockedFileError as exc:
                yield {"type": "_result", "content": str(exc)}
                return
            lines = raw.splitlines()
            numbered = "\n".join(f"{i + 1}: {l}" for i, l in enumerate(lines[:150]))
            suffix = f"\n... ({len(lines) - 150} more lines)" if len(lines) > 150 else ""
            yield {"type": "_result", "content": numbered + suffix}

        elif name == "list_files":
            files = list_files(workspace_root)  # type: ignore[arg-type]
            body = "\n".join(files[:300])
            suffix = f"\n... ({len(files) - 300} more files)" if len(files) > 300 else ""
            yield {"type": "_result", "content": body + suffix or "(empty workspace)"}

        elif name == "search_code":
            query = args.get("query", "")
            file_pattern = args.get("file_pattern")
            case_sensitive = bool(args.get("case_sensitive", False))
            matches, _ = await search_code(
                workspace_root,  # type: ignore[arg-type]
                query,
                case_sensitive=case_sensitive,
                file_pattern=file_pattern,
                max_results=50,
            )
            if not matches:
                yield {"type": "_result", "content": f"No matches found for: {query!r}"}
            else:
                lines = [f"{m.file}:{m.line}: {m.content}" for m in matches]
                yield {
                    "type": "_result",
                    "content": f"{len(matches)} match(es):\n" + "\n".join(lines),
                }

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            edit_svc = get_edit_service()
            registry = get_approval_registry()

            try:
                edit = edit_svc.preview_write(workspace_root, path, content)  # type: ignore[arg-type]
            except (PathTraversalError, ValueError) as exc:
                yield {"type": "_result", "content": f"Cannot stage write: {exc}"}
                return

            registry.register(edit.id)
            yield {
                "type": "pending_edit",
                "id": edit.id,
                "path": edit.path,
                "diff": edit.diff(),
                "original": edit.original,
                "proposed": edit.proposed,
                "is_new_file": edit.is_new_file,
            }

            result = await registry.wait_for(edit.id, timeout=300.0)
            if result and result.get("applied"):
                yield {"type": "_result", "content": f"File written: {path}"}
            else:
                yield {"type": "_result", "content": f"Edit rejected by user: {path}"}

        elif name == "replace_in_file":
            path = args.get("path", "")
            old_text = args.get("old_text", "")
            new_text = args.get("new_text", "")
            edit_svc = get_edit_service()
            registry = get_approval_registry()

            try:
                edit = edit_svc.preview_replace(workspace_root, path, old_text, new_text)  # type: ignore[arg-type]
            except (PathTraversalError, FileNotFoundError, ValueError) as exc:
                yield {"type": "_result", "content": f"Cannot stage replacement: {exc}"}
                return

            registry.register(edit.id)
            yield {
                "type": "pending_edit",
                "id": edit.id,
                "path": edit.path,
                "diff": edit.diff(),
                "original": edit.original,
                "proposed": edit.proposed,
                "is_new_file": edit.is_new_file,
            }

            result = await registry.wait_for(edit.id, timeout=300.0)
            if result and result.get("applied"):
                yield {"type": "_result", "content": f"Replacement applied: {path}"}
            else:
                yield {"type": "_result", "content": f"Replacement rejected by user: {path}"}

        elif name == "execute_terminal":
            command = args.get("command", "")
            terminal_svc = get_terminal_service()
            registry = get_approval_registry()

            try:
                cmd = terminal_svc.stage(command, workspace_root)  # type: ignore[arg-type]
            except (ValueError, NotADirectoryError) as exc:
                yield {"type": "_result", "content": f"Cannot stage command: {exc}"}
                return

            registry.register(cmd.id)
            yield {
                "type": "pending_terminal",
                "id": cmd.id,
                "command": cmd.command,
                "workspace_root": cmd.workspace_root,
            }

            result = await registry.wait_for(cmd.id, timeout=None)
            if result is None:
                yield {"type": "_result", "content": "Approval timed out — command was not executed."}
            elif not result.get("approved"):
                yield {"type": "_result", "content": f"Command denied by user: {command}"}
            else:
                parts: List[str] = []
                if result.get("stdout"):
                    parts.append(f"stdout:\n{result['stdout']}")
                if result.get("stderr"):
                    parts.append(f"stderr:\n{result['stderr']}")
                parts.append(f"exit code: {result.get('exit_code', -1)}")
                if result.get("timed_out"):
                    parts.append("(timed out)")
                yield {"type": "_result", "content": "\n".join(parts) or "(no output)"}

        else:
            yield {"type": "_result", "content": f"Unknown tool: {name!r}"}

    except Exception as exc:  # noqa: BLE001
        yield {"type": "_result", "content": f"Tool error ({name}): {exc}"}


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

    tools = AGENT_TOOLS if mode == "agent" else ASK_TOOLS

    context: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system(workspace_root, mode)},
        *clean_messages,
    ]

    total_output_tokens = 0
    total_cache_hit_tokens = 0
    total_cache_miss_tokens = 0
    resolved_model = model or cfg.deepseek.model

    while True:
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
                        total_cache_hit_tokens += cached
                        total_cache_miss_tokens += max(0, u.prompt_tokens - cached)
                        total_output_tokens += u.completion_tokens
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": _sanitize_err(f"LLM error: {exc}", api_key)}
            yield {"type": "done"}
            return

        # ── Final answer ──────────────────────────────────────────────────────
        if finish_reason == "stop" or not tool_calls:
            yield {
                "type": "usage",
                "model": resolved_model,
                "output_tokens": total_output_tokens,
                "cache_hit_tokens": total_cache_hit_tokens,
                "cache_miss_tokens": total_cache_miss_tokens,
            }
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
                "content": sanitized_tool_result[:300],
            }

