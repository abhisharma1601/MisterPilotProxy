import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...agents import react_agent
from ...logging_config import log_pii_findings
from ...services.approval_registry import get_approval_registry
from ...models import AVAILABLE_MODELS
from ...services.cost_service import get_cost_service
from ...llm.deepseek_client import LLMAuthError
from ...services.key_service import key_type, resolve_api_key
from ...services.pii_service import get_pii_pipeline

log = logging.getLogger("agent")

router = APIRouter()


class AgentMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class AgentRequest(BaseModel):
    messages: List[AgentMessage]
    workspace_root: Optional[str] = None
    model: Optional[str] = None
    mode: str = "agent"  # "agent" | "ask"
    mcp_servers: Optional[Dict[str, Any]] = None


@router.get("/models")
async def list_models() -> Dict[str, List[str]]:
    return {"models": AVAILABLE_MODELS}


@router.post("/stream")
async def agent_stream(
    request: AgentRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> EventSourceResponse:
    pipeline = get_pii_pipeline()
    cost_service = get_cost_service()
    messages: List[Dict[str, Any]] = []

    all_findings: List = []
    for m in request.messages:
        msg = m.model_dump(exclude_none=True)
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            sanitized, findings = pipeline.redact(msg["content"])
            if findings:
                log_pii_findings(log, "/agent/stream", findings)
                all_findings.extend(findings)
            msg["content"] = sanitized
        messages.append(msg)

    # Detect if the last user message was redacted so the UI can show the placeholder
    last_user_sanitized: Optional[str] = next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        None,
    )
    last_user_original: Optional[str] = next(
        (m.content for m in reversed(request.messages)
         if m.role == "user" and isinstance(m.content, str)),
        None,
    )
    input_was_redacted = (
        last_user_original is not None
        and last_user_sanitized is not None
        and last_user_original != last_user_sanitized
    )

    # Resolve the inbound header key: a MisterPilot key (mp_…) is swapped for our
    # own DeepSeek key from .env; a real DeepSeek key is passed through unchanged.
    llm_key = resolve_api_key(x_api_key)
    model = request.model if request.model in AVAILABLE_MODELS else None
    mode = request.mode if request.mode in ("agent", "ask") else "agent"
    # Accumulated token counts across all LLM turns — billed once at end.
    accumulated: Dict[str, Any] = {
        "output": 0, "cache_hit": 0, "cache_miss": 0,
        "model": model, "billed": False,
    }

    async def generate():
        if input_was_redacted:
            yield {"data": json.dumps({"type": "sanitized_input", "content": last_user_sanitized})}
        try:
            async for event in react_agent.run(
                messages, request.workspace_root,
                api_key=llm_key, model=model, mode=mode,
                mcp_servers=request.mcp_servers,
            ):
                if event.get("type") == "usage":
                    # Accumulate — do NOT call cost service yet
                    accumulated["output"] += event.get("output_tokens", 0)
                    accumulated["cache_hit"] += event.get("cache_hit_tokens", 0)
                    accumulated["cache_miss"] += event.get("cache_miss_tokens", 0)
                    accumulated["model"] = event.get("model") or accumulated["model"]
                    continue
                yield {"data": json.dumps(event)}
                if event.get("type") == "done":
                    # Natural completion — bill once and send cost to client
                    if accumulated["output"] or accumulated["cache_hit"] or accumulated["cache_miss"]:
                        data = await cost_service.calc_cost(
                            model=accumulated["model"],
                            output=accumulated["output"],
                            cache_hit=accumulated["cache_hit"],
                            cache_miss=accumulated["cache_miss"],
                            api_key=x_api_key,
                        )
                        accumulated["billed"] = True
                        yield {
                            "data": json.dumps({
                                "type": "cost",
                                "usd": data["costUsd"],
                                "inr": data.get("costInr"),
                            })
                        }
                    break
        except LLMAuthError as exc:
            log.warning("[STREAM] POST /agent/stream  auth_error")
            yield {"data": json.dumps({"type": "error", "message": str(exc), "code": 401})}
        except Exception as exc:  # noqa: BLE001
            log.error("[STREAM] POST /agent/stream  error_type=%s", type(exc).__name__)
            yield {"data": json.dumps({"type": "error", "message": "An internal error occurred"})}
            yield {"data": json.dumps({"type": "done"})}
        finally:
            # User stopped mid-stream — bill server-side without yielding to client
            if not accumulated["billed"] and (accumulated["output"] or accumulated["cache_hit"] or accumulated["cache_miss"]):
                try:
                    await cost_service.calc_cost(
                        model=accumulated["model"],
                        output=accumulated["output"],
                        cache_hit=accumulated["cache_hit"],
                        cache_miss=accumulated["cache_miss"],
                        api_key=x_api_key,
                    )
                except Exception:
                    pass
            log.info(
                "[STREAM] POST /agent/stream  mode=%s  model=%s  redacted=%d  out=%d in=%d",
                mode, model or "default",
                len(all_findings), accumulated["cache_hit"] + accumulated["cache_miss"], accumulated["output"],
            )

    return EventSourceResponse(generate())


class McpStatusRequest(BaseModel):
    mcp_servers: Optional[Dict[str, Any]] = None


@router.post("/mcp-status")
async def mcp_status(request: McpStatusRequest):
    """Probe configured MCP servers and return connection status."""
    if not request.mcp_servers:
        return {"servers": []}
    try:
        from ...mcp.client import get_mcp_manager
        manager = get_mcp_manager()
        servers = await manager.get_status(request.mcp_servers)
        return {"servers": servers}
    except Exception as exc:
        log.error("mcp-status error: %s", exc)
        return {"servers": [
            {"name": n, "connected": False, "error": str(exc)}
            for n in request.mcp_servers.keys()
        ]}


class ToolResultRequest(BaseModel):
    call_id: str
    content: str


class ToolResultResponse(BaseModel):
    ok: bool


@router.post("/tool_result")
async def submit_tool_result(req: ToolResultRequest) -> ToolResultResponse:
    """
    Called by the VS Code extension after it has executed a tool locally.
    Unblocks the waiting agent coroutine in ApprovalRegistry.
    Tool result content is redacted for PII before reaching the agent loop.
    """
    pipeline = get_pii_pipeline()
    sanitized, findings = pipeline.redact(req.content)
    if findings:
        log_pii_findings(log, "/agent/tool_result", findings)

    resolved = get_approval_registry().resolve(req.call_id, {"content": sanitized})
    if not resolved:
        raise HTTPException(status_code=404, detail=f"No pending tool call: {req.call_id!r}")
    return ToolResultResponse(ok=True)
