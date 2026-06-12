import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...agents import react_agent
from ...logging_config import log_pii_findings
from ...services.approval_registry import get_approval_registry
from ...services.cost_service import AVAILABLE_MODELS, get_cost_service
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
    deepseek_key = resolve_api_key(x_api_key)
    # Key type drives billing: MisterPilot keys carry a profit margin, DeepSeek
    # keys are charged the raw cost.
    client_key_type = key_type(x_api_key)

    model = request.model if request.model in AVAILABLE_MODELS else None
    mode = request.mode if request.mode in ("agent", "ask") else "agent"
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def generate():
        if input_was_redacted:
            yield {"data": json.dumps({"type": "sanitized_input", "content": last_user_sanitized})}
        try:
            async for event in react_agent.run(
                messages, request.workspace_root, api_key=deepseek_key, model=model, mode=mode
            ):
                if event.get("type") == "usage":
                    usage["prompt_tokens"] = event.get("cache_hit_tokens", 0) + event.get("cache_miss_tokens", 0)
                    usage["completion_tokens"] = event.get("output_tokens", 0)
                    cost_usd = cost_service.calc_cost(
                        event.get("model"),
                        event.get("output_tokens", 0),
                        event.get("cache_hit_tokens", 0),
                        event.get("cache_miss_tokens", 0),
                        key_type=client_key_type,
                    )
                    yield {"data": json.dumps({"type": "cost", "usd": cost_usd})}
                    continue
                yield {"data": json.dumps(event)}
                if event.get("type") == "done":
                    break
        except Exception as exc:  # noqa: BLE001
            log.error("[STREAM] POST /agent/stream  error_type=%s", type(exc).__name__)
            yield {"data": json.dumps({"type": "error", "message": "An internal error occurred"})}
            yield {"data": json.dumps({"type": "done"})}
        finally:
            log.info(
                "[STREAM] POST /agent/stream  mode=%s  model=%s  redacted=%d  out=%d in=%d",
                mode, model or "default",
                len(all_findings), usage["prompt_tokens"], usage["completion_tokens"],
            )

    return EventSourceResponse(generate())


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
