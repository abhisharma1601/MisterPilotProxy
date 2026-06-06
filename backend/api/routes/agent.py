import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...agents import react_agent
from ...logging_config import log_pii_findings
from ...services.approval_registry import get_approval_registry
from ...services.pii_service import get_pii_pipeline

log = logging.getLogger("agent")

router = APIRouter()

AVAILABLE_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]

# Per-token prices in USD, derived from DeepSeek billing data
PRICING: dict = {
    "deepseek-v4-pro": {
        "output":    0.00000087,
        "cache_hit": 0.000000003625,
        "cache_miss": 0.000000435,
    },
    "deepseek-v4-flash": {
        "output":    0.00000028,
        "cache_hit": 0.0000000028,
        "cache_miss": 0.00000014,
    },
}


def _calc_cost(model: str, output: int, cache_hit: int, cache_miss: int) -> float:
    p = PRICING.get(model, PRICING["deepseek-v4-pro"])
    return output * p["output"] + cache_hit * p["cache_hit"] + cache_miss * p["cache_miss"]


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

    model = request.model if request.model in AVAILABLE_MODELS else None
    mode = request.mode if request.mode in ("agent", "ask") else "agent"
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def generate():
        try:
            async for event in react_agent.run(
                messages, request.workspace_root, api_key=x_api_key, model=model, mode=mode
            ):
                if event.get("type") == "usage":
                    usage["prompt_tokens"] = event.get("cache_hit_tokens", 0) + event.get("cache_miss_tokens", 0)
                    usage["completion_tokens"] = event.get("output_tokens", 0)
                    cost_usd = _calc_cost(
                        event.get("model", mode),
                        event.get("output_tokens", 0),
                        event.get("cache_hit_tokens", 0),
                        event.get("cache_miss_tokens", 0),
                    )
                    yield {"data": json.dumps({"type": "cost", "usd": cost_usd})}
                    continue
                yield {"data": json.dumps(event)}
                if event.get("type") == "done":
                    break
        except Exception as exc:  # noqa: BLE001
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}
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
    """
    resolved = get_approval_registry().resolve(req.call_id, {"content": req.content})
    if not resolved:
        raise HTTPException(status_code=404, detail=f"No pending tool call: {req.call_id!r}")
    return ToolResultResponse(ok=True)
