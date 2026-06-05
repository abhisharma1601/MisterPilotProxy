import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...agents import react_agent
from ...logging_config import log_pii_findings
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

    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    async def generate():
        try:
            async for event in react_agent.run(messages, request.workspace_root, api_key=x_api_key):
                if event.get("type") == "usage":
                    usage["prompt_tokens"] = event.get("prompt_tokens", 0)
                    usage["completion_tokens"] = event.get("completion_tokens", 0)
                    continue
                yield {"data": json.dumps(event)}
                if event.get("type") == "done":
                    break
        except Exception as exc:  # noqa: BLE001
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}
            yield {"data": json.dumps({"type": "done"})}
        finally:
            log.info(
                "[STREAM] POST /agent/stream  redacted=%d  out=%d in=%d",
                len(all_findings), usage["prompt_tokens"], usage["completion_tokens"],
            )

    return EventSourceResponse(generate())
