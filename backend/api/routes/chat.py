import json
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...llm.deepseek_client import get_deepseek_client
from ...logging_config import log_pii_findings
from ...services.pii_service import get_pii_pipeline
from ...services.workspace import list_files

log = logging.getLogger("chat")

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    workspace_root: Optional[str] = None


def _build_system(request: ChatRequest) -> Optional[str]:
    """Compose a system message from explicit prompt + workspace context."""
    parts: List[str] = []

    if request.system_prompt:
        parts.append(request.system_prompt)

    if request.workspace_root:
        parts.append(f"The user's workspace is located at: {request.workspace_root}")
        try:
            files = list_files(request.workspace_root)
            if files:
                shown = files[:500]
                listing = "\n".join(f"  {f}" for f in shown)
                suffix = f"\n  ... and {len(files) - 500} more" if len(files) > 500 else ""
                parts.append(f"Workspace files:\n{listing}{suffix}")
        except Exception:
            pass

    return "\n\n".join(parts) if parts else None


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    client = get_deepseek_client()
    pipeline = get_pii_pipeline()

    messages: List[dict] = []
    system = _build_system(request)
    if system:
        messages.append({"role": "system", "content": system})

    all_findings: List = []
    for m in request.messages:
        msg = m.model_dump()
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            sanitized, findings = pipeline.redact(msg["content"])
            if findings:
                log_pii_findings(log, "/chat/stream", findings)
                all_findings.extend(findings)
            msg["content"] = sanitized
        messages.append(msg)

    usage_out: List = []

    async def generate():
        try:
            async for chunk in client.stream_chat(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                usage_out=usage_out,
            ):
                yield {
                    "event": "message",
                    "data": json.dumps({"type": "chunk", "content": chunk}),
                }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": str(exc)}),
            }
        finally:
            usage = usage_out[0] if usage_out else {"prompt_tokens": 0, "completion_tokens": 0}
            log.info(
                "[STREAM] POST /chat/stream  redacted=%d  out=%d in=%d",
                len(all_findings), usage["prompt_tokens"], usage["completion_tokens"],
            )
            yield {
                "event": "done",
                "data": json.dumps({"type": "done"}),
            }

    return EventSourceResponse(generate())


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
