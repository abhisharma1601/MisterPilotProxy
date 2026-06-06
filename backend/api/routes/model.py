"""
Direct model access API.

POST /model/chat   — takes a model name, API key, and prompt; returns a full
                     OpenAI-format chat completion JSON response.
POST /model/stream — same inputs but returns an SSE stream of chunks.

Both endpoints apply PII redaction on the prompt before sending to the LLM,
matching the behaviour of /chat/stream.
"""

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ...llm.deepseek_client import get_deepseek_client
from ...logging_config import log_pii_findings
from ...services.pii_service import get_pii_pipeline

log = logging.getLogger("model")

router = APIRouter()


class ModelRequest(BaseModel):
    model: str
    apikey: str
    prompt: str


# ── shared helper ────────────────────────────────────────────────────

def _sanitize(prompt: str, route: str) -> tuple[str, int]:
    """Run the PII redaction pipeline on *prompt* and return (safe_text, count)."""
    pipeline = get_pii_pipeline()
    sanitized, findings = pipeline.redact(prompt)
    if findings:
        log_pii_findings(log, route, findings)
    return sanitized, len(findings)


# ── /model/chat ──────────────────────────────────────────────────────

@router.post("/chat")
async def model_chat(request: ModelRequest) -> JSONResponse:
    """
    Non-streaming completion using the caller's own model and API key.

    Returns the raw OpenAI ChatCompletion JSON with id, object, created,
    model, choices, usage, and system_fingerprint fields.
    """
    sanitized, num_findings = _sanitize(request.prompt, "/model/chat")

    client = get_deepseek_client(request.apikey)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": sanitized}]

    try:
        completion = await client.complete(
            messages=messages,
            model=request.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    result = completion.model_dump()
    log.info(
        "[CHAT] POST /model/chat  model=%s  redacted=%d  out=%d in=%d",
        request.model,
        num_findings,
        result.get("usage", {}).get("prompt_tokens", 0),
        result.get("usage", {}).get("completion_tokens", 0),
    )
    return JSONResponse(content=result)


# ── /model/stream ────────────────────────────────────────────────────

@router.post("/stream")
async def model_stream(request: ModelRequest) -> EventSourceResponse:
    """
    Stream a completion using the caller's own model and API key.

    Request body:
        model  — model identifier (e.g. "deepseek-v4-pro")
        apikey — the API key for the provider
        prompt — the user prompt to send (PII-redacted before dispatch)

    Returns an SSE stream with events:
        message  → {"type": "chunk", "content": "..."}
        error    → {"type": "error", "message": "..."}
        done     → {"type": "done"}
    """
    sanitized, num_findings = _sanitize(request.prompt, "/model/stream")

    client = get_deepseek_client(request.apikey)
    messages: List[dict] = [{"role": "user", "content": sanitized}]
    usage_out: List = []

    async def generate():
        try:
            async for chunk in client.stream_chat(
                messages=messages,
                model=request.model,
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
                "[STREAM] POST /model/stream  model=%s  redacted=%d  out=%d in=%d",
                request.model,
                num_findings,
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            yield {
                "event": "done",
                "data": json.dumps({"type": "done"}),
            }

    return EventSourceResponse(generate())
