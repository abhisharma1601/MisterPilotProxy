"""
OpenAI-compatible chat completions API.

POST /model/chat   — single endpoint, behaves like OpenAI /v1/chat/completions.
  - `stream=false` → JSONResponse with a full ChatCompletion object.
  - `stream=true`  → StreamingResponse with OpenAI-format SSE chunks.
  - API key is read from the ``Authorization: Bearer <key>`` header, falling
    back to ``body.apikey`` and then to the configured default.

POST /model/stream — deprecated alias, kept for backward compat; delegates
                     to /model/chat with stream=true.

Both modes apply PII redaction on user messages before they reach the LLM.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ...llm.deepseek_client import get_deepseek_client, LLMAuthError
from ...logging_config import log_pii_findings
from ...services.cost_service import get_cost_service
from ...services.key_service import key_type, resolve_api_key
from ...services.pii_service import get_pii_pipeline

log = logging.getLogger("model")

router = APIRouter()


# ── request model (matches OpenAI chat/completions shape) ─────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 8192
    apikey: Optional[str] = None


# ── helpers ───────────────────────────────────────────────────────────

def _resolve_key(
    raw_request: Request, body: ChatCompletionRequest
) -> tuple[str, str]:
    """Return (raw_key, resolved_key).

    raw_key     — key as sent by the client; used for billing/key-type checks.
    resolved_key — key forwarded to the LLM client; mp-… tokens are swapped for
                   our DeepSeek key loaded from the environment.
    """
    auth = raw_request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        raw_key = auth[7:].strip()
    elif body.apikey:
        raw_key = body.apikey
    else:
        raw_key = ""
    return raw_key, resolve_api_key(raw_key)


async def _calc_cost_usage(
    cost_service,
    model: str,
    completion_tokens: int,
    cache_hit: int,
    cache_miss: int,
    api_key: str,
) -> dict:
    """Await cost calculation, routing to charge vs. calculate endpoint by key type."""
    return await cost_service.calc_cost(
        model=model,
        output=completion_tokens,
        cache_hit=cache_hit,
        cache_miss=cache_miss,
        api_key=api_key,
    )


def _sanitize_messages(
    messages: List[Message], route: str
) -> tuple[List[Dict[str, Any]], int]:
    pipeline = get_pii_pipeline()
    cleaned: List[Dict[str, Any]] = []
    total_findings = 0

    for m in messages:
        msg: Dict[str, Any] = {"role": m.role, "content": m.content}
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            sanitized, findings = pipeline.redact(msg["content"])
            if findings:
                log_pii_findings(log, route, findings)
                total_findings += len(findings)
            msg["content"] = sanitized
        cleaned.append(msg)

    return cleaned, total_findings


# ── /model/chat ───────────────────────────────────────────────────────

@router.post("/chat")
async def model_chat(
    body: ChatCompletionRequest,
    raw_request: Request,
):
    api_key, resolved_key = _resolve_key(raw_request, body)
    if not resolved_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide it via Authorization: Bearer <key> header or body.apikey.",
        )

    messages, num_findings = _sanitize_messages(body.messages, "/model/chat")
    client = get_deepseek_client(resolved_key)
    cost_service = get_cost_service()

    if body.stream:
        return StreamingResponse(
            _stream_chunks(client, messages, body, num_findings,
                           cost_service, api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            completion = await client.complete(
                messages=messages,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except LLMAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        except Exception as exc:
            log.error("[CHAT] POST /model/chat  stream=false  error_type=%s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Upstream model error")

        result = completion.model_dump()
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_details = usage.get("prompt_tokens_details", {})
        cache_hit = prompt_details.get("cached_tokens", 0) if isinstance(prompt_details, dict) else 0
        cache_miss = prompt_tokens - cache_hit
        cost_result = await _calc_cost_usage(
            cost_service,
            body.model,
            completion_tokens,
            cache_hit,
            cache_miss,
            api_key,
        )
        cost_usd = cost_result.get("costUsd", 0.0)
        log.info(
            "[CHAT] POST /model/chat  stream=false  model=%s  redacted=%d  in=%d out=%d  cache_hit=%d cache_miss=%d cost_usd=%.8f key_type=%s",
            body.model,
            num_findings,
            prompt_tokens,
            completion_tokens,
            cache_hit,
            cache_miss,
            cost_usd,
            client_key_type,
        )
        return JSONResponse(content=result)


async def _stream_chunks(
    client,
    messages: List[Dict[str, Any]],
    body: ChatCompletionRequest,
    num_findings: int,
    cost_service,
    api_key: str,
) -> AsyncIterator[str]:
    try:
        async for chunk in client.stream_chat_raw(
            messages=messages,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        ):
            # Capture usage from the final chunk (DeepSeek sends it with
            # stream_options={"include_usage": True}) and log the cost.
            if "usage" in chunk and chunk["usage"]:
                u = chunk["usage"]
                prompt_tokens = u.get("prompt_tokens", 0)
                completion_tokens = u.get("completion_tokens", 0)
                prompt_details = u.get("prompt_tokens_details", {}) or {}
                cache_hit = prompt_details.get("cached_tokens", 0) if isinstance(prompt_details, dict) else 0
                cache_miss = prompt_tokens - cache_hit
                cost_result = await _calc_cost_usage(
                    cost_service,
                    body.model,
                    completion_tokens,
                    cache_hit,
                    cache_miss,
                    api_key,
                )
                cost_usd = cost_result.get("costUsd", 0.0)
                log.info(
                    "[CHAT] POST /model/chat  stream=true  model=%s  redacted=%d  in=%d out=%d  cache_hit=%d cache_miss=%d cost_usd=%.8f key_type=%s",
                    body.model,
                    num_findings,
                    prompt_tokens,
                    completion_tokens,
                    cache_hit,
                    cache_miss,
                    cost_usd,
                    key_type(api_key),
                )
            yield f"data: {json.dumps(chunk)}\n\n"
    except LLMAuthError as exc:
        log.warning("[CHAT] POST /model/chat  stream=true  auth_error")
        yield f"data: {json.dumps({'id': 'error', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}], 'error': {'code': 401, 'message': str(exc)}})}\n\n"
    except Exception as exc:
        log.error("[CHAT] POST /model/chat  stream=true  error_type=%s", type(exc).__name__)
        error_chunk: Dict[str, Any] = {
            "id": "error",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "error"}
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        log.info(
            "[CHAT] POST /model/chat  stream=true  model=%s  redacted=%d",
            body.model,
            num_findings,
        )
        yield "data: [DONE]\n\n"


# ── /model/stream (backward-compat alias) ─────────────────────────────

@router.post("/stream")
async def model_stream(
    body: ChatCompletionRequest,
    raw_request: Request,
):
    """Deprecated alias — delegates to /model/chat with stream=true."""
    body.stream = True
    return await model_chat(body, raw_request)
