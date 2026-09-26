"""
OpenAI-compatible chat completions handler for ``POST /v1/chat/completions``.

  - `stream=false` → JSONResponse with a full ChatCompletion object.
  - `stream=true`  → StreamingResponse with OpenAI-format SSE chunks.
  - API key is read from the ``Authorization: Bearer <key>`` header, falling
    back to ``body.apikey``.
  - The model picks the provider (DeepSeek / OpenAI / Claude) through
    :class:`LLMClient`; which keys a model accepts is enforced by
    :func:`open_model_access` (``misterpilot-auto`` refuses BYOK keys).
  - ``misterpilot-auto`` fails over to its next candidate model when the
    routed one fails before sending anything (:func:`_failover`).

PII redaction is applied to user messages before they reach the LLM.
"""

import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ...config import get_config
from ...llm.llm_client import LLMAuthError, LLMRequestError, Provider, output_tokens
from ...logging_config import log_pii_findings
from ...pii import get_pii_pipeline
from ...services import provider_health
from ...services.auto_session import record_spend
from ...services.cost_service import get_cost_service, spawn
from ...services.key_service import KEY_TYPE_MISTERPILOT
from ...services.model_access import ModelAccess, is_auto_model, next_auto_access, open_model_access

log = logging.getLogger("model")


# ── request model (matches OpenAI chat/completions shape) ─────────────

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 8192
    apikey: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

    # Keep client-specific extras (e.g. an editor's "mode") instead of dropping
    # them — the router reads them, and they are never forwarded upstream.
    model_config = {"extra": "allow"}


# ── helpers ───────────────────────────────────────────────────────────

def _raw_key(raw_request: Request, body: ChatCompletionRequest) -> str:
    """The key exactly as the client sent it — header first, then body.apikey.

    Never forwarded upstream as-is: :func:`open_model_access` resolves it
    (``mp-…`` → our provider key) and applies the model's key policy.
    """
    auth = raw_request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return body.apikey or ""


def _auth_failure(access: ModelAccess, exc: LLMAuthError) -> HTTPException:
    """Map a provider auth rejection to the right error for the key type.

    A rejected BYOK key is the user's to fix, so they see the provider's
    message. A rejected MisterPilot request means *our* service key failed —
    the user did nothing wrong, and "check your DeepSeek key" would mislead.
    """
    if access.key_type == KEY_TYPE_MISTERPILOT:
        log.error("[CHAT] provider=%s rejected the MisterPilot service key", access.provider.value)
        return HTTPException(status_code=502, detail="Upstream provider unavailable")
    return HTTPException(status_code=401, detail=str(exc))


def _upstream_error(exc: Exception) -> str:
    """The provider's own error message, for the server log only.

    Clients get a generic error; without this the log shows just the
    exception class, which says nothing about why a request was rejected.
    """
    status = getattr(exc, "status_code", None)
    return f"{status} {exc}" if status else "-"


Usage = tuple[int, int, int, int, int]   # (prompt, completion, cache_hit, cache_miss, cache_write)


def _usage_numbers(usage: Dict[str, Any]) -> Usage:
    """Token counts from an OpenAI-shape usage dict, as the provider reported them.

    ``cache_write_tokens`` is not an OpenAI field: the native Claude transport
    adds it for Anthropic's cache-write premium.
    """
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    cache_hit = details.get("cached_tokens", 0) or 0
    cache_write = details.get("cache_write_tokens", 0) or 0
    return prompt_tokens, completion_tokens, cache_hit, max(0, prompt_tokens - cache_hit - cache_write), cache_write


# ── usage estimation (when the provider never reports usage) ──────────
#
# A stream the client abandons — stop button, closed tab, dropped connection —
# or one that errors midway never delivers its final usage chunk, yet the
# provider bills us for every token it processed. Such requests are charged an
# estimate built to be a LOWER bound, so a user is never overcharged:
#
#   - ~4 characters per token; real code and prose run ~3-4, so this
#     under-counts tokens rather than over-counts them;
#   - input billed at the cache-hit rate — the cheapest — because cache
#     status is unknown. The exception is Claude through Anthropic's
#     OpenAI-compatible endpoint (the native API's fallback): it has no prompt
#     caching, so its input is always a cache miss.

_CHARS_PER_TOKEN = 4


def _input_chars(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> int:
    """Characters of everything sent as input: message text, tool calls, tool schemas."""
    chars = 0
    for m in messages:
        if isinstance(m.get("content"), str):
            chars += len(m["content"])
        if m.get("tool_calls"):
            chars += len(json.dumps(m["tool_calls"]))
    if tools:
        chars += len(json.dumps(tools))
    return chars


def _delta_chars(part: Dict[str, Any]) -> int:
    """Characters of generated output in a message or stream delta."""
    chars = 0
    for key in ("content", "reasoning_content"):
        if isinstance(part.get(key), str):
            chars += len(part[key])
    for call in part.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        for key in ("name", "arguments"):
            if isinstance(fn.get(key), str):
                chars += len(fn[key])
    return chars


def _estimate_usage(access: ModelAccess, input_chars: int, output_chars: int) -> Usage:
    prompt = input_chars // _CHARS_PER_TOKEN
    completion = output_chars // _CHARS_PER_TOKEN
    caches = getattr(access.client, "caches_prompts", access.provider is not Provider.CLAUDE)
    if not caches:
        return prompt, completion, 0, prompt, 0
    return prompt, completion, prompt, 0, 0


class _StreamMeter:
    """What a stream consumed: the provider's usage if it reported any, else
    enough to estimate it."""

    def __init__(self) -> None:
        self.usage: Optional[Dict[str, Any]] = None
        self.chunks = 0
        self.output_chars = 0

    def observe(self, chunk: Dict[str, Any]) -> None:
        self.chunks += 1
        # Usage is cumulative, so the last report is the total. It is billed
        # once, after the stream — never per chunk — so a provider that
        # reports usage on several chunks cannot cause a double charge.
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            self.output_chars += _delta_chars((choice or {}).get("delta") or {})


# ── billing ───────────────────────────────────────────────────────────

async def _bill(
    access: ModelAccess, usage: Usage, *, stream: bool, estimated: bool, num_findings: int
) -> None:
    """Price the request, charge a MisterPilot wallet, and log it. Background
    task: never raises into, or delays, the chat response."""
    prompt_tokens, completion_tokens, cache_hit, cache_miss, cache_write = usage
    # Always billed at the price of the model actually called — for Auto, the
    # model the router picked (a deepseek-flash answer costs deepseek-flash
    # rates), then the usual margin.
    billed_model = access.upstream_model
    try:
        cost = await get_cost_service().calc_cost(
            model=billed_model,
            output=completion_tokens,
            cache_hit=cache_hit,
            cache_miss=cache_miss,
            cache_write=cache_write,
            api_key=access.raw_key,
        )
    except Exception:  # noqa: BLE001
        log.exception(
            "[CHAT] billing failed  model=%s  in=%d out=%d  key_type=%s",
            billed_model, prompt_tokens, completion_tokens, access.key_type,
        )
        return
    cost_usd = cost.get("costUsd", 0.0)
    if access.is_auto:
        record_spend(access.raw_key, cost_usd)     # for the optional daily budget
    _log_usage(access, stream, num_findings, prompt_tokens, completion_tokens,
               cache_hit, cache_miss + cache_write, cost_usd, estimated)


# ── Auto route line ───────────────────────────────────────────────────
#
# Copilot Chat does not display the response's "model" field, so the only way
# to show which model Auto picked is in the reply itself: one italic line at
# the top, e.g.
#
#     _MisterPilot Auto · `claude-sonnet-5` · complexity 12/20_
#
# It is added to streamed replies that answer a new user message only — not to
# every step of an agent's tool loop (one line per turn, not per tool call),
# and not to non-streamed requests, which editors use for side jobs such as
# chat titles where the line would be noise.
#
# The client sends the line back as part of the conversation history. It is
# stripped from assistant messages before they go upstream, so the model never
# sees it — a model that sees its past replies open with a line like this
# starts imitating it.

_ROUTE_LINE_PREFIX = "_MisterPilot Auto · "
_ROUTE_LINE_RE = re.compile(r"\A_MisterPilot Auto · [^\n]*_\n\n")


def _route_line(access: ModelAccess) -> str:
    parts = [f"`{access.upstream_model}`"]
    if access.auto_score is not None:
        parts.append(f"complexity {access.auto_score}/20")
    if access.auto_fallback:
        parts.append("fallback")
    return f"{_ROUTE_LINE_PREFIX}{' · '.join(parts)}_\n\n"


def _failover(access: ModelAccess, exc: Exception) -> Optional[ModelAccess]:
    """Auto only: the next model to try after ``access`` failed before any
    output reached the client, or ``None`` when there is none (or not Auto).

    A provider-side failure (rate limit, outage, our service key refused)
    counts against the model's breaker; a request the provider rejected
    (e.g. too long for its context window) does not — but another model may
    still take it, so it fails over too.
    """
    if not access.is_auto:
        return None
    if not isinstance(exc, LLMRequestError):
        provider_health.record_failure(access.upstream_model)
    log.warning("[CHAT] %s failed before output (%s: %s)", access.upstream_model,
                type(exc).__name__, _upstream_error(exc.__cause__ or exc))
    return next_auto_access(access)


def _wants_route_line(access: ModelAccess, body: ChatCompletionRequest) -> bool:
    """Show the route line on this response?"""
    return (
        access.is_auto
        and body.stream
        and get_config().auto.show_route
        and bool(body.messages)
        and body.messages[-1].role == "user"     # a new turn, not a tool-loop step
    )


def _sanitize_messages(
    messages: List[Message], route: str
) -> tuple[List[Dict[str, Any]], int]:
    pipeline = get_pii_pipeline()
    cleaned: List[Dict[str, Any]] = []
    total_findings = 0

    for m in messages:
        msg: Dict[str, Any] = m.model_dump(exclude_none=True)
        if msg.get("content") is None:
            # An assistant turn with neither text nor tool calls (e.g. a failed
            # reply saved in the client's history) says nothing: drop it.
            # Anything else needs a string — OpenAI rejects a null content.
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                continue
            if msg.get("role") != "assistant":
                msg["content"] = ""
        # Our Auto route line, echoed back in history: not the model's words.
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            msg["content"] = _ROUTE_LINE_RE.sub("", msg["content"], count=1)
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            sanitized, findings = pipeline.redact(msg["content"])
            if findings:
                log_pii_findings(log, route, findings)
                total_findings += len(findings)
            msg["content"] = sanitized
        cleaned.append(msg)

    return cleaned, total_findings


# ── chat completion handler ───────────────────────────────────────────

async def model_chat(
    body: ChatCompletionRequest,
    raw_request: Request,
):
    # Key policy + provider selection. Raises 401/403/400/503 with a precise
    # reason — e.g. a BYOK key on misterpilot-auto is a 403 here, before any
    # upstream call or billing. Only misterpilot-auto gets the payload: it is
    # scored to pick the model. Explicit models skip scoring entirely.
    auto = is_auto_model(body.model)
    payload = body.model_dump(exclude_none=True) if auto else None
    access = await open_model_access(
        body.model, _raw_key(raw_request, body), payload,
        headers=dict(raw_request.headers) if auto else None,
    )

    messages, num_findings = _sanitize_messages(body.messages, "/model/chat")

    if body.stream:
        return StreamingResponse(
            _stream_chunks(access, messages, body, num_findings),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    while True:
        try:
            completion = await access.client.complete(
                messages,
                temperature=body.temperature,
                max_tokens=output_tokens(access.upstream_model, body.max_tokens),
                tools=body.tools or None,
                tool_choice=body.tool_choice,
                effort=access.effort,
            )
            if access.is_auto:
                provider_health.record_success(access.upstream_model)
            break
        except Exception as exc:
            nxt = _failover(access, exc)
            if nxt is not None:
                access = nxt
                continue
            if isinstance(exc, LLMAuthError):
                raise _auth_failure(access, exc)
            log.error("[CHAT] POST /model/chat  stream=false  provider=%s  error_type=%s  upstream=%s",
                      access.provider.value, type(exc).__name__, _upstream_error(exc.__cause__ or exc))
            raise HTTPException(status_code=502, detail="Upstream model error")

    result = completion.model_dump()
    if result.get("usage"):
        usage, estimated = _usage_numbers(result["usage"]), False
    else:
        output_chars = sum(_delta_chars((c or {}).get("message") or {}) for c in result.get("choices") or [])
        usage = _estimate_usage(access, _input_chars(messages, body.tools), output_chars)
        estimated = True
    # Billed in the background: the response never waits on the exchange rate
    # or the wallet.
    spawn(_bill(access, usage, stream=False, estimated=estimated, num_findings=num_findings))
    return JSONResponse(content=result)


def _log_usage(
    access: ModelAccess,
    stream: bool,
    num_findings: int,
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit: int,
    cache_miss: int,
    cost_usd: float,
    estimated: bool,
) -> None:
    log.info(
        "[CHAT] POST /model/chat  stream=%s  model=%s  upstream=%s  provider=%s  redacted=%d"
        "  in=%d out=%d  cache_hit=%d cache_miss=%d cost_usd=%.8f key_type=%s usage=%s",
        "true" if stream else "false",
        access.requested_model,
        access.upstream_model,
        access.provider.value,
        num_findings,
        prompt_tokens,
        completion_tokens,
        cache_hit,
        cache_miss,
        cost_usd,
        access.key_type,
        "estimated" if estimated else "reported",
    )


def _schedule_stream_billing(
    access: ModelAccess,
    meter: _StreamMeter,
    messages: List[Dict[str, Any]],
    body: ChatCompletionRequest,
    num_findings: int,
) -> None:
    """Bill a finished, failed or abandoned stream exactly once."""
    if meter.usage:
        usage, estimated = _usage_numbers(meter.usage), False
    elif meter.chunks:
        # Generation started but no usage arrived (client left, or the stream
        # broke): charge the conservative estimate rather than nothing.
        usage = _estimate_usage(access, _input_chars(messages, body.tools), meter.output_chars)
        estimated = True
        log.warning(
            "[CHAT] stream ended without usage after %d chunk(s); billing an estimate  model=%s",
            meter.chunks, access.upstream_model,
        )
    else:
        return   # nothing came back (e.g. auth failure) — nothing to bill
    spawn(_bill(access, usage, stream=True, estimated=estimated, num_findings=num_findings))


async def _stream_chunks(
    access: ModelAccess,
    messages: List[Dict[str, Any]],
    body: ChatCompletionRequest,
    num_findings: int,
) -> AsyncIterator[str]:
    meter = _StreamMeter()
    wants_line = _wants_route_line(access, body)
    route_line = _route_line(access) if wants_line else None
    try:
        while True:
            started = False
            try:
                async for chunk in access.client.stream_chat_raw(
                    messages,
                    temperature=body.temperature,
                    max_tokens=output_tokens(access.upstream_model, body.max_tokens),
                    tools=body.tools or None,
                    tool_choice=body.tool_choice,
                    effort=access.effort,
                ):
                    started = True
                    meter.observe(chunk)       # before the route line: it isn't model output
                    if route_line and chunk.get("choices"):
                        delta = chunk["choices"][0].setdefault("delta", {})
                        delta["content"] = route_line + (delta.get("content") or "")
                        route_line = None
                    yield f"data: {json.dumps(chunk)}\n\n"
                if access.is_auto:
                    provider_health.record_success(access.upstream_model)
                break
            except Exception as exc:
                # Nothing reached the client yet: Auto can still switch model.
                nxt = None if started else _failover(access, exc)
                if nxt is None:
                    raise
                access = nxt
                route_line = _route_line(access) if wants_line else None
    except LLMAuthError as exc:
        failure = _auth_failure(access, exc)
        log.warning("[CHAT] POST /model/chat  stream=true  auth_error  provider=%s", access.provider.value)
        yield f"data: {json.dumps({'id': 'error', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}], 'error': {'code': failure.status_code, 'message': failure.detail}})}\n\n"
    except Exception as exc:
        log.error("[CHAT] POST /model/chat  stream=true  provider=%s  error_type=%s  upstream=%s",
                  access.provider.value, type(exc).__name__, _upstream_error(exc.__cause__ or exc))
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
        # Runs on success, on error, and when the client disconnects. Billing
        # is scheduled here, never awaited: after a disconnect this generator
        # is being cancelled, and any await in this block would be cancelled
        # with it — losing the charge.
        try:
            _schedule_stream_billing(access, meter, messages, body, num_findings)
        except Exception:  # noqa: BLE001 — must not replace a cancellation in flight
            log.exception("[CHAT] could not schedule stream billing  model=%s", access.upstream_model)

    # Outside the finally: a yield inside it breaks when the stream is being
    # cancelled. Reached on success and on handled errors, not on disconnect
    # (there is no client left to send it to).
    yield "data: [DONE]\n\n"
