"""
Claude through Anthropic's native Messages API, behind the OpenAI chunk shape.

Why not the OpenAI-compatible endpoint: it has no prompt caching. Copilot
re-sends its whole context (60-80 tool schemas, the chat, file bodies) on
every step of an agent loop, so on Claude that input was billed in full every
time. Here the same request reads the repeated prefix from Anthropic's cache
at 0.1x the input price.

:class:`AnthropicClient` has the same interface as ``ProviderClient``
(``complete`` / ``stream_chat_raw`` taking OpenAI-style arguments and
producing OpenAI-style results), so nothing above :class:`LLMClient` knows
which API served a Claude request.

Prompt caching
--------------
Three breakpoints, in render order (tools -> system -> messages):

  - the last tool definition: tools rarely change, so they stay cached even
    when the editor rewrites its system prompt;
  - the last system block;
  - automatic caching (top-level ``cache_control``) for the conversation
    tail, which moves forward as the chat grows.

A cache is only as good as the byte-stability of the prefix, so the
OpenAI -> Anthropic translation is deterministic: the same history always
renders the same request.

Thinking blocks
---------------
Claude Sonnet 5 and Opus 5.5 run adaptive thinking; each reply carries signed
``thinking`` blocks that the API expects back, unchanged, when the
conversation continues. OpenAI-format clients never see them and so never
send them back. :class:`_TurnStash` remembers each reply's block layout —
thinking blocks plus where the text and tool calls sat — keyed by its tool-
call ids or text, and scoped to the customer (``tenant``). When the reply
returns in the history it is rebuilt exactly as Claude produced it, which
keeps both the reasoning and the cached prefix intact. A reply the stash
doesn't know (another server instance, expired) is sent without thinking; if
the API rejects that, :class:`LLMClient` retries through the compatible
endpoint.

Opus 5.5 binds thinking blocks to the conversation that produced them. The
``thinking-binding-controls`` beta header makes the API drop a block whose
conversation changed (the editor trimmed history, say) instead of failing the
request.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import anthropic
from openai.types.chat import ChatCompletion

from ..config import ProviderConfig
from .llm_client import LLMAuthError, LLMRequestError, LLMUnavailableError

log = logging.getLogger("llm.anthropic")

# Models that think adaptively and take an effort level. Forced tool use is
# not compatible with thinking, so these get tool_choice "auto" instead.
_ADAPTIVE_THINKING = {"claude-opus-5-5", "claude-sonnet-5"}
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# Thinking blocks bound to their conversation (preserved thinking).
_BOUND_THINKING = {"claude-opus-5-5"}
_BINDING_BETA = "thinking-binding-controls-2026-08-01"

# Same floor as the OpenAI-protocol path: thinking shares the output budget.
_THINKING_MIN_OUTPUT_TOKENS = 32000

_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}

_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")
_DATA_URL_RE = re.compile(r"\Adata:(image/[a-zA-Z0-9.+-]+);base64,(.*)\Z", re.S)
_EPHEMERAL = {"type": "ephemeral"}


# ── thinking-block stash ──────────────────────────────────────────────

class _TurnStash:
    """Recent Claude replies' block layouts, so a replayed history is exact.

    Bounded (LRU + TTL) and in-process: a miss only costs the reasoning and
    cache of that one turn, never correctness.
    """

    def __init__(self, max_items: int = 20_000, ttl_seconds: float = 3600.0) -> None:
        self._items: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._max = max_items
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored, value = item
            if time.monotonic() - stored > self._ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value


_stash = _TurnStash()


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def _turn_key(tenant: str, text: str, tool_ids: List[str]) -> str:
    # Tool-call ids are unique per reply; text-only replies key on their text.
    return _digest(tenant, tool_ids[0]) if tool_ids else _digest(tenant, "text", text.strip())


# ── OpenAI -> Anthropic request ───────────────────────────────────────

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") in (None, "text") and isinstance(p.get("text"), str)
        )
    return ""


def _user_blocks(content: Any) -> List[Dict[str, Any]]:
    """An OpenAI user message's content as Anthropic blocks (text, images)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    blocks: List[Dict[str, Any]] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str) and part["text"].strip():
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            match = _DATA_URL_RE.match(url)
            if match:
                try:
                    base64.b64decode(match.group(2), validate=True)
                except (binascii.Error, ValueError):
                    continue
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": match.group(1), "data": match.group(2)}})
            elif url.startswith(("http://", "https://")):
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _tool_id(raw: Any) -> str:
    return _TOOL_ID_RE.sub("_", str(raw or "")) or f"toolu_{uuid.uuid4().hex[:24]}"


def _tool_input(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _assistant_blocks(msg: Dict[str, Any], tenant: str) -> List[Dict[str, Any]]:
    """An OpenAI assistant message as Anthropic blocks — the stashed original
    (with its thinking) when this reply is one we produced."""
    text = _text_of(msg.get("content"))
    calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
    tool_uses = {
        _tool_id(c.get("id")): {
            "type": "tool_use",
            "id": _tool_id(c.get("id")),
            "name": (c.get("function") or {}).get("name", ""),
            "input": _tool_input((c.get("function") or {}).get("arguments")),
        }
        for c in calls
    }
    ids = list(tool_uses)

    stored = _stash.get(_turn_key(tenant, text, ids))
    if stored and stored["tool_ids"] == ids:
        rebuilt = _rebuild(stored, text, tool_uses)
        if rebuilt is not None:
            return rebuilt

    blocks: List[Dict[str, Any]] = [{"type": "text", "text": text}] if text.strip() else []
    return blocks + list(tool_uses.values())


def _rebuild(stored: Dict[str, Any], text: str, tool_uses: Dict[str, Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Lay the client's text and tool calls back into the stored layout."""
    exact = stored["text_digest"] == _digest(text)
    if not exact and stored["text_digest_stripped"] != _digest(text.strip()):
        return None     # the client changed the reply: not ours any more
    blocks: List[Dict[str, Any]] = []
    pos = 0
    placed_text = False
    for kind, value in stored["layout"]:
        if kind == "thinking":
            blocks.append(value)
        elif kind == "tool_use":
            blocks.append(tool_uses[value])
        elif kind == "text":
            if exact:
                piece, pos = text[pos:pos + value], pos + value
            else:
                # Whitespace was trimmed: keep the text whole in its first slot.
                piece, placed_text = ("" if placed_text else text.strip()), True
            if piece.strip():
                blocks.append({"type": "text", "text": piece})
    return blocks


def to_anthropic_messages(
    messages: List[Dict[str, Any]], tenant: str = ""
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(system blocks, messages)`` for the Messages API. Deterministic."""
    system: List[Dict[str, Any]] = []
    turns: List[Tuple[str, List[Dict[str, Any]]]] = []

    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            text = _text_of(msg.get("content"))
            if text.strip():
                system.append({"type": "text", "text": text})
            continue
        if role == "assistant":
            blocks = _assistant_blocks(msg, tenant)
            out_role = "assistant"
        elif role == "tool":
            blocks = [{
                "type": "tool_result",
                "tool_use_id": _tool_id(msg.get("tool_call_id")),
                "content": _text_of(msg.get("content")) or "(no output)",
            }]
            out_role = "user"
        else:
            blocks = _user_blocks(msg.get("content"))
            out_role = "user"
        if not blocks:
            continue
        if turns and turns[-1][0] == out_role:
            turns[-1][1].extend(blocks)
        else:
            turns.append((out_role, blocks))

    # A trailing assistant turn without tool calls is a prefill, which current
    # Claude models reject.
    while turns and turns[-1][0] == "assistant" and not any(b["type"] == "tool_use" for b in turns[-1][1]):
        turns.pop()
    if not turns or turns[0][0] != "user":
        turns.insert(0, ("user", [{"type": "text", "text": "(continuing the conversation)"}]))

    out: List[Dict[str, Any]] = []
    for role, blocks in turns:
        if role == "user":
            # tool_result blocks must open the user turn that answers them.
            blocks = sorted(blocks, key=lambda b: b["type"] != "tool_result")
        out.append({"role": role, "content": blocks})

    if system:
        system[-1] = {**system[-1], "cache_control": _EPHEMERAL}
    return system, out


def _to_anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        entry: Dict[str, Any] = {
            "name": fn["name"],
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            entry["description"] = fn["description"]
        out.append(entry)
    if out:
        out[-1] = {**out[-1], "cache_control": _EPHEMERAL}
    return out


def _to_anthropic_tool_choice(choice: Any, model: str) -> Optional[Dict[str, Any]]:
    if choice in (None, "auto"):
        return None
    if choice == "none":
        return {"type": "none"}
    if model in _ADAPTIVE_THINKING:
        return {"type": "auto"}         # forced tool use is refused with thinking on
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
        return {"type": "tool", "name": choice["function"]["name"]}
    return None


# ── Anthropic -> OpenAI response ──────────────────────────────────────

def _openai_usage(usage: Dict[str, int]) -> Dict[str, Any]:
    uncached = usage.get("input_tokens", 0) or 0
    read = usage.get("cache_read_input_tokens", 0) or 0
    write = usage.get("cache_creation_input_tokens", 0) or 0
    output = usage.get("output_tokens", 0) or 0
    prompt = uncached + read + write
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        # cached_tokens is OpenAI's field; cache_write_tokens is ours, so
        # billing can charge Anthropic's write premium.
        "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": write},
    }


class _Translator:
    """Turns one Messages API event stream into OpenAI chunk dicts."""

    def __init__(self, model: str) -> None:
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.model = model
        self.blocks: Dict[int, Dict[str, Any]] = {}
        self.tool_index: Dict[int, int] = {}
        self.usage: Dict[str, int] = {}
        self.finish: Optional[str] = None
        self.done = False

    def _chunk(self, delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
        return {
            "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def _merge_usage(self, usage: Any) -> None:
        if usage is None:
            return
        for key, value in usage.model_dump(exclude_none=True).items():
            if isinstance(value, int):
                self.usage[key] = value

    def feed(self, event: Any) -> List[Dict[str, Any]]:
        kind = event.type
        if kind == "message_start":
            self._merge_usage(getattr(event.message, "usage", None))
            return [self._chunk({"role": "assistant", "content": ""})]

        if kind == "content_block_start":
            block = event.content_block.model_dump(exclude_none=True)
            self.blocks[event.index] = block
            if block.get("type") == "tool_use":
                i = len(self.tool_index)
                self.tool_index[event.index] = i
                block["_json"] = ""
                return [self._chunk({"tool_calls": [{
                    "index": i, "id": block["id"], "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }]})]
            if block.get("type") == "text" and block.get("text"):
                return [self._chunk({"content": block["text"]})]
            return []

        if kind == "content_block_delta":
            block = self.blocks.get(event.index)
            delta = event.delta
            if block is None:
                return []
            if delta.type == "text_delta":
                block["text"] = block.get("text", "") + delta.text
                return [self._chunk({"content": delta.text})]
            if delta.type == "input_json_delta":
                block["_json"] = block.get("_json", "") + delta.partial_json
                return [self._chunk({"tool_calls": [{
                    "index": self.tool_index[event.index], "function": {"arguments": delta.partial_json},
                }]})]
            if delta.type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.thinking
            elif delta.type == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.signature
            return []

        if kind == "message_delta":
            self._merge_usage(getattr(event, "usage", None))
            reason = getattr(event.delta, "stop_reason", None)
            if reason:
                self.finish = _FINISH_REASONS.get(reason, "stop")
            return []

        if kind == "message_stop":
            self.done = True
            return [self._chunk({}, self.finish or "stop")]
        return []

    def usage_chunk(self) -> Dict[str, Any]:
        return {
            "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "choices": [], "usage": _openai_usage(self.usage),
        }

    def ordered_blocks(self) -> List[Dict[str, Any]]:
        return [self.blocks[i] for i in sorted(self.blocks)]

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.ordered_blocks() if b.get("type") == "text")

    def tool_calls(self) -> List[Dict[str, Any]]:
        return [
            {"id": b["id"], "type": "function",
             "function": {"name": b.get("name", ""), "arguments": b.get("_json") or "{}"}}
            for b in self.ordered_blocks() if b.get("type") == "tool_use"
        ]

    def stash(self, tenant: str) -> None:
        """Remember this reply's layout if it carries thinking to replay."""
        blocks = self.ordered_blocks()
        if not any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks):
            return
        layout: List[Tuple[str, Any]] = []
        for b in blocks:
            kind = b.get("type")
            if kind in ("thinking", "redacted_thinking"):
                layout.append(("thinking", {k: v for k, v in b.items() if not k.startswith("_")}))
            elif kind == "text":
                layout.append(("text", len(b.get("text", ""))))
            elif kind == "tool_use":
                layout.append(("tool_use", b["id"]))
        text = self.text()
        ids = [b["id"] for b in blocks if b.get("type") == "tool_use"]
        _stash.put(_turn_key(tenant, text, ids), {
            "layout": layout,
            "tool_ids": ids,
            "text_digest": _digest(text),
            "text_digest_stripped": _digest(text.strip()),
        })


# ── transport ─────────────────────────────────────────────────────────

def _safe_error(exc: Exception) -> RuntimeError:
    """Map an SDK error to ours, with a generic message; the cause is chained."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        err: RuntimeError = LLMAuthError("Invalid API key — check your Claude key.")
    elif isinstance(exc, anthropic.APIConnectionError) or (
        isinstance(status, int) and (status >= 500 or status in (408, 409, 429))
    ):
        err = LLMUnavailableError("LLM provider unavailable")
    elif isinstance(exc, anthropic.APIStatusError):
        err = LLMRequestError("LLM request rejected")
    else:
        err = RuntimeError("LLM request failed")
    err.__cause__ = exc
    return err


class AnthropicClient:
    """One Anthropic key: native Messages API calls in OpenAI shape.

    The SDK retries 408/409/429/5xx and connection errors itself
    (``max_retries`` from ``config.yaml``), before any output is produced.
    """

    native = True

    def __init__(self, api_key: str, cfg: ProviderConfig) -> None:
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(cfg.timeout),
            "max_retries": max(0, cfg.max_retries - 1),
        }
        if cfg.native_base_url:
            kwargs["base_url"] = cfg.native_base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _params(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        effort: Optional[str] = None,
        tenant: str = "",
    ) -> Dict[str, Any]:
        system, msgs = to_anthropic_messages(messages, tenant)
        params: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens or 4096,
            "cache_control": _EPHEMERAL,     # automatic breakpoint on the tail
        }
        if system:
            params["system"] = system
        anthropic_tools = _to_anthropic_tools(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools
            choice = _to_anthropic_tool_choice(tool_choice, model)
            if choice:
                params["tool_choice"] = choice
        if model in _ADAPTIVE_THINKING:
            params["thinking"] = {"type": "adaptive"}
            params["max_tokens"] = max(params["max_tokens"], _THINKING_MIN_OUTPUT_TOKENS)
            if effort in _EFFORTS:
                params["output_config"] = {"effort": effort}
        elif temperature is not None:
            params["extra_body"] = {"temperature": temperature}
        if model in _BOUND_THINKING:
            params["extra_headers"] = {"anthropic-beta": _BINDING_BETA}
        return params

    async def stream_chat_raw(self, *, tenant: str = "", **request: Any) -> AsyncIterator[Dict[str, Any]]:
        """OpenAI chunk dicts for one streamed Messages API call, then a usage chunk."""
        params = self._params(tenant=tenant, **request)
        translator = _Translator(params["model"])
        try:
            stream = await self._client.messages.create(**params, stream=True)
        except Exception as exc:  # noqa: BLE001 — mapped to our error types
            raise _safe_error(exc)
        try:
            async for event in stream:
                for chunk in translator.feed(event):
                    yield chunk
        except Exception as exc:  # noqa: BLE001
            raise _safe_error(exc)
        finally:
            await stream.close()
        if translator.done:
            translator.stash(tenant)
        yield translator.usage_chunk()

    async def complete(self, *, tenant: str = "", **request: Any) -> ChatCompletion:
        """A non-streaming result, built from a streamed call.

        Streaming underneath avoids HTTP timeouts on long thinking turns.
        """
        params = self._params(tenant=tenant, **request)
        translator = _Translator(params["model"])
        try:
            stream = await self._client.messages.create(**params, stream=True)
            try:
                async for event in stream:
                    translator.feed(event)
            finally:
                await stream.close()
        except Exception as exc:  # noqa: BLE001
            raise _safe_error(exc)
        translator.stash(tenant)

        message: Dict[str, Any] = {"role": "assistant", "content": translator.text() or None}
        calls = translator.tool_calls()
        if calls:
            message["tool_calls"] = calls
        return ChatCompletion.model_validate({
            "id": translator.id,
            "object": "chat.completion",
            "created": translator.created,
            "model": translator.model,
            "choices": [{"index": 0, "message": message, "finish_reason": translator.finish or "stop"}],
            "usage": _openai_usage(translator.usage),
        })
