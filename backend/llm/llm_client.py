"""
LLM access for every provider.

DeepSeek and OpenAI speak the OpenAI chat-completions protocol, served by
:class:`ProviderClient`. Claude goes through Anthropic's native Messages API
(:mod:`.anthropic_client`, for prompt caching and effort) when
``claude.native`` is on, and otherwise — or when the native API rejects a
request — through Anthropic's OpenAI-compatible endpoint with
:class:`ProviderClient`. Either way callers see OpenAI-shaped chunks.

:class:`LLMClient` is what callers use. It binds one upstream model to one
resolved key: callers pass messages and sampling parameters, never a model
name, so a routing alias such as ``misterpilot-auto`` cannot leak to a
provider, and the model that is billed is always the model that was called.

This module knows nothing about keys or billing. Which key a request may use,
and whether a model may be billed to a MisterPilot key, is decided in
:mod:`services.model_access` before an ``LLMClient`` is built.
"""
from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from ..config import ProviderConfig, get_config

log = logging.getLogger("llm")


class LLMAuthError(RuntimeError):
    """The upstream provider rejected the API key."""


class LLMUnavailableError(RuntimeError):
    """The provider is rate-limiting, overloaded or unreachable.

    Transient and not the request's fault: another model may serve it.
    """


class LLMRequestError(RuntimeError):
    """The provider refused the request itself (a 4xx other than auth)."""


class UnknownModelError(ValueError):
    """The model name maps to no provider we serve."""


class Provider(StrEnum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    CLAUDE = "claude"

    @property
    def display_name(self) -> str:
        return {"deepseek": "DeepSeek", "openai": "OpenAI", "claude": "Claude"}[self.value]


# ── models ────────────────────────────────────────────────────────────

# The ONLY models that can be used, with any key. Anything else is refused
# before a provider is contacted — a new model needs an entry here and a price
# in services/cost_service.py.
MODELS: Dict[str, Provider] = {
    "gpt-5.5": Provider.OPENAI,
    "gpt-5.4": Provider.OPENAI,
    "gpt-5.4-mini": Provider.OPENAI,
    "gpt-5.4-nano": Provider.OPENAI,
    "gpt-5.3-codex": Provider.OPENAI,
    "claude-opus-5-5": Provider.CLAUDE,
    "claude-sonnet-5": Provider.CLAUDE,
    "claude-haiku-4-5": Provider.CLAUDE,
    "deepseek-v4-pro": Provider.DEEPSEEK,
    "deepseek-flash": Provider.DEEPSEEK,
}

def canonical_model(model: str) -> str:
    """The supported model id for ``model``, or :class:`UnknownModelError`.

    Matching ignores case and surrounding whitespace, and the id returned is
    the one in :data:`MODELS` — so "GPT-5.5" is accepted and sent upstream as
    "gpt-5.5", which is what the provider actually recognises.
    """
    name = (model or "").strip().lower()
    if name not in MODELS:
        raise UnknownModelError(model)
    return name


def provider_for(model: str) -> Provider:
    """Which provider serves ``model``. Raises :class:`UnknownModelError`."""
    return MODELS[canonical_model(model)]


# ── Claude request rules ──────────────────────────────────────────────

# Claude 5-generation models reject sampling parameters with a 400
# ("`temperature` is deprecated for this model"). Haiku 4.5 still takes them.
_CLAUDE_NO_SAMPLING = {"claude-opus-5-5", "claude-sonnet-5"}
# Opus 5.5 also rejects forced tool use (Anthropic's tool_choice any/tool,
# which the compat layer maps from OpenAI's "required" / a named function).
_CLAUDE_NO_FORCED_TOOLS = {"claude-opus-5-5"}


# Output budget for each model on client chat requests, whatever the client
# asks for (internal calls such as the complexity verifier set their own).
# A client's small max_tokens (8192 by default) cuts big replies off — and on
# thinking models the thinking eats the same budget. Billing is per token
# used, so a higher ceiling costs nothing unused.
_OUTPUT_TOKENS: Dict[str, int] = {
    "deepseek-v4-pro": 128000,
    "deepseek-flash": 128000,
    "gpt-5.5": 64000,
    "gpt-5.4": 64000,
    "gpt-5.4-mini": 64000,
    "claude-opus-5-5": 64000,
    "claude-sonnet-5": 64000,
    "gpt-5.4-nano": 32000,
    "claude-haiku-4-5": 16384,
    "gpt-5.3-codex": 16384,
}
_DEFAULT_OUTPUT_TOKENS = 8192


def output_tokens(model: str, requested: Optional[int]) -> int:
    """The output budget for ``model``; ``requested`` only for an unlisted one."""
    return _OUTPUT_TOKENS.get(model, requested or _DEFAULT_OUTPUT_TOKENS)


# Models that think/reason before answering and take an effort level.
_THINKING_MODELS = {
    "claude-opus-5-5", "claude-sonnet-5",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.3-codex",
}

# OpenAI reasoning_effort values we send; Claude-only levels are capped.
_OPENAI_EFFORTS = ("low", "medium", "high")


def _openai_effort(effort: Optional[str]) -> Optional[str]:
    if not effort:
        return None
    return effort if effort in _OPENAI_EFFORTS else "high"


def _retryable_status(exc: Exception) -> bool:
    """429, 408, 409 and 5xx: transient, worth a retry or another model."""
    if isinstance(exc, RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status >= 500 or status in (408, 409))


def _adapt_claude_5(params: Dict[str, Any]) -> None:
    """Drop what the Claude 5 models refuse, in place.

    - ``temperature`` / ``top_p``: removed on these models.
    - Forced ``tool_choice``: downgraded to ``"auto"`` on Opus 5.5.
    - A trailing assistant message is a prefill, which these models reject;
      a tool-call turn is kept (its tool result follows it).
    """
    params.pop("temperature", None)
    params.pop("top_p", None)
    if params.get("model") in _CLAUDE_NO_FORCED_TOOLS and params.get("tool_choice") not in (None, "auto", "none"):
        params["tool_choice"] = "auto"
    messages = params.get("messages") or []
    if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
        params["messages"] = messages[:-1]


# ── transport ─────────────────────────────────────────────────────────

class ProviderClient:
    """One provider, one key: an OpenAI-protocol client with retries.

    Retries transient failures — connection errors, timeouts, 429 and 5xx —
    with exponential backoff. Authentication failures raise
    :class:`LLMAuthError` at once; exhausted transient failures raise
    :class:`LLMUnavailableError`, other refusals :class:`LLMRequestError`.
    Messages are generic, so provider error details never reach the caller;
    the upstream exception is chained for the server log.
    """

    native = False

    def __init__(self, provider: Provider, api_key: str, cfg: ProviderConfig) -> None:
        self.provider = provider
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=float(cfg.timeout),
            max_retries=0,          # retries are ours, below
        )

    def _safe_error(self, exc: Optional[Exception]) -> RuntimeError:
        """Re-raise API errors with a safe generic message — never leak upstream details."""
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            err: RuntimeError = LLMAuthError(f"Invalid API key — check your {self.provider.display_name} key.")
        elif exc is None or self._transient(exc):
            err = LLMUnavailableError("LLM provider unavailable")
        elif isinstance(exc, APIStatusError):
            err = LLMRequestError("LLM request rejected")
        else:
            err = RuntimeError("LLM request failed")
        err.__cause__ = exc
        return err

    @staticmethod
    def _transient(exc: Exception) -> bool:
        return isinstance(exc, (APIConnectionError, APITimeoutError)) or _retryable_status(exc)

    def _params(self, *, stream: bool, **request: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {k: v for k, v in request.items() if v is not None}
        effort = params.pop("effort", None)
        if self.provider is Provider.OPENAI:
            # GPT-5.x are reasoning models: they reject ``max_tokens`` (renamed
            # ``max_completion_tokens``) and any temperature but the default.
            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            params.pop("temperature", None)
            # Chat Completions refuses reasoning_effort alongside function tools
            # ("use /v1/responses or set reasoning_effort to 'none'"), and
            # 'none' would switch reasoning off. With tools, keep the default.
            if params.get("model") in _THINKING_MODELS and _openai_effort(effort) and not params.get("tools"):
                params["reasoning_effort"] = _openai_effort(effort)
        elif self.provider is Provider.CLAUDE and params.get("model") in _CLAUDE_NO_SAMPLING:
            _adapt_claude_5(params)
        if self._cfg.system_instruction and params.get("messages"):
            params["messages"] = [
                {"role": "system", "content": self._cfg.system_instruction},
                *params["messages"],
            ]
        params["stream"] = stream
        if stream and self._cfg.include_usage:
            params["stream_options"] = {"include_usage": True}
        return params

    async def _backoff(self, attempt: int) -> None:
        if attempt < self._cfg.max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    async def complete(self, **request: Any) -> Any:
        """Non-streaming completion; returns the raw ``ChatCompletion``."""
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, self._cfg.max_retries)):
            try:
                return await self._client.chat.completions.create(**self._params(stream=False, **request))
            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                if not self._transient(exc):
                    raise self._safe_error(exc)
                last_exc = exc
                await self._backoff(attempt)
        raise self._safe_error(last_exc)

    async def stream_chat_raw(self, **request: Any) -> AsyncIterator[Dict[str, Any]]:
        """Stream OpenAI chunk dicts (id, choices[].delta, usage …), SSE-ready.

        A dropped connection is retried only if nothing has been yielded yet.
        Retrying after the first chunk would restart generation from scratch
        and send the client a second copy of text it already has.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, self._cfg.max_retries)):
            started = False
            try:
                stream = await self._client.chat.completions.create(**self._params(stream=True, **request))
                async for chunk in stream:
                    started = True
                    yield chunk.model_dump(exclude_none=True)
                return
            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                if started or not self._transient(exc):
                    raise self._safe_error(exc)
                last_exc = exc
                await self._backoff(attempt)
        raise self._safe_error(last_exc)


_provider_clients: Dict[tuple[Provider, str, bool], Any] = {}


def get_provider_client(provider: Provider, api_key: str, native: Optional[bool] = None) -> Any:
    """Cached transport per (provider, key, native) — reuses connections.

    Claude gets the native Messages client when ``claude.native`` is on
    (``native=None``) or when asked for explicitly; ``native=False`` always
    returns the OpenAI-compatible :class:`ProviderClient`.
    """
    cfg: ProviderConfig = getattr(get_config(), provider.value)
    use_native = provider is Provider.CLAUDE and (cfg.native if native is None else native)
    cache_key = (provider, api_key, use_native)
    if cache_key not in _provider_clients:
        if use_native:
            from .anthropic_client import AnthropicClient
            _provider_clients[cache_key] = AnthropicClient(api_key, cfg)
        else:
            _provider_clients[cache_key] = ProviderClient(provider, api_key, cfg)
    return _provider_clients[cache_key]


# ── facade ────────────────────────────────────────────────────────────

class LLMClient:
    """One upstream model, one provider, one resolved key.

    Build it with an already-resolved provider key — never a raw ``mp-…``
    token; see :func:`services.model_access.open_model_access`.
    """

    def __init__(
        self, model: str, api_key: str, provider: Optional[Provider] = None, *, tenant: str = ""
    ) -> None:
        if not api_key:
            raise ValueError("LLMClient requires a resolved API key")
        # Enforced here too, so no caller can reach a provider with a model
        # outside MODELS — or with a provider that doesn't serve it.
        self.model = canonical_model(model)
        self.provider = MODELS[self.model]
        if provider is not None and provider is not self.provider:
            raise ValueError(f"{self.model} is served by {self.provider.value}, not {provider.value}")
        self._api_key = api_key
        # Scopes per-conversation state a transport keeps (Claude's replayed
        # thinking blocks) to one customer. An opaque hash, never a key.
        self._tenant = tenant
        self._backend = get_provider_client(self.provider, api_key)

    @property
    def caches_prompts(self) -> bool:
        """Does the provider cache (and report cache hits for) this client's prompts?

        False only for Claude through the OpenAI-compatible endpoint.
        """
        return self.provider is not Provider.CLAUDE or bool(getattr(self._backend, "native", False))

    def _native(self) -> bool:
        return bool(getattr(self._backend, "native", False))

    def _compat(self) -> Any:
        """Claude's OpenAI-compatible transport: the native API's fallback."""
        return get_provider_client(self.provider, self._api_key, native=False)

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        effort: Optional[str] = None,
    ) -> Any:
        """Non-streaming completion; returns the provider's raw ChatCompletion."""
        request: Dict[str, Any] = dict(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools or None,
            tool_choice=tool_choice,
            effort=effort,
        )
        if not self._native():
            return await self._backend.complete(**request)
        try:
            return await self._backend.complete(**request, tenant=self._tenant)
        except LLMRequestError as exc:
            log.warning("claude native API rejected the request (%s); retrying via compat endpoint", exc.__cause__)
            return await self._compat().complete(**request)

    def stream_chat_raw(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        effort: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream OpenAI-format chunk dicts, SSE-ready."""
        request: Dict[str, Any] = dict(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools or None,
            tool_choice=tool_choice,
            effort=effort,
        )
        if not self._native():
            return self._backend.stream_chat_raw(**request)
        return self._stream_native(request)

    async def _stream_native(self, request: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Native Claude stream. A request it rejects before any output is
        retried once through the compatible endpoint."""
        started = False
        try:
            async for chunk in self._backend.stream_chat_raw(**request, tenant=self._tenant):
                started = True
                yield chunk
            return
        except LLMRequestError as exc:
            if started:
                raise
            log.warning("claude native API rejected the request (%s); retrying via compat endpoint", exc.__cause__)
        async for chunk in self._compat().stream_chat_raw(**request):
            yield chunk

    def __repr__(self) -> str:  # never includes the key
        return f"LLMClient(provider={self.provider.value!r}, model={self.model!r})"
