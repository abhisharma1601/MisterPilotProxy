import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, AuthenticationError, PermissionDeniedError

from ..config import get_config


class LLMAuthError(RuntimeError):
    """Raised when the upstream LLM rejects the API key."""


class DeepSeekClient:
    def __init__(self, api_key: str) -> None:
        config = get_config()
        cfg = config.deepseek
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=api_key or "dummy-key",
            base_url=cfg.base_url,
            timeout=float(cfg.timeout),
            max_retries=0,
        )

    def _safe_error(self, exc: Exception) -> RuntimeError:
        """Re-raise API errors with a safe generic message — never leak upstream details."""
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return LLMAuthError("Invalid API key — check your DeepSeek key.")
        return RuntimeError("LLM request failed")

    async def stream_chat_raw(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a chat completion yielding raw OpenAI chunk dicts.

        Each chunk is a dict with id, object, created, model, choices[].delta,
        finish_reason, and optionally usage — ready for SSE serialization.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, self._cfg.max_retries)):
            try:
                stream = await self._client.chat.completions.create(
                    model=model or self._cfg.model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    yield chunk.model_dump(exclude_none=True)
                return

            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                continue

        raise self._safe_error(last_exc) if last_exc else RuntimeError("stream_chat_raw failed after retries")

    async def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Any:
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, self._cfg.max_retries)):
            try:
                return await self._client.chat.completions.create(
                    model=model or self._cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                continue

        raise self._safe_error(last_exc) if last_exc else RuntimeError("complete_with_tools failed after retries")

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        usage_out: Optional[list] = None,
    ) -> AsyncIterator[str]:
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, self._cfg.max_retries)):
            try:
                stream = await self._client.chat.completions.create(
                    model=model or self._cfg.model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    if usage_out is not None and chunk.usage:
                        usage_out.append({
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                        })
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return

            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                continue

        raise self._safe_error(last_exc) if last_exc else RuntimeError("stream_chat failed after retries")

    async def stream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a tool-enabled LLM call.
        Yields:
          {"type": "content_chunk", "content": str}
          {"type": "complete", "finish_reason": str, "content": str|None,
           "tool_calls": list|None, "usage": usage|None}
        """
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, self._cfg.max_retries)):
            try:
                content_parts: List[str] = []
                tool_calls_acc: Dict[int, Dict[str, str]] = {}
                finish_reason: Optional[str] = None
                usage = None

                stream = await self._client.chat.completions.create(
                    model=model or self._cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream_options={"include_usage": True},
                )

                async for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta

                    if delta.content:
                        content_parts.append(delta.content)
                        yield {"type": "content_chunk", "content": delta.content}

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

                tool_calls = [
                    {
                        "id": tool_calls_acc[i]["id"],
                        "type": "function",
                        "function": {
                            "name": tool_calls_acc[i]["name"],
                            "arguments": tool_calls_acc[i]["arguments"],
                        },
                    }
                    for i in sorted(tool_calls_acc.keys())
                ] if tool_calls_acc else None

                yield {
                    "type": "complete",
                    "finish_reason": finish_reason,
                    "content": "".join(content_parts) or None,
                    "tool_calls": tool_calls,
                    "usage": usage,
                }
                return

            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                continue

        raise self._safe_error(last_exc) if last_exc else RuntimeError("stream_with_tools failed after retries")

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Any:
        """Non-streaming completion, returns the raw OpenAI ChatCompletion object."""
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, self._cfg.max_retries)):
            try:
                return await self._client.chat.completions.create(
                    model=model or self._cfg.model,
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (AuthenticationError, PermissionDeniedError) as exc:
                raise self._safe_error(exc)
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self._cfg.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                continue

        raise self._safe_error(last_exc) if last_exc else RuntimeError("complete failed after retries")

    async def close(self) -> None:
        await self._client.close()


_instances: Dict[str, DeepSeekClient] = {}


def get_deepseek_client(api_key: Optional[str] = None) -> DeepSeekClient:
    global _instances
    key = api_key or get_config().deepseek.api_key or "dummy-key"
    if key not in _instances:
        _instances[key] = DeepSeekClient(api_key=key)
    return _instances[key]
