import json
from contextlib import contextmanager
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.llm import llm_client
from backend.llm.llm_client import Provider
from backend.main import app
from backend.pii.pipeline import Finding


# ── helpers ──────────────────────────────────────────────────────────

def _mock_pipeline():
    pipe = MagicMock()
    pipe.redact.return_value = ("safe text", [])
    return pipe


@contextmanager
def _patch_client(client_mock):
    """Serve DeepSeek requests from ``client_mock``.

    Patches the provider client factory rather than anything in the route, so
    key resolution, provider selection and LLMClient all run for real — only
    the upstream provider is replaced. Debug scoring is forced off even if
    DEBUG_ROUTE_REQUEST is set in the shell: it would make a real AI call.
    """
    def factory(provider, key):
        assert provider is Provider.DEEPSEEK, f"unexpected provider {provider}"
        return client_mock

    with (
        patch.object(llm_client, "get_provider_client", factory),
        patch("backend.debug.enabled", return_value=False),
    ):
        yield


@contextmanager
def _patch_deps(client_mock):
    with (
        _patch_client(client_mock),
        patch("backend.api.routes.model.get_pii_pipeline", return_value=_mock_pipeline()),
    ):
        yield


# ── non-streaming /v1/chat/completions ───────────────────────────────

@pytest.mark.asyncio
async def test_model_chat_returns_openai_format():
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1780745059,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 65, "total_tokens": 77},
        "system_fingerprint": "fp_test",
    }

    mock_client = MagicMock()

    async def mock_complete(*args, **kwargs):
        return mock_completion

    mock_client.complete = mock_complete

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
                headers={"Authorization": "Bearer sk-test-key"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_model_chat_missing_api_key_returns_401():
    mock_client = MagicMock()
    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_model_chat_propagates_error():
    mock_client = MagicMock()

    async def failing_complete(*args, **kwargs):
        raise RuntimeError("connection refused")

    mock_client.complete = failing_complete

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"Authorization": "Bearer sk-test-key"},
            )
    assert response.status_code == 502


@pytest.mark.asyncio
async def test_model_chat_sanitizes_pii():
    captured_message = {}

    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "system_fingerprint": "fp",
    }

    mock_client = MagicMock()

    async def capture_complete(messages, **kwargs):
        captured_message["content"] = messages[0]["content"]
        return mock_completion

    mock_client.complete = capture_complete

    pipe = MagicMock()
    pipe.redact.return_value = ("[REDACTED]", [Finding(entity_type="EMAIL", original="user@example.com", placeholder="[REDACTED]", context="...")])

    with (
        _patch_client(mock_client),
        patch("backend.api.routes.model.get_pii_pipeline", return_value=pipe),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "my email is user@example.com"}],
                },
                headers={"Authorization": "Bearer sk-test-key"},
            )

    assert captured_message["content"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_model_chat_sends_the_model_output_limit_not_the_clients():
    captured = {}
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "x", "object": "chat.completion", "created": 1, "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_client = MagicMock()

    async def capture_complete(*args, **kwargs):
        captured.update(kwargs)
        return mock_completion

    mock_client.complete = capture_complete

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4000},
                headers={"Authorization": "Bearer sk-test-key"},
            )

    assert captured["max_tokens"] == 128000


# ── streaming /v1/chat/completions ───────────────────────────────────

@pytest.mark.asyncio
async def test_model_chat_stream_returns_openai_sse():
    mock_chunks = [
        {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m",
         "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    ]

    mock_client = MagicMock()

    async def mock_stream(*args, **kwargs):
        for c in mock_chunks:
            yield c

    mock_client.stream_chat_raw = mock_stream

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                },
                headers={"Authorization": "Bearer sk-test-key"},
            )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: [DONE]" in response.text


# ── /v1/chat/completions alias ───────────────────────────────────────

@pytest.mark.asyncio
async def test_v1_chat_completions_alias():
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "v1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "system_fingerprint": "fp",
    }

    mock_client = MagicMock()

    async def mock_complete(*args, **kwargs):
        return mock_completion

    mock_client.complete = mock_complete

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Authorization": "Bearer sk-test-key"},
            )

    assert response.status_code == 200

