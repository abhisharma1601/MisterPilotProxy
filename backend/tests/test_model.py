import json
from contextlib import contextmanager
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


# ── helpers ──────────────────────────────────────────────────────────

def _mock_pipeline():
    """Return a mock PII pipeline that passes text through unchanged."""
    pipe = MagicMock()
    pipe.redact.return_value = ("safe text", [])  # sanitized, no findings
    return pipe


@contextmanager
def _patch_deps(client_mock):
    """Patch both get_deepseek_client and get_pii_pipeline together."""
    with (
        patch("backend.api.routes.model.get_deepseek_client", return_value=client_mock),
        patch("backend.api.routes.model.get_pii_pipeline", return_value=_mock_pipeline()),
    ):
        yield


# ── /model/chat ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_chat_returns_openai_format():
    """POST /model/chat returns a full OpenAI ChatCompletion JSON object."""

    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1780745059,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 65,
            "total_tokens": 77,
        },
        "system_fingerprint": "fp_test",
    }

    mock_client = MagicMock()

    async def mock_complete(*args, **kwargs):
        return mock_completion

    mock_client.complete = mock_complete

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/model/chat",
                json={
                    "model": "deepseek-v4-pro",
                    "apikey": "sk-test-key",
                    "prompt": "Hello!",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "deepseek-v4-pro"
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
    assert data["usage"]["prompt_tokens"] == 12


@pytest.mark.asyncio
async def test_model_chat_propagates_error():
    """When the LLM raises, POST /model/chat returns a 502."""

    mock_client = MagicMock()

    async def failing_complete(*args, **kwargs):
        raise RuntimeError("connection refused")

    mock_client.complete = failing_complete

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/model/chat",
                json={
                    "model": "deepseek-v4-pro",
                    "apikey": "sk-test-key",
                    "prompt": "Hi",
                },
            )

    assert response.status_code == 502
    assert "connection refused" in response.json()["detail"]


@pytest.mark.asyncio
async def test_model_chat_sanitizes_pii():
    """The prompt is PII-redacted before the LLM sees it."""

    captured_message = {}

    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
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
    pipe.redact.return_value = ("email@example.com", [])  # "redacted" but same

    with (
        patch("backend.api.routes.model.get_deepseek_client", return_value=mock_client),
        patch("backend.api.routes.model.get_pii_pipeline", return_value=pipe),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/model/chat",
                json={
                    "model": "deepseek-v4-pro",
                    "apikey": "sk-test-key",
                    "prompt": "john.doe@example.com",
                },
            )

    # The pipeline was called with the raw prompt
    pipe.redact.assert_called_once_with("john.doe@example.com")
    # The sanitized version went to the LLM
    assert captured_message["content"] == "email@example.com"


# ── /model/stream ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_stream_returns_sse():
    """POST /model/stream emits SSE chunks and a done event."""

    async def mock_stream(*args, **kwargs) -> AsyncIterator[str]:
        for token in ["Hello", ", ", "world", "!"]:
            yield token

    mock_client = MagicMock()
    mock_client.stream_chat = mock_stream

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/model/stream",
                json={
                    "model": "deepseek-v4-pro",
                    "apikey": "sk-test-key",
                    "prompt": "Hi",
                },
            )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    lines = response.text.splitlines()
    data_lines = [l[6:] for l in lines if l.startswith("data: ")]

    payloads = [json.loads(d) for d in data_lines if d.strip()]
    types = [p["type"] for p in payloads]

    assert "chunk" in types
    assert types[-1] == "done"

    chunks = [p["content"] for p in payloads if p["type"] == "chunk"]
    assert "".join(chunks) == "Hello, world!"


@pytest.mark.asyncio
async def test_model_stream_propagates_error():
    """When the LLM client raises, the SSE stream emits an error event."""

    async def failing_stream(*args, **kwargs) -> AsyncIterator[str]:
        raise RuntimeError("connection refused")
        yield  # make it a generator

    mock_client = MagicMock()
    mock_client.stream_chat = failing_stream

    with _patch_deps(mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/model/stream",
                json={
                    "model": "deepseek-v4-pro",
                    "apikey": "sk-test-key",
                    "prompt": "Hi",
                },
            )

    data_lines = [
        l[6:] for l in response.text.splitlines() if l.startswith("data: ")
    ]
    payloads = [json.loads(d) for d in data_lines if d.strip()]
    types = [p["type"] for p in payloads]

    assert "error" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_model_stream_passes_model_and_key():
    """The provided model name and API key are forwarded to the LLM client."""

    async def capture_stream(messages, model=None, **kwargs) -> AsyncIterator[str]:
        capture_stream.called_model = model
        yield "ok"

    mock_client = MagicMock()
    mock_client.stream_chat = capture_stream

    with (
        patch("backend.api.routes.model.get_deepseek_client") as mock_factory,
        patch("backend.api.routes.model.get_pii_pipeline", return_value=_mock_pipeline()),
    ):
        mock_factory.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/model/stream",
                json={
                    "model": "deepseek-v4-flash",
                    "apikey": "sk-custom-key-123",
                    "prompt": "Hello",
                },
            )

    # The factory should have been called with the user's API key
    mock_factory.assert_called_once_with("sk-custom-key-123")
    # The model should be forwarded to stream_chat
    assert capture_stream.called_model == "deepseek-v4-flash"
