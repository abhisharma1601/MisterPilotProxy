import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.mark.asyncio
async def test_health_returns_ok():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/chat/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_stream_returns_sse():
    """Verifies that /chat/stream emits SSE events for each chunk and a done event."""

    async def mock_stream(*args, **kwargs) -> AsyncIterator[str]:
        for token in ["Hello", ", ", "world", "!"]:
            yield token

    mock_client = MagicMock()
    mock_client.stream_chat = mock_stream

    with patch("backend.api.routes.chat.get_deepseek_client", return_value=mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chat/stream",
                json={"messages": [{"role": "user", "content": "Hi"}]},
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
async def test_chat_stream_propagates_error():
    """When the LLM client raises, the SSE stream emits an error event."""

    async def failing_stream(*args, **kwargs) -> AsyncIterator[str]:
        raise RuntimeError("connection refused")
        yield  # make it a generator

    mock_client = MagicMock()
    mock_client.stream_chat = failing_stream

    with patch("backend.api.routes.chat.get_deepseek_client", return_value=mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/chat/stream",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

    data_lines = [
        l[6:] for l in response.text.splitlines() if l.startswith("data: ")
    ]
    payloads = [json.loads(d) for d in data_lines if d.strip()]
    types = [p["type"] for p in payloads]

    assert "error" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_chat_stream_injects_system_prompt():
    """system_prompt is prepended as a system message."""
    captured: list = []

    async def capture_stream(messages, **kwargs) -> AsyncIterator[str]:
        captured.extend(messages)
        yield "ok"

    mock_client = MagicMock()
    mock_client.stream_chat = capture_stream

    with patch("backend.api.routes.chat.get_deepseek_client", return_value=mock_client):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "system_prompt": "You are a coding assistant.",
                },
            )

    assert captured[0] == {"role": "system", "content": "You are a coding assistant."}
    assert captured[1]["role"] == "user"
