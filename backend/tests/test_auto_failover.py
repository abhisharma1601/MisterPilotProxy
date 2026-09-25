"""Auto failover through the real /v1/chat/completions handler.

Key resolution, the scorer and provider backends are faked; routing,
LLMClient, the stream handler, failover and billing bookkeeping run for real.
Wallet charges are captured, never sent.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.routes import model as model_route
from backend.llm import llm_client
from backend.llm.llm_client import LLMRequestError, LLMUnavailableError, Provider
from backend.main import app
from backend.services import auto_session, model_access, provider_health
from backend.services.complexity_scorer import ComplexityScore, ReasoningDepth, Scope, TaskType

MP_KEY = "mp-test-wallet-key-123"


def chunk(content=None, finish=None, usage=None, model="m"):
    c = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": model,
         "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


class Backend:
    """A provider: serves chunks, or fails before (or after) the first."""

    def __init__(self, provider, fail=None, fail_after_first=False):
        self.provider = provider
        self.fail = fail
        self.fail_after_first = fail_after_first
        self.calls = []

    async def stream_chat_raw(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail and not self.fail_after_first:
            raise self.fail
        yield chunk(f"hello from {kwargs['model']}", model=kwargs["model"])
        if self.fail:
            raise self.fail
        yield chunk(finish="stop", usage=USAGE, model=kwargs["model"])

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        completion = MagicMock()
        completion.model_dump.return_value = {
            "id": "c", "object": "chat.completion", "created": 1, "model": kwargs["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": USAGE,
        }
        return completion


@pytest.fixture
def env(monkeypatch):
    """Score 12 (claude-sonnet-5, alternates gpt-5.4 then gpt-5.5)."""
    auto_session._sessions.clear()
    auto_session._spend.clear()
    model_access._verdicts.clear()
    provider_health._states.clear()

    state = SimpleNamespace(backends={}, spawned=[], fail={})

    def factory(provider, key):
        backend = Backend(provider, **state.fail.get(provider, {}))
        state.backends.setdefault(provider, []).append(backend)
        return backend

    async def fake_resolve(key, provider="deepseek"):
        return f"server-{provider}"

    def fake_score(payload):
        return ComplexityScore(12, TaskType.FEATURE, ReasoningDepth.MEDIUM, Scope.MODULE, 0.9, {"prompt": "p"})

    def fake_spawn(coro):
        state.spawned.append(coro)
        coro.close()                       # billing is not under test here

    pipeline = MagicMock()
    pipeline.redact.side_effect = lambda text: (text, [])

    monkeypatch.setattr(llm_client, "get_provider_client", factory)
    monkeypatch.setattr(model_access, "resolve_api_key", fake_resolve)
    monkeypatch.setattr(model_access, "get_server_key", lambda provider: f"server-{provider}")
    monkeypatch.setattr(model_access, "score_request", fake_score)
    monkeypatch.setattr(model_access, "has_pricing", lambda model: True)
    monkeypatch.setattr(model_route, "spawn", fake_spawn)
    monkeypatch.setattr(model_route, "get_pii_pipeline", lambda: pipeline)
    with patch("backend.debug.enabled", return_value=False):
        yield state
    auto_session._sessions.clear()
    provider_health._states.clear()


async def post(body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {MP_KEY}"})


def sse_chunks(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: {")]


STREAM_BODY = {"model": "misterpilot-auto", "stream": True,
               "messages": [{"role": "user", "content": "implement oauth"}]}


@pytest.mark.asyncio
async def test_stream_fails_over_before_first_token(env):
    env.fail[Provider.CLAUDE] = {"fail": LLMUnavailableError("overloaded")}
    response = await post(STREAM_BODY)

    assert response.status_code == 200
    text = response.text
    assert "hello from gpt-5.4" in text
    assert "`gpt-5.4`" in text and "fallback" in text          # route line names the model that served
    assert "data: [DONE]" in text
    assert provider_health._states["claude-sonnet-5"].failures == 1
    # The chat's next tool step stays on the model that served.
    assert auto_session.get_session(auto_session.session_key(MP_KEY, STREAM_BODY)).model == "gpt-5.4"


@pytest.mark.asyncio
async def test_rejected_request_fails_over_without_tripping_the_breaker(env):
    env.fail[Provider.CLAUDE] = {"fail": LLMRequestError("too long")}
    response = await post(STREAM_BODY)

    assert "hello from gpt-5.4" in response.text
    assert "claude-sonnet-5" not in provider_health._states


@pytest.mark.asyncio
async def test_no_failover_after_output_started(env):
    env.fail[Provider.CLAUDE] = {"fail": LLMUnavailableError("dropped"), "fail_after_first": True}
    response = await post(STREAM_BODY)

    chunks = sse_chunks(response.text)
    assert "hello from claude-sonnet-5" in response.text
    assert chunks[-1]["choices"][0]["finish_reason"] == "error"
    assert Provider.OPENAI not in env.backends                 # never switched mid-reply


@pytest.mark.asyncio
async def test_all_candidates_failing_ends_with_an_error_chunk(env):
    down = {"fail": LLMUnavailableError("down")}
    env.fail.update({Provider.CLAUDE: down, Provider.OPENAI: down, Provider.DEEPSEEK: down})
    response = await post(STREAM_BODY)

    assert sse_chunks(response.text)[-1]["choices"][0]["finish_reason"] == "error"
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_non_stream_fails_over(env):
    env.fail[Provider.CLAUDE] = {"fail": LLMUnavailableError("overloaded")}
    body = {**STREAM_BODY, "stream": False, "tools": [{"type": "function", "function": {"name": "x"}}]}
    response = await post(body)

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_explicit_model_does_not_fail_over(env):
    env.fail[Provider.CLAUDE] = {"fail": LLMUnavailableError("overloaded")}
    response = await post({**STREAM_BODY, "model": "claude-sonnet-5", "stream": False})

    assert response.status_code == 502
    assert Provider.OPENAI not in env.backends


@pytest.mark.asyncio
async def test_effort_reaches_the_provider(env):
    await post(STREAM_BODY)
    assert env.backends[Provider.CLAUDE][0].calls[0]["effort"] == "medium"


@pytest.mark.asyncio
async def test_route_line_has_no_cost_text(env):
    response = await post(STREAM_BODY)
    first = sse_chunks(response.text)[0]["choices"][0]["delta"]["content"]
    assert first.startswith("_MisterPilot Auto · `claude-sonnet-5` · complexity 12/20_\n\n")
    assert "₹" not in response.text and "saved" not in response.text


@pytest.mark.asyncio
async def test_successful_stream_is_billed_once(env):
    await post(STREAM_BODY)
    assert len(env.spawned) == 1


# ── usage parsing ─────────────────────────────────────────────────────

def test_usage_numbers_split_cache_writes():
    usage = {"prompt_tokens": 1000, "completion_tokens": 20,
             "prompt_tokens_details": {"cached_tokens": 900, "cache_write_tokens": 60}}
    assert model_route._usage_numbers(usage) == (1000, 20, 900, 40, 60)


def test_usage_numbers_without_details():
    assert model_route._usage_numbers({"prompt_tokens": 10, "completion_tokens": 2}) == (10, 2, 0, 10, 0)
