"""ProviderClient request shaping and retries, LLMClient's native Claude
fallback, and the pricing Auto routes on. No network."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from backend.config import ProviderConfig
from backend.llm import llm_client
from backend.llm.llm_client import (
    LLMClient,
    LLMRequestError,
    LLMUnavailableError,
    Provider,
    ProviderClient,
)
from backend.services.cost_service import estimate_turn_usd, price_usd

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
MESSAGES = [{"role": "user", "content": "hi"}]


def provider_client(provider, **cfg):
    return ProviderClient(provider, "sk-test", ProviderConfig(base_url="https://unused/v1", **cfg))


def api_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://unused/v1/chat/completions"))
    return cls("upstream said no", response=response, body=None)


# ── request shaping ───────────────────────────────────────────────────

def test_openai_effort_is_sent_without_tools():
    params = provider_client(Provider.OPENAI)._params(
        stream=False, messages=MESSAGES, model="gpt-5.5", max_tokens=100, effort="medium")
    assert params["reasoning_effort"] == "medium"
    assert "effort" not in params


def test_openai_effort_is_not_sent_with_tools():
    # Chat Completions rejects reasoning_effort alongside function tools.
    params = provider_client(Provider.OPENAI)._params(
        stream=False, messages=MESSAGES, model="gpt-5.5", tools=TOOLS, effort="high")
    assert "reasoning_effort" not in params


def test_claude_only_effort_levels_are_capped_for_openai():
    params = provider_client(Provider.OPENAI)._params(
        stream=False, messages=MESSAGES, model="gpt-5.5", effort="xhigh")
    assert params["reasoning_effort"] == "high"


def test_deepseek_never_gets_effort():
    params = provider_client(Provider.DEEPSEEK)._params(
        stream=False, messages=MESSAGES, model="deepseek-v4-pro", effort="high")
    assert "effort" not in params and "reasoning_effort" not in params


def test_system_instruction_is_prepended():
    client = provider_client(Provider.DEEPSEEK, system_instruction="Reply in the user's language.")
    params = client._params(stream=False, messages=MESSAGES, model="deepseek-flash")
    assert params["messages"] == [{"role": "system", "content": "Reply in the user's language."}, *MESSAGES]


def test_no_system_instruction_leaves_messages_alone():
    params = provider_client(Provider.DEEPSEEK)._params(stream=False, messages=MESSAGES, model="deepseek-flash")
    assert params["messages"] == MESSAGES


def test_deepseek_config_pins_the_language():
    from backend.config import get_config
    assert "language" in get_config().deepseek.system_instruction


# ── retries and error types ───────────────────────────────────────────

def _scripted(client, monkeypatch, outcomes):
    calls = []

    async def create(**params):
        calls.append(params)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def no_sleep(attempt):
        return None

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    monkeypatch.setattr(client, "_backoff", no_sleep)
    return calls


@pytest.mark.asyncio
async def test_rate_limit_is_retried(monkeypatch):
    client = provider_client(Provider.OPENAI)
    calls = _scripted(client, monkeypatch, [api_error(openai.RateLimitError, 429), "ok"])
    assert await client.complete(messages=MESSAGES, model="gpt-5.5") == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_server_error_exhausts_retries_as_unavailable(monkeypatch):
    client = provider_client(Provider.OPENAI, max_retries=2)
    _scripted(client, monkeypatch, [api_error(openai.InternalServerError, 503)] * 2)
    with pytest.raises(LLMUnavailableError) as exc:
        await client.complete(messages=MESSAGES, model="gpt-5.5")
    assert isinstance(exc.value.__cause__, openai.InternalServerError)


@pytest.mark.asyncio
async def test_bad_request_is_not_retried(monkeypatch):
    client = provider_client(Provider.OPENAI)
    calls = _scripted(client, monkeypatch, [api_error(openai.BadRequestError, 400), "ok"])
    with pytest.raises(LLMRequestError):
        await client.complete(messages=MESSAGES, model="gpt-5.5")
    assert len(calls) == 1


# ── LLMClient: native Claude falls back to the compatible endpoint ────

class FakeTransport:
    def __init__(self, native, fail=None, chunks=("a", "b"), fail_after=None):
        self.native = native
        self.fail = fail
        self.chunks = chunks
        self.fail_after = fail_after
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return SimpleNamespace(served_by="native" if self.native else "compat")

    async def stream_chat_raw(self, **kwargs):
        self.calls.append(kwargs)
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise self.fail
            yield c
        if self.fail and self.fail_after is None:
            raise self.fail


@pytest.fixture
def transports(monkeypatch):
    state = SimpleNamespace(native=FakeTransport(True), compat=FakeTransport(False, chunks=("compat",)))

    def factory(provider, key, native=None):
        return state.compat if native is False else state.native

    monkeypatch.setattr(llm_client, "get_provider_client", factory)
    return state


@pytest.mark.asyncio
async def test_native_rejection_retries_through_compat(transports):
    transports.native.fail = LLMRequestError("rejected")
    result = await LLMClient("claude-sonnet-5", "sk-ant", tenant="t").complete(MESSAGES)
    assert result.served_by == "compat"
    assert transports.native.calls[0]["tenant"] == "t"
    assert "tenant" not in transports.compat.calls[0]


@pytest.mark.asyncio
async def test_native_stream_rejection_before_output_retries_through_compat(transports):
    transports.native.fail = LLMRequestError("rejected")
    transports.native.fail_after = 0
    chunks = [c async for c in LLMClient("claude-sonnet-5", "sk-ant").stream_chat_raw(MESSAGES)]
    assert chunks == ["compat"]


@pytest.mark.asyncio
async def test_native_stream_failure_after_output_is_raised(transports):
    transports.native.fail = LLMRequestError("rejected")
    transports.native.fail_after = 1
    with pytest.raises(LLMRequestError):
        _ = [c async for c in LLMClient("claude-sonnet-5", "sk-ant").stream_chat_raw(MESSAGES)]
    assert transports.compat.calls == []


@pytest.mark.asyncio
async def test_native_unavailable_is_not_retried_through_compat(transports):
    # A provider outage is for Auto's failover, not a protocol retry.
    transports.native.fail = LLMUnavailableError("down")
    with pytest.raises(LLMUnavailableError):
        await LLMClient("claude-sonnet-5", "sk-ant").complete(MESSAGES)
    assert transports.compat.calls == []


def test_caches_prompts(transports):
    assert LLMClient("claude-sonnet-5", "sk-ant").caches_prompts
    transports.native = transports.compat                   # compat-only Claude
    assert not LLMClient("claude-sonnet-5", "sk-ant").caches_prompts


def test_get_provider_client_picks_native_claude_from_config():
    from backend.llm.anthropic_client import AnthropicClient
    assert isinstance(llm_client.get_provider_client(Provider.CLAUDE, "sk-ant-x", native=True), AnthropicClient)
    assert isinstance(llm_client.get_provider_client(Provider.CLAUDE, "sk-ant-x", native=False), ProviderClient)
    assert isinstance(llm_client.get_provider_client(Provider.OPENAI, "sk-x", native=True), ProviderClient)


# ── pricing ───────────────────────────────────────────────────────────

def test_claude_cache_writes_bill_at_the_write_rate():
    assert price_usd("claude-sonnet-5", output=0, cache_hit=0, cache_miss=0, cache_write=1_000_000) == pytest.approx(2.5)


def test_models_without_a_write_rate_bill_writes_as_input():
    assert price_usd("gpt-5.5", output=0, cache_hit=0, cache_miss=0, cache_write=1_000_000) == pytest.approx(5.0)


def test_warm_turns_are_cheaper_than_cold():
    warm = estimate_turn_usd("claude-sonnet-5", context_tokens=60_000, output_tokens=1500, warm=True)
    cold = estimate_turn_usd("claude-sonnet-5", context_tokens=60_000, output_tokens=1500, warm=False)
    assert warm < cold
    assert cold == pytest.approx(60_000 * 0.0000025 + 1500 * 0.00001)   # cold Claude pays the write premium
