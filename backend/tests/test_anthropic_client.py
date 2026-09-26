"""Native Claude transport: OpenAI <-> Messages API translation, caching
markers, thinking-block replay and stream translation.

No network: the translation is pure, and streams are fed fake SDK events.
"""
import json
from types import SimpleNamespace

import pytest

from backend.config import ProviderConfig
from backend.llm import anthropic_client
from backend.llm.anthropic_client import (
    AnthropicClient,
    _to_anthropic_tool_choice,
    _to_anthropic_tools,
    _Translator,
    _TurnStash,
    to_anthropic_messages,
)
from backend.llm.llm_client import LLMRequestError

EPHEMERAL = {"type": "ephemeral"}


@pytest.fixture(autouse=True)
def fresh_stash(monkeypatch):
    monkeypatch.setattr(anthropic_client, "_stash", _TurnStash())


class Dumpable(SimpleNamespace):
    """A fake SDK model: attributes plus ``model_dump``."""

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def ev(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def tool_turn_events(text="Let me look.", tool_id="toolu_01", args=('{"path":', ' "a.py"}')):
    """A reply: thinking, text, one tool call — as the Messages API streams it."""
    events = [
        ev("message_start", message=SimpleNamespace(usage=Dumpable(
            input_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=50, output_tokens=1))),
        ev("content_block_start", index=0, content_block=Dumpable(type="thinking", thinking="", signature="")),
        ev("content_block_delta", index=0, delta=SimpleNamespace(type="thinking_delta", thinking="hmm")),
        ev("content_block_delta", index=0, delta=SimpleNamespace(type="signature_delta", signature="SIG")),
        ev("content_block_stop", index=0),
        ev("content_block_start", index=1, content_block=Dumpable(type="text", text="", citations=None)),
        ev("content_block_delta", index=1, delta=SimpleNamespace(type="text_delta", text=text)),
        ev("content_block_stop", index=1),
        ev("content_block_start", index=2, content_block=Dumpable(type="tool_use", id=tool_id, name="read_file", input={})),
        *[ev("content_block_delta", index=2, delta=SimpleNamespace(type="input_json_delta", partial_json=a)) for a in args],
        ev("content_block_stop", index=2),
        ev("message_delta", delta=SimpleNamespace(stop_reason="tool_use"), usage=Dumpable(output_tokens=42)),
        ev("message_stop"),
    ]
    return events


def history_with_reply(text="Let me look.", tool_id="toolu_01", arguments='{"path": "a.py"}'):
    return [
        {"role": "system", "content": "You are Copilot."},
        {"role": "user", "content": "open a.py"},
        {"role": "assistant", "content": text, "tool_calls": [
            {"id": tool_id, "type": "function", "function": {"name": "read_file", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": tool_id, "content": "print('hi')"},
    ]


# ── OpenAI messages -> Messages API ───────────────────────────────────

def test_system_messages_become_cached_system_blocks():
    system, messages = to_anthropic_messages([
        {"role": "system", "content": "rules"},
        {"role": "developer", "content": "more rules"},
        {"role": "user", "content": "hi"},
    ])
    assert system == [{"type": "text", "text": "rules"},
                      {"type": "text", "text": "more rules", "cache_control": EPHEMERAL}]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_tool_calls_and_results_translate():
    _, messages = to_anthropic_messages(history_with_reply())
    assistant, results = messages[1], messages[2]

    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_01", "name": "read_file", "input": {"path": "a.py"}},
    ]
    assert results == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "print('hi')"}]}


def test_consecutive_tool_results_and_user_text_share_one_turn_results_first():
    _, messages = to_anthropic_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "user", "content": "also this"},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
    ])
    last = messages[-1]
    assert [b["type"] for b in last["content"]] == ["tool_result", "tool_result", "text"]


def test_trailing_assistant_prefill_is_dropped_and_first_turn_is_user():
    _, messages = to_anthropic_messages([
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prefill"},
    ])
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user"


def test_empty_messages_are_skipped_and_bad_tool_ids_sanitised():
    _, messages = to_anthropic_messages([
        {"role": "user", "content": "   "},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call:1/2", "type": "function", "function": {"name": "x", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "call:1/2", "content": ""},
    ])
    tool_use = messages[1]["content"][0]
    assert tool_use["id"] == "call_1_2"
    assert tool_use["input"] == {}
    assert messages[2]["content"][0]["tool_use_id"] == "call_1_2"
    assert messages[2]["content"][0]["content"] == "(no output)"


def test_images_translate():
    _, messages = to_anthropic_messages([{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}},    # invalid: dropped
    ]}])
    blocks = messages[0]["content"]
    assert blocks[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}
    assert blocks[2] == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
    assert len(blocks) == 3


def test_translation_is_deterministic():
    history = history_with_reply()
    assert to_anthropic_messages(history) == to_anthropic_messages(history)


def test_tools_translate_with_a_cache_breakpoint_on_the_last():
    tools = _to_anthropic_tools([
        {"type": "function", "function": {"name": "a", "description": "A", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "b"}},
        {"type": "not-a-function"},
    ])
    assert tools == [
        {"name": "a", "description": "A", "input_schema": {"type": "object"}},
        {"name": "b", "input_schema": {"type": "object", "properties": {}}, "cache_control": EPHEMERAL},
    ]


@pytest.mark.parametrize("choice,model,expected", [
    (None, "claude-sonnet-5", None),
    ("auto", "claude-sonnet-5", None),
    ("none", "claude-sonnet-5", {"type": "none"}),
    ("required", "claude-sonnet-5", {"type": "auto"}),     # forced tool use + thinking is refused
    ("required", "claude-haiku-4-5", {"type": "any"}),
    ({"type": "function", "function": {"name": "x"}}, "claude-opus-5-5", {"type": "auto"}),
    ({"type": "function", "function": {"name": "x"}}, "claude-haiku-4-5", {"type": "tool", "name": "x"}),
])
def test_tool_choice(choice, model, expected):
    assert _to_anthropic_tool_choice(choice, model) == expected


# ── request parameters ────────────────────────────────────────────────

@pytest.fixture
def client():
    return AnthropicClient("sk-ant-test", ProviderConfig(base_url="https://unused/"))


def test_thinking_models_get_adaptive_thinking_effort_and_caching(client):
    params = client._params(messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-5",
                            temperature=0.7, max_tokens=8192, effort="high")
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    assert params["max_tokens"] == 8192                    # the per-model limit is set by the route
    assert params["cache_control"] == EPHEMERAL
    assert "extra_body" not in params                      # no temperature on Claude 5
    assert "extra_headers" not in params


def test_opus_sends_the_thinking_binding_beta(client):
    params = client._params(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5-5")
    assert params["extra_headers"] == {"anthropic-beta": "thinking-binding-controls-2026-08-01"}


def test_haiku_keeps_temperature_and_skips_effort(client):
    params = client._params(messages=[{"role": "user", "content": "hi"}], model="claude-haiku-4-5",
                            temperature=0.2, max_tokens=100, effort="high")
    assert params["extra_body"] == {"temperature": 0.2}
    assert "thinking" not in params and "output_config" not in params
    assert params["max_tokens"] == 100


def test_unknown_effort_is_not_sent(client):
    params = client._params(messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-5", effort="turbo")
    assert "output_config" not in params


# ── stream translation ────────────────────────────────────────────────

def test_stream_events_become_openai_chunks():
    translator = _Translator("claude-sonnet-5")
    chunks = [c for e in tool_turn_events() for c in translator.feed(e)]
    deltas = [c["choices"][0]["delta"] for c in chunks]

    assert deltas[0] == {"role": "assistant", "content": ""}
    assert {"content": "Let me look."} in deltas
    start = next(d for d in deltas if "tool_calls" in d and "id" in d["tool_calls"][0])
    assert start["tool_calls"][0] == {"index": 0, "id": "toolu_01", "type": "function",
                                      "function": {"name": "read_file", "arguments": ""}}
    args = "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas if "tool_calls" in d)
    assert json.loads(args) == {"path": "a.py"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert not any("hmm" in json.dumps(c) for c in chunks)   # thinking never reaches the client


def test_usage_splits_cache_reads_and_writes():
    translator = _Translator("claude-sonnet-5")
    for e in tool_turn_events():
        translator.feed(e)
    usage = translator.usage_chunk()["usage"]
    assert usage["prompt_tokens"] == 960
    assert usage["completion_tokens"] == 42
    assert usage["prompt_tokens_details"] == {"cached_tokens": 900, "cache_write_tokens": 50}


@pytest.mark.parametrize("reason,finish", [("end_turn", "stop"), ("max_tokens", "length"),
                                           ("refusal", "content_filter"), ("tool_use", "tool_calls")])
def test_stop_reasons_map_to_finish_reasons(reason, finish):
    translator = _Translator("m")
    translator.feed(ev("message_delta", delta=SimpleNamespace(stop_reason=reason), usage=None))
    assert translator.feed(ev("message_stop"))[0]["choices"][0]["finish_reason"] == finish


# ── thinking-block replay ─────────────────────────────────────────────

def _stash_reply(tenant="tenant-a", **kw):
    translator = _Translator("claude-sonnet-5")
    for e in tool_turn_events(**kw):
        translator.feed(e)
    translator.stash(tenant)


def test_our_reply_is_replayed_with_its_thinking():
    _stash_reply()
    _, messages = to_anthropic_messages(history_with_reply(), tenant="tenant-a")
    assert messages[1]["content"] == [
        {"type": "thinking", "thinking": "hmm", "signature": "SIG"},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_01", "name": "read_file", "input": {"path": "a.py"}},
    ]


def test_replay_is_scoped_to_the_customer():
    _stash_reply(tenant="tenant-a")
    _, messages = to_anthropic_messages(history_with_reply(), tenant="tenant-b")
    assert all(b["type"] != "thinking" for b in messages[1]["content"])


def test_edited_reply_is_not_replayed():
    _stash_reply()
    _, messages = to_anthropic_messages(history_with_reply(text="Something else."), tenant="tenant-a")
    assert all(b["type"] != "thinking" for b in messages[1]["content"])


def test_whitespace_trimmed_reply_still_gets_its_thinking():
    _stash_reply(text="Let me look.\n")
    _, messages = to_anthropic_messages(history_with_reply(text="Let me look."), tenant="tenant-a")
    assert messages[1]["content"][0]["type"] == "thinking"
    assert {"type": "text", "text": "Let me look."} in messages[1]["content"]


def test_reply_without_thinking_is_not_stashed():
    translator = _Translator("claude-haiku-4-5")
    for e in [ev("message_start", message=SimpleNamespace(usage=None)),
              ev("content_block_start", index=0, content_block=Dumpable(type="text", text="plain")),
              ev("message_stop")]:
        translator.feed(e)
    translator.stash("tenant-a")
    assert anthropic_client._stash._items == {}


def test_stash_expires_and_is_bounded():
    stash = _TurnStash(max_items=2, ttl_seconds=0)
    stash.put("a", {"x": 1})
    assert stash.get("a") is None                          # expired
    stash = _TurnStash(max_items=2, ttl_seconds=60)
    for key in "abc":
        stash.put(key, {})
    assert stash.get("a") is None and stash.get("c") == {}


# ── client calls ──────────────────────────────────────────────────────

class FakeStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self._events:
            yield e

    async def close(self):
        self.closed = True


def _fake_create(client, monkeypatch, events=None, exc=None):
    seen = {}

    async def create(**params):
        seen.update(params)
        if exc is not None:
            raise exc
        seen["stream_obj"] = FakeStream(events or [])
        return seen["stream_obj"]

    monkeypatch.setattr(client._client.messages, "create", create)
    return seen


@pytest.mark.asyncio
async def test_stream_yields_chunks_then_usage_and_stashes(client, monkeypatch):
    seen = _fake_create(client, monkeypatch, tool_turn_events())
    chunks = [c async for c in client.stream_chat_raw(
        messages=[{"role": "user", "content": "open a.py"}], model="claude-sonnet-5", tenant="tenant-a")]

    assert seen["stream"] is True
    assert seen["stream_obj"].closed
    assert chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    _, messages = to_anthropic_messages(history_with_reply(), tenant="tenant-a")
    assert messages[1]["content"][0]["type"] == "thinking"


@pytest.mark.asyncio
async def test_complete_returns_a_chat_completion(client, monkeypatch):
    _fake_create(client, monkeypatch, tool_turn_events())
    completion = await client.complete(messages=[{"role": "user", "content": "open a.py"}], model="claude-sonnet-5")
    data = completion.model_dump()

    message = data["choices"][0]["message"]
    assert message["content"] == "Let me look."
    assert message["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.py"}'}
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["usage"]["prompt_tokens"] == 960


@pytest.mark.asyncio
async def test_rejected_request_raises_llm_request_error(client, monkeypatch):
    import anthropic
    import httpx2

    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    _fake_create(client, monkeypatch, exc=anthropic.BadRequestError("bad", response=response, body=None))
    with pytest.raises(LLMRequestError):
        await client.complete(messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-5")
