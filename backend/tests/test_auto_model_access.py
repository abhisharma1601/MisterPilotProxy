"""MisterPilot Auto through open_model_access: scoring, routing, client choice.

No network: key resolution, the scorer and every provider backend are faked.
Provider backends are swapped at ``llm_client.get_provider_client``, so these
tests exercise the real LLMClient and the real routing path — only the
outermost I/O is replaced.
"""
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.llm import llm_client
from backend.llm.llm_client import Provider
from backend.services import auto_session, model_access, provider_health
from backend.services.complexity_scorer import ComplexityScore, ReasoningDepth, Scope, TaskType
from backend.services.model_access import available_models, next_auto_access, open_model_access

MP_KEY = "mp-test-wallet-key-123"
BYOK_OPENAI = "sk-user-openai-key-456"
FIRST = {"role": "user", "content": "SECRET-PROMPT-TEXT implement oauth"}
# A first message the real history-floor scorer rates low, for tests where
# the history floor must not hold the score up.
SIMPLE_FIRST = {"role": "user", "content": "rename the variable x to count"}
# A Copilot-style chat request: streamed. Non-streamed tool-less requests are
# side jobs (titles, commit messages) and route differently.
PAYLOAD = {"model": "misterpilot-auto", "stream": True, "messages": [FIRST]}

TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}


def chat(*messages, **extra):
    return {"model": "misterpilot-auto", "stream": True, "messages": list(messages), **extra}


def tool_step(first=FIRST):
    return chat(first,
                {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file body"})


def next_turn(first=FIRST, text="now the next part"):
    return chat(first, {"role": "assistant", "content": "done"}, {"role": "user", "content": text})


# ── fakes ─────────────────────────────────────────────────────────────

class FakeBackend:
    """Stands in for ProviderClient."""

    def __init__(self, provider: Provider, key: str) -> None:
        self.provider = provider
        self.key = key
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(provider=self.provider, model=kwargs["model"])

    def stream_chat_raw(self, **kwargs):
        self.calls.append(kwargs)

        async def gen():
            yield {"model": kwargs["model"]}
        return gen()


@pytest.fixture(autouse=True)
def fresh_auto_state():
    """Auto keeps per-chat routing, verdicts, spend and breakers in memory."""
    auto_session._sessions.clear()
    auto_session._spend.clear()
    model_access._verdicts.clear()
    provider_health._states.clear()
    yield
    auto_session._sessions.clear()
    auto_session._spend.clear()
    model_access._verdicts.clear()
    provider_health._states.clear()


@pytest.fixture
def backends(monkeypatch):
    """Replace the provider client factory; record what gets built."""
    built: dict[Provider, list[FakeBackend]] = {p: [] for p in Provider}

    def factory(provider, key):
        backend = FakeBackend(provider, key)
        built[provider].append(backend)
        return backend

    monkeypatch.setattr(llm_client, "get_provider_client", factory)
    return built


@pytest.fixture
def keys(monkeypatch):
    """MisterPilot keys verify and resolve to a per-provider server key."""
    state = SimpleNamespace(resolve_calls=[])

    async def fake_resolve(key, provider="deepseek"):
        state.resolve_calls.append((key, provider))
        if key == "mp-invalid":
            raise HTTPException(status_code=401, detail="Invalid or Low Balance in MisterPilot API key")
        return f"server-{provider}" if key.startswith("mp") else key

    monkeypatch.setattr(model_access, "resolve_api_key", fake_resolve)
    monkeypatch.setattr(model_access, "get_server_key", lambda provider: f"server-{provider}")
    return state


@pytest.fixture
def scorer(monkeypatch):
    """Controls the score the existing scorer + verifier report.

    ``confidence`` defaults below ``auto.verify_min_confidence``, so the AI
    check runs unless a test raises it.
    """
    state = SimpleNamespace(score=5, status="verified", confidence=0.5, calls=0,
                            verifier_keys=[], payloads=[], raise_exc=None)

    def fake_score_request(payload):
        state.calls += 1
        state.payloads.append(payload)
        if state.raise_exc:
            raise state.raise_exc
        return ComplexityScore(
            complexity_score=state.score,
            task_type=TaskType.FEATURE,
            reasoning_depth=ReasoningDepth.MEDIUM,
            scope=Scope.MODULE,
            confidence=state.confidence,
            signals={"prompt": f"prompt-{state.calls}"},
        )

    async def fake_verify(computed, api_key=None):
        state.verifier_keys.append(api_key)
        return SimpleNamespace(complexity_score=state.score, status=state.status)

    monkeypatch.setattr(model_access, "score_request", fake_score_request)
    monkeypatch.setattr(model_access, "verify_score", fake_verify)
    return state


@pytest.fixture
def priced(monkeypatch):
    """Pretend every model has a price, so routing isn't masked by the
    MisterPilot-key pricing rule (tested separately below)."""
    monkeypatch.setattr(model_access, "has_pricing", lambda model: True)


# ── Auto invokes the scorer ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_request_invokes_scorer(backends, keys, scorer, priced):
    scorer.score = 12
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    assert scorer.calls == 1
    assert scorer.payloads[0]["messages"] == PAYLOAD["messages"]
    assert access.upstream_model == "claude-sonnet-5"
    assert access.provider is Provider.CLAUDE
    assert access.auto_score == 12
    assert access.is_auto


@pytest.mark.asyncio
async def test_verifier_uses_our_deepseek_key_not_the_wallet_key(backends, keys, scorer, priced):
    await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    assert scorer.verifier_keys == ["server-deepseek"]


# ── explicit models bypass the scorer ─────────────────────────────────

@pytest.mark.parametrize("model", [
    "gpt-5.5", "gpt-5.4-mini", "gpt-5.3-codex",
    "claude-sonnet-5", "claude-opus-5-5",
    "deepseek-flash", "deepseek-v4-pro",
])
@pytest.mark.asyncio
async def test_explicit_model_bypasses_scorer(backends, keys, scorer, priced, model):
    scorer.raise_exc = AssertionError("scorer must not run for explicit models")
    access = await open_model_access(model, MP_KEY, PAYLOAD)

    assert scorer.calls == 0
    assert access.upstream_model == model
    assert access.auto_score is None
    assert not access.is_auto


# ── the selected model reaches the right existing provider client ─────

@pytest.mark.parametrize("score,model,provider", [
    (2, "deepseek-flash", Provider.DEEPSEEK),
    (5, "deepseek-v4-pro", Provider.DEEPSEEK),
    (8, "gpt-5.4-mini", Provider.OPENAI),
    (12, "claude-sonnet-5", Provider.CLAUDE),
    (16, "gpt-5.5", Provider.OPENAI),
    (19, "claude-opus-5-5", Provider.CLAUDE),
])
@pytest.mark.asyncio
async def test_selected_model_goes_to_correct_provider_client(
    backends, keys, scorer, priced, score, model, provider
):
    scorer.score = score
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    await access.client.complete([{"role": "user", "content": "hi"}], max_tokens=10)

    # Exactly one backend built, for the right provider, with our key for it.
    assert [p for p, b in backends.items() if b] == [provider]
    backend = backends[provider][0]
    assert backend.key == f"server-{provider.value}"
    # The upstream call carries the concrete model — never the Auto alias.
    assert backend.calls[0]["model"] == model
    assert backend.calls[0]["model"] != "misterpilot-auto"


@pytest.mark.asyncio
async def test_auto_streaming_uses_selected_model(backends, keys, scorer, priced):
    scorer.score = 16
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    chunks = [c async for c in access.client.stream_chat_raw([{"role": "user", "content": "hi"}])]
    assert chunks == [{"model": "gpt-5.5"}]


@pytest.mark.parametrize("score,effort", [(11, "medium"), (14, "high"), (8, "low"), (20, "xhigh"), (2, None)])
@pytest.mark.asyncio
async def test_effort_follows_position_in_tier(backends, keys, scorer, priced, score, effort):
    scorer.score = score
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    await access.client.complete([{"role": "user", "content": "hi"}], effort=access.effort)

    assert access.effort == effort
    backend = next(b for bs in backends.values() for b in bs)
    assert backend.calls[0]["effort"] == effort


# ── invalid scores and scorer failures fall back safely ───────────────

@pytest.mark.parametrize("bad", [None, 0, 21, 99, -3, "12", 12.5, True])
@pytest.mark.asyncio
async def test_invalid_score_uses_fallback_model(backends, keys, scorer, priced, bad):
    scorer.score = bad
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    assert access.upstream_model == "deepseek-v4-pro"       # auto.model in config.yaml
    assert access.upstream_model not in ("claude-opus-5-5", "gpt-5.5")
    assert access.auto_score is None


@pytest.mark.asyncio
async def test_scorer_exception_uses_fallback_model(backends, keys, scorer, priced):
    scorer.raise_exc = RuntimeError("scorer exploded")
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    assert access.upstream_model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_missing_provider_service_key_falls_back(backends, keys, scorer, priced, monkeypatch):
    def no_key(provider):
        raise RuntimeError("secret missing")
    monkeypatch.setattr(model_access, "get_server_key", no_key)

    scorer.score = 8                                         # -> gpt-5.4-mini
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    # Only DeepSeek is servable (its key comes from key verification); the
    # tier's DeepSeek alternate serves.
    assert access.upstream_model == "deepseek-v4-pro"
    assert access.auto_fallback
    assert "no openai service key" in access.auto_route


@pytest.mark.asyncio
async def test_unservable_tier_goes_up_a_tier_before_down(backends, keys, scorer, monkeypatch):
    unpriced = {"claude-sonnet-5", "gpt-5.4", "gpt-5.5"}   # the whole 11-14 tier
    monkeypatch.setattr(model_access, "has_pricing", lambda model: model not in unpriced)

    scorer.score = 12
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    assert access.upstream_model == "claude-opus-5-5"         # next tier up, not gpt-5.4-mini
    assert access.auto_fallback


# ── tool loops, side jobs and the AI check ────────────────────────────

@pytest.mark.asyncio
async def test_tool_loop_step_is_pinned_without_rescoring(backends, keys, scorer, priced):
    scorer.score = 12
    first = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    assert first.upstream_model == "claude-sonnet-5"

    scorer.score = 2                                         # would be deepseek-flash
    step = await open_model_access("misterpilot-auto", MP_KEY, tool_step())

    assert step.upstream_model == "claude-sonnet-5"
    assert step.effort == first.effort
    assert "pinned" in step.auto_route
    assert scorer.calls == 1                                 # no scoring, no verifier
    assert len(scorer.verifier_keys) == 1


@pytest.mark.asyncio
async def test_tool_step_without_session_is_scored_but_not_verified(backends, keys, scorer, priced):
    scorer.score = 12
    access = await open_model_access("misterpilot-auto", MP_KEY, tool_step())

    assert scorer.calls == 1
    assert scorer.verifier_keys == []
    assert access.upstream_model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_side_job_skips_verifier_and_session(backends, keys, scorer, priced):
    scorer.score = 3
    side_job = {"model": "misterpilot-auto", "messages": [FIRST]}     # no stream, no tools
    access = await open_model_access("misterpilot-auto", MP_KEY, side_job)

    assert scorer.verifier_keys == []
    assert access.session_key is None
    assert access.upstream_model == "deepseek-flash"


@pytest.mark.asyncio
async def test_side_job_ignores_conversation_floors(backends, keys, scorer, priced):
    scorer.score = 19
    await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)    # chat is on opus

    scorer.score = 2
    title = {"model": "misterpilot-auto", "messages": next_turn()["messages"]}
    access = await open_model_access("misterpilot-auto", MP_KEY, title)

    assert access.upstream_model == "deepseek-flash"


@pytest.mark.parametrize("score,verified", [(12, False), (11, True), (14, True)])
@pytest.mark.asyncio
async def test_confident_score_away_from_tier_edge_skips_ai_check(
    backends, keys, scorer, priced, score, verified
):
    scorer.score, scorer.confidence = score, 0.9
    await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    assert bool(scorer.verifier_keys) is verified


@pytest.mark.asyncio
async def test_verified_verdicts_are_cached_by_request(backends, keys, scorer, priced, monkeypatch):
    monkeypatch.setattr(model_access, "_verdict_key", lambda computed: "same-request")
    await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    await open_model_access("misterpilot-auto", MP_KEY, chat(SIMPLE_FIRST))
    assert len(scorer.verifier_keys) == 1


# ── modes, caps and budgets ───────────────────────────────────────────

@pytest.mark.parametrize("model,expected,effort", [
    ("misterpilot-auto", "claude-sonnet-5", "medium"),        # 12
    ("misterpilot-auto-economy", "gpt-5.4-mini", "medium"),   # 12 - 2 = 10
    ("misterpilot-auto-quality", "claude-sonnet-5", "high"),  # 12 + 2 = 14
])
@pytest.mark.asyncio
async def test_mode_aliases_shift_the_score(backends, keys, scorer, priced, model, expected, effort):
    scorer.score = 12
    access = await open_model_access(model, MP_KEY, {**PAYLOAD, "model": model})

    assert access.upstream_model == expected
    assert access.effort == effort
    assert access.auto_score == 12                            # the real score, unshifted
    assert access.requested_model == model


@pytest.mark.asyncio
async def test_mode_header_and_body_precedence(backends, keys, scorer, priced):
    scorer.score = 12
    by_header = await open_model_access(
        "misterpilot-auto", MP_KEY, PAYLOAD, headers={"X-MisterPilot-Auto-Mode": "economy"})
    assert by_header.upstream_model == "gpt-5.4-mini"

    auto_session._sessions.clear()
    body_wins = await open_model_access(
        "misterpilot-auto", MP_KEY, {**PAYLOAD, "misterpilot": {"mode": "quality"}},
        headers={"X-MisterPilot-Auto-Mode": "economy"})
    assert body_wins.effort == "high"


@pytest.mark.asyncio
async def test_unknown_mode_is_ignored(backends, keys, scorer, priced):
    scorer.score = 12
    access = await open_model_access(
        "misterpilot-auto", MP_KEY, PAYLOAD, headers={"x-misterpilot-auto-mode": "turbo"})
    assert access.upstream_model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_max_model_caps_the_tier(backends, keys, scorer, priced):
    scorer.score = 19
    access = await open_model_access(
        "misterpilot-auto", MP_KEY, PAYLOAD, headers={"X-MisterPilot-Auto-Max-Model": "claude-sonnet-5"})

    assert access.upstream_model == "claude-sonnet-5"
    assert "gpt-5.5" not in access.fallback_models             # above the cap
    assert "claude-opus-5-5" not in access.fallback_models


@pytest.mark.asyncio
async def test_unknown_max_model_is_ignored(backends, keys, scorer, priced):
    scorer.score = 19
    access = await open_model_access(
        "misterpilot-auto", MP_KEY, PAYLOAD, headers={"X-MisterPilot-Auto-Max-Model": "gpt-99"})
    assert access.upstream_model == "claude-opus-5-5"


@pytest.mark.parametrize("budget_inr,expected", [
    (100, "gpt-5.4-mini"),       # ₹100 of ₹100 spent: capped at budget_cap_model
    (120, "gpt-5.5"),            # 83% spent: score 19 - 2 = 17
    (1000, "claude-opus-5-5"),   # 10% spent: untouched
])
@pytest.mark.asyncio
async def test_daily_budget_steps_auto_down(backends, keys, scorer, priced, monkeypatch, budget_inr, expected):
    monkeypatch.setenv("USD_INR_RATE", "100")
    auto_session.record_spend(MP_KEY, 1.0)                    # ₹100 today
    scorer.score = 19
    access = await open_model_access(
        "misterpilot-auto", MP_KEY, PAYLOAD, headers={"X-MisterPilot-Daily-Budget-INR": str(budget_inr)})
    assert access.upstream_model == expected


@pytest.mark.asyncio
async def test_budget_is_per_key(backends, keys, scorer, priced, monkeypatch):
    monkeypatch.setenv("USD_INR_RATE", "100")
    auto_session.record_spend("mp-someone-else", 5.0)
    scorer.score = 19
    access = await open_model_access(
        "misterpilot-auto", MP_KEY, PAYLOAD, headers={"X-MisterPilot-Daily-Budget-INR": "100"})
    assert access.upstream_model == "claude-opus-5-5"


# ── cache-aware routing across turns ──────────────────────────────────

@pytest.mark.asyncio
async def test_downgrade_that_does_not_pay_keeps_warm_model(backends, keys, scorer, priced):
    scorer.score = 19
    await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    scorer.score = 16                                         # gpt-5.5 tier
    access = await open_model_access("misterpilot-auto", MP_KEY, next_turn())

    # Cold gpt-5.5 (full input, pricier output) costs more than warm Opus.
    assert access.upstream_model == "claude-opus-5-5"
    assert "kept warm" in access.auto_route


@pytest.mark.asyncio
async def test_downgrade_that_pays_switches(backends, keys, scorer, priced):
    scorer.score = 12
    await open_model_access("misterpilot-auto", MP_KEY, chat(SIMPLE_FIRST))

    scorer.score = 8                                          # sticky floor 12 - 3 = 9
    access = await open_model_access("misterpilot-auto", MP_KEY, next_turn(SIMPLE_FIRST))
    assert access.upstream_model == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_upgrade_always_switches(backends, keys, scorer, priced):
    scorer.score = 8
    await open_model_access("misterpilot-auto", MP_KEY, chat(SIMPLE_FIRST))

    scorer.score = 19
    access = await open_model_access("misterpilot-auto", MP_KEY, next_turn(SIMPLE_FIRST))
    assert access.upstream_model == "claude-opus-5-5"


@pytest.mark.asyncio
async def test_claude_effort_does_not_drop_within_a_chat(backends, keys, scorer, priced):
    scorer.score = 14                                         # sonnet, high
    await open_model_access("misterpilot-auto", MP_KEY, chat(SIMPLE_FIRST))

    scorer.score = 12                                         # sonnet, medium on its own
    access = await open_model_access("misterpilot-auto", MP_KEY, next_turn(SIMPLE_FIRST))

    assert access.upstream_model == "claude-sonnet-5"
    assert access.effort == "high"


@pytest.mark.asyncio
async def test_session_sticky_floor_limits_the_drop_per_turn(backends, keys, scorer, priced):
    scorer.score = 19
    await open_model_access("misterpilot-auto", MP_KEY, chat(SIMPLE_FIRST))

    scorer.score = 1
    access = await open_model_access("misterpilot-auto", MP_KEY, next_turn(SIMPLE_FIRST))
    assert "sticky floor" in access.auto_route


@pytest.mark.asyncio
async def test_chats_are_separate_sessions(backends, keys, scorer, priced):
    scorer.score = 12
    first = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    other = await open_model_access("misterpilot-auto", MP_KEY, chat(SIMPLE_FIRST))
    other_key = await open_model_access("misterpilot-auto", "mp-another-key", PAYLOAD)

    assert len({first.session_key, other.session_key, other_key.session_key}) == 3


# ── failover and breakers ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_breaker_skips_model(backends, keys, scorer, priced):
    for _ in range(3):                                        # auto.breaker_failures
        provider_health.record_failure("claude-sonnet-5")
    scorer.score = 12
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    assert access.upstream_model == "gpt-5.4"                 # the tier's first alternate
    assert "breaker open" in access.auto_route


@pytest.mark.asyncio
async def test_all_breakers_open_still_serves(backends, keys, scorer, priced, monkeypatch):
    monkeypatch.setattr(provider_health, "is_available", lambda model: False)
    scorer.score = 12
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    assert access.upstream_model == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_next_auto_access_fails_over_and_moves_the_session(backends, keys, scorer, priced):
    scorer.score = 12
    access = await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)
    nxt = next_auto_access(access)

    assert nxt is not None
    assert nxt.upstream_model == "gpt-5.4"
    assert nxt.provider is Provider.OPENAI
    assert nxt.auto_fallback
    assert "failover claude-sonnet-5 -> gpt-5.4" in nxt.auto_route
    assert "gpt-5.4" not in nxt.fallback_models

    # The chat's tool loop continues on the model that actually served it.
    step = await open_model_access("misterpilot-auto", MP_KEY, tool_step())
    assert step.upstream_model == "gpt-5.4"


@pytest.mark.asyncio
async def test_next_auto_access_is_none_for_explicit_models(backends, keys, scorer, priced):
    access = await open_model_access("claude-sonnet-5", MP_KEY)
    assert next_auto_access(access) is None


# ── policy runs before any scoring spend ──────────────────────────────

@pytest.mark.parametrize("model", ["misterpilot-auto", "misterpilot-auto-economy", "misterpilot-auto-quality"])
@pytest.mark.asyncio
async def test_auto_with_byok_rejected_before_scoring(backends, keys, scorer, model):
    with pytest.raises(HTTPException) as exc:
        await open_model_access(model, BYOK_OPENAI, PAYLOAD)
    assert exc.value.status_code == 403
    assert scorer.calls == 0
    assert keys.resolve_calls == []


@pytest.mark.asyncio
async def test_invalid_wallet_key_rejected_before_scoring(backends, keys, scorer):
    with pytest.raises(HTTPException) as exc:
        await open_model_access("misterpilot-auto", "mp-invalid", PAYLOAD)
    assert exc.value.status_code == 401
    assert scorer.calls == 0


def test_auto_models_listed_for_wallet_keys_only():
    wallet = {m["id"] for m in available_models(MP_KEY)}
    byok = {m["id"] for m in available_models(BYOK_OPENAI)}
    autos = {"misterpilot-auto", "misterpilot-auto-economy", "misterpilot-auto-quality"}

    assert autos <= wallet
    assert not autos & byok


# ── existing explicit-model behaviour is unchanged ────────────────────

@pytest.mark.asyncio
async def test_explicit_byok_goes_straight_to_provider(backends, keys, scorer):
    access = await open_model_access("gpt-5.5", BYOK_OPENAI)
    await access.client.complete([{"role": "user", "content": "hi"}])

    backend = backends[Provider.OPENAI][0]
    assert backend.key == BYOK_OPENAI                        # user's key, untouched
    assert backend.calls[0]["model"] == "gpt-5.5"
    assert access.key_type == "byok"


@pytest.mark.asyncio
async def test_explicit_deepseek_with_wallet_key(backends, keys, scorer):
    access = await open_model_access("deepseek-v4-pro", MP_KEY)
    await access.client.complete([{"role": "user", "content": "hi"}])

    backend = backends[Provider.DEEPSEEK][0]
    assert backend.key == "server-deepseek"
    assert backend.calls[0]["model"] == "deepseek-v4-pro"
    assert access.key_type == "misterpilot"


@pytest.mark.asyncio
async def test_explicit_unpriced_model_on_wallet_key_still_refused(backends, keys, scorer, monkeypatch):
    """A supported model that somehow lacks a price is never billed to a
    MisterPilot key: the explicit path refuses (403); only Auto falls back."""
    monkeypatch.setattr(model_access, "has_pricing", lambda model: model != "gpt-5.5")
    with pytest.raises(HTTPException) as exc:
        await open_model_access("gpt-5.5", MP_KEY)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_explicit_unknown_model_is_400(backends, keys, scorer):
    with pytest.raises(HTTPException) as exc:
        await open_model_access("not-a-model", MP_KEY)
    assert exc.value.status_code == 400


# ── logging ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_logs_route_without_secrets_or_prompt(backends, keys, scorer, priced, caplog):
    caplog.set_level(logging.INFO, logger="model_access")
    scorer.score = 12
    await open_model_access("misterpilot-auto", MP_KEY, PAYLOAD)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "MisterPilot Auto" in text
    assert "auto_routing=enabled" in text
    assert "score=12" in text
    assert "model=claude-sonnet-5" in text
    assert "provider=claude" in text
    assert "effort=medium" in text
    assert MP_KEY not in text
    assert "server-" not in text
    assert "SECRET-PROMPT-TEXT" not in text
