"""Unit tests for the MisterPilot Auto score -> model mapping."""
import pytest

from backend.config import TierConfig, get_config
from backend.services.auto_router import (
    DEFAULT_TIERS,
    AutoRouter,
    Tier,
    get_auto_router,
    max_effort,
    validate_score,
)

FALLBACK = "deepseek-v4-pro"


@pytest.fixture
def router() -> AutoRouter:
    return AutoRouter(fallback_model=FALLBACK)


# ── score -> model ────────────────────────────────────────────────────

@pytest.mark.parametrize("score,model", [
    (1, "deepseek-flash"),
    (3, "deepseek-flash"),
    (4, "deepseek-v4-pro"),
    (6, "deepseek-v4-pro"),
    (7, "gpt-5.4-mini"),
    (10, "gpt-5.4-mini"),
    (11, "claude-sonnet-5"),
    (14, "claude-sonnet-5"),
    (15, "gpt-5.5"),
    (17, "gpt-5.5"),
    (18, "claude-opus-5-5"),
    (20, "claude-opus-5-5"),
])
def test_score_routes_to_tier(router, score, model):
    decision = router.select_model(score)
    assert decision.model == model
    assert decision.score == score
    assert decision.fallback is False


def test_every_score_has_exactly_one_model(router):
    for score in range(1, 21):
        assert router.select_model(score).fallback is False


def test_codex_is_never_selected_by_score(router):
    """gpt-5.3-codex needs an agentic signal the request doesn't reliably carry."""
    models = {router.select_model(s).model for s in range(1, 21)}
    assert "gpt-5.3-codex" not in models


def test_request_context_does_not_change_routing(router):
    agentic_looking = {"tools": [{"type": "function"}] * 80, "messages": []}
    assert router.select_model(12, agentic_looking).model == router.select_model(12).model


def test_integral_float_is_accepted(router):
    assert router.select_model(12.0).model == "claude-sonnet-5"


# ── invalid scores ────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, 0, -1, 21, 99, 12.5, "12", True, False, [], {}, float("nan")])
def test_invalid_score_falls_back(router, bad):
    decision = router.select_model(bad)
    assert decision.model == FALLBACK
    assert decision.score is None
    assert decision.fallback is True
    assert decision.reason.startswith("invalid score")


def test_out_of_range_is_never_clamped_to_expensive_tier(router):
    """A score of 25 must not become 20 and buy claude-opus."""
    assert router.select_model(25).model == FALLBACK
    assert router.select_model(10_000).model == FALLBACK


@pytest.mark.parametrize("value,expected", [
    (1, 1), (20, 20), (7.0, 7),
    (0, None), (21, None), (True, None), (None, None), ("5", None), (5.5, None),
])
def test_validate_score(value, expected):
    assert validate_score(value) == expected


# ── configuration ─────────────────────────────────────────────────────

def test_fallback_comes_from_config():
    from backend.config import get_config
    assert get_auto_router().fallback_model == get_config().auto.model


def test_default_tiers_cover_1_to_20():
    AutoRouter(fallback_model=FALLBACK, tiers=DEFAULT_TIERS)  # must not raise


@pytest.mark.parametrize("tiers", [
    (Tier(1, 10, "a"),),                                  # stops short of 20
    (Tier(1, 10, "a"), Tier(12, 20, "b")),                # gap at 11
    (Tier(1, 10, "a"), Tier(10, 20, "b")),                # overlap at 10
    (Tier(2, 20, "a"),),                                  # misses 1
    (Tier(1, 20, ""),),                                   # no model
])
def test_bad_tier_tables_are_rejected(tiers):
    with pytest.raises(ValueError):
        AutoRouter(fallback_model=FALLBACK, tiers=tiers)


def test_fallback_model_is_required():
    with pytest.raises(ValueError):
        AutoRouter(fallback_model="")


def test_unknown_effort_level_is_rejected():
    tiers = (Tier(1, 20, "a", efforts=("turbo",)),)
    with pytest.raises(ValueError):
        AutoRouter(fallback_model=FALLBACK, tiers=tiers)


# ── effort ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,effort", [
    (1, None), (6, None),                       # DeepSeek tiers: no effort control
    (7, "low"), (8, "low"), (9, "medium"), (10, "medium"),
    (11, "medium"), (12, "medium"), (13, "high"), (14, "high"),
    (15, "medium"), (16, "medium"), (17, "high"),
    (18, "high"), (19, "high"), (20, "xhigh"),
])
def test_effort_by_position_in_tier(router, score, effort):
    assert router.select_model(score).effort == effort


@pytest.mark.parametrize("a,b,expected", [
    ("low", "high", "high"), ("xhigh", "medium", "xhigh"), (None, "low", "low"), ("high", None, "high"),
])
def test_max_effort(a, b, expected):
    assert max_effort(a, b) == expected


# ── modes and caps ────────────────────────────────────────────────────

def test_shift_moves_the_score_but_reports_the_real_one(router):
    decision = router.select_model(12, shift=-2)
    assert decision.model == "gpt-5.4-mini"
    assert decision.score == 12
    assert "mode -2" in decision.reason


def test_shift_is_clamped_to_the_scale(router):
    assert router.select_model(20, shift=5).model == "claude-opus-5-5"
    assert router.select_model(1, shift=-5).model == "deepseek-flash"


def test_max_model_caps_the_tier(router):
    decision = router.select_model(19, max_model="claude-sonnet-5")
    assert decision.model == "claude-sonnet-5"
    assert "capped at claude-sonnet-5" in decision.reason
    assert decision.effort == "high"                      # top of the capped tier


def test_cap_below_the_score_does_nothing(router):
    assert router.select_model(5, max_model="claude-sonnet-5").model == "deepseek-v4-pro"


# ── failover candidates ───────────────────────────────────────────────

def test_candidates_start_with_alternates_then_go_up_before_down(router):
    candidates = router.select_model(12).candidates
    assert candidates[:2] == ("gpt-5.4", "gpt-5.5")
    assert candidates.index("claude-opus-5-5") < candidates.index("gpt-5.4-mini")
    assert "claude-sonnet-5" not in candidates            # the chosen model isn't repeated


def test_candidates_never_exceed_the_cap(router):
    candidates = router.select_model(19, max_model="claude-sonnet-5").candidates
    assert "claude-opus-5-5" not in candidates
    assert "gpt-5.5" not in candidates                    # primary of a tier above the cap


def test_candidates_end_with_the_fallback(router):
    assert FALLBACK in router.select_model(20).candidates


def test_invalid_score_still_has_candidates(router):
    decision = router.select_model(None)
    assert decision.model == FALLBACK
    assert decision.candidates and FALLBACK not in decision.candidates


@pytest.mark.parametrize("score,distance", [
    (1, 3), (2, 2), (3, 1),          # 1-3: only the top edge counts
    (11, 1), (12, 2), (13, 2), (14, 1),
    (20, 3),                          # 18-20: only the bottom edge counts
])
def test_boundary_distance(router, score, distance):
    assert router.boundary_distance(score) == distance


# ── table from config.yaml ────────────────────────────────────────────

def test_tiers_can_come_from_config(monkeypatch):
    cfg = get_config().auto
    monkeypatch.setattr(cfg, "tiers", [
        TierConfig(low=1, high=10, model="deepseek-flash"),
        TierConfig(low=11, high=20, model="claude-sonnet-5", alternates=["gpt-5.5"], efforts=["low", "high"]),
    ])
    router = get_auto_router()
    decision = router.select_model(20)
    assert decision.model == "claude-sonnet-5"
    assert decision.effort == "high"
    assert decision.candidates[0] == "gpt-5.5"
