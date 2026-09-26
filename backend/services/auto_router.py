"""
MisterPilot Auto router: complexity score -> model.

The only place the score-to-model mapping lives. Given the (already computed
and verified) 1-20 complexity score, :meth:`AutoRouter.select_model` returns
the model that should serve the request, the reasoning effort to run it at,
and an ordered list of candidates to fail over to. It knows nothing about
keys, providers, pricing or HTTP — :mod:`services.model_access` handles those,
and sends the chosen model down the exact path a manually selected model takes.

Routing table
-------------
    score   model             alternates (same class)       effort by position
    1-3     deepseek-flash    gpt-5.4-nano, deepseek-v4-pro -
    4-6     deepseek-v4-pro   gpt-5.4-mini                  -
    7-10    gpt-5.4-mini      deepseek-v4-pro, haiku-4-5    low, medium
    11-14   claude-sonnet-5   gpt-5.4, gpt-5.5              medium, high
    15-17   gpt-5.5           claude-sonnet-5, opus-5-5     medium, high
    18-20   claude-opus-5-5   gpt-5.5, claude-sonnet-5      high, high, xhigh

``auto.tiers`` in ``config.yaml`` replaces it. ``claude-opus-5-5`` is the id
registered in ``llm.llm_client.MODELS`` and the one Anthropic's API accepts.

Effort
------
Reasoning tokens bill as output, so a score at the bottom of a tier runs at
lower effort than one at the top. Models without an effort control
(DeepSeek, Haiku) ignore it; OpenAI caps Claude-only levels at ``high``.

Candidates
----------
Failover order when a model can't serve (unpriced, no service key, breaker
open, or it fails before the first token): the tier's model, its alternates,
then neighbouring tiers nearest first — one tier up before one tier down, so
a hard task is never silently handed to a much weaker model — then the
configured fallback. Tiers above a ``max_model`` cap are never candidates.

Modes and caps
--------------
``shift`` moves the score before the lookup (economy -2, quality +2 by
default); ``max_model`` caps the tier. Both come from the customer, via
:mod:`services.model_access`.

gpt-5.3-codex
-------------
Deliberately never selected. It is meant for agentic coding/editing, but the
request exposes no reliable signal for that: VS Code Copilot Chat sends its
full tool catalogue (60-80 tools) on every request in every mode, and sends no
mode field, so "tools present" does not distinguish an agent loop from a
question. Rather than guess, Auto uses the score alone. ``request_context`` is
accepted so a reliable signal can be wired in here later without changing
callers.

Invalid scores
--------------
``None``, non-integers, booleans and anything outside 1-20 route to the
fallback model (``auto.model`` in ``config.yaml``, the existing Auto default)
— never clamped to an end of the table, so malformed data can't silently buy
the most expensive tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..config import get_config

MIN_SCORE = 1
MAX_SCORE = 20

EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, slots=True)
class Tier:
    """An inclusive score range, the model that serves it, and how."""

    low: int
    high: int
    model: str
    alternates: tuple[str, ...] = ()
    efforts: tuple[str, ...] = ()     # by position in the range, lowest first

    def effort_for(self, score: int) -> Optional[str]:
        if not self.efforts:
            return None
        span = self.high - self.low + 1
        i = (score - self.low) * len(self.efforts) // span
        return self.efforts[max(0, min(i, len(self.efforts) - 1))]


DEFAULT_TIERS: tuple[Tier, ...] = (
    Tier(1, 3, "deepseek-flash", ("gpt-5.4-nano", "deepseek-v4-pro")),
    Tier(4, 6, "deepseek-v4-pro", ("gpt-5.4-mini",)),
    Tier(7, 10, "gpt-5.4-mini", ("deepseek-v4-pro", "claude-haiku-4-5"), ("low", "medium")),
    Tier(11, 14, "claude-sonnet-5", ("gpt-5.4", "gpt-5.5"), ("medium", "high")),
    Tier(15, 17, "gpt-5.5", ("claude-sonnet-5", "claude-opus-5-5"), ("medium", "high")),
    Tier(18, 20, "claude-opus-5-5", ("gpt-5.5", "claude-sonnet-5"), ("high", "high", "xhigh")),
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The router's answer, with enough context to log why."""

    model: str
    score: Optional[int]        # the validated score; None if it was invalid
    fallback: bool              # True when the score could not be used
    reason: str                 # e.g. "tier 11-14" or "invalid score: None"
    effort: Optional[str] = None
    # Failover order after ``model``, best first.
    candidates: tuple[str, ...] = field(default=())


def _check_tiers(tiers: Sequence[Tier]) -> tuple[Tier, ...]:
    """Tiers must cover 1-20 exactly once, in order — no gaps, no overlaps."""
    ordered = tuple(sorted(tiers, key=lambda t: t.low))
    expected = MIN_SCORE
    for tier in ordered:
        if tier.low != expected or tier.high < tier.low or not tier.model:
            raise ValueError(f"routing tiers must cover {MIN_SCORE}-{MAX_SCORE} contiguously; bad tier {tier}")
        bad = [e for e in tier.efforts if e not in EFFORT_ORDER]
        if bad:
            raise ValueError(f"unknown effort level(s) {bad} in tier {tier.low}-{tier.high}")
        expected = tier.high + 1
    if expected != MAX_SCORE + 1:
        raise ValueError(f"routing tiers stop at {expected - 1}, must reach {MAX_SCORE}")
    return ordered


def validate_score(score: Any) -> Optional[int]:
    """The score as an int in 1-20, or ``None`` if it is not one.

    ``bool`` is rejected even though it subclasses ``int`` — ``True`` is not a
    score of 1. Integral floats (``12.0``) are accepted; ``12.5`` is not.
    """
    if isinstance(score, bool) or score is None:
        return None
    if isinstance(score, float):
        if not score.is_integer():
            return None
        score = int(score)
    if not isinstance(score, int):
        return None
    return score if MIN_SCORE <= score <= MAX_SCORE else None


def max_effort(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """The higher of two effort levels (``None`` counts as unset)."""
    if a is None or b is None:
        return a or b
    return a if EFFORT_ORDER.index(a) >= EFFORT_ORDER.index(b) else b


class AutoRouter:
    """Maps a complexity score to a model. Stateless and side-effect free."""

    def __init__(self, fallback_model: str, tiers: Sequence[Tier] = DEFAULT_TIERS) -> None:
        if not fallback_model:
            raise ValueError("fallback_model is required")
        self.fallback_model = fallback_model
        self.tiers = _check_tiers(tiers)

    # ── lookups ──

    def tier_for_score(self, score: int) -> Tier:
        for tier in self.tiers:
            if tier.low <= score <= tier.high:
                return tier
        raise ValueError(score)   # unreachable: _check_tiers covers 1-20

    def tier_index(self, model: str) -> Optional[int]:
        """Index of the tier ``model`` is the primary of, else ``None``."""
        for i, tier in enumerate(self.tiers):
            if tier.model == model:
                return i
        return None

    def boundary_distance(self, score: int) -> int:
        """How many points ``score`` can move before it changes tier."""
        tier = self.tier_for_score(score)
        below = score - tier.low + 1 if tier.low > MIN_SCORE else MAX_SCORE
        above = tier.high - score + 1 if tier.high < MAX_SCORE else MAX_SCORE
        return min(below, above)

    def _cap_index(self, max_model: Optional[str]) -> int:
        cap = self.tier_index(max_model) if max_model else None
        return len(self.tiers) - 1 if cap is None else cap

    def candidates(self, index: int, *, max_model: Optional[str] = None) -> tuple[str, ...]:
        """Failover order for tier ``index``: its models, then nearest tiers."""
        cap = self._cap_index(max_model)
        index = min(index, cap)
        order = [index]
        for step in range(1, len(self.tiers)):
            for j in (index + step, index - step):          # up first, then down
                if 0 <= j <= cap:
                    order.append(j)
        seen: list[str] = []
        for j in order:
            for model in (self.tiers[j].model, *self.tiers[j].alternates):
                above_cap = (self.tier_index(model) or 0) > cap
                if model not in seen and not above_cap:
                    seen.append(model)
        if self.fallback_model not in seen:
            seen.append(self.fallback_model)
        return tuple(seen)

    # ── routing ──

    def select_model(
        self,
        complexity_score: Any,
        request_context: Optional[Mapping[str, Any]] = None,  # hook for gpt-5.3-codex; unused
        *,
        shift: int = 0,
        max_model: Optional[str] = None,
    ) -> RouteDecision:
        score = validate_score(complexity_score)
        if score is None:
            fallback_index = self.tier_index(self.fallback_model)
            candidates = (
                self.candidates(fallback_index, max_model=max_model)
                if fallback_index is not None else (self.fallback_model,)
            )
            return RouteDecision(
                model=self.fallback_model,
                score=None,
                fallback=True,
                reason=f"invalid score: {complexity_score!r}"[:80],
                candidates=tuple(m for m in candidates if m != self.fallback_model),
            )

        effective = max(MIN_SCORE, min(MAX_SCORE, score + shift))
        notes = [f"mode {shift:+d}"] if effective != score else []
        cap = self._cap_index(max_model)
        if effective > self.tiers[cap].high:
            effective = self.tiers[cap].high
            notes.append(f"capped at {self.tiers[cap].model}")

        tier = self.tier_for_score(effective)
        index = self.tiers.index(tier)
        candidates = self.candidates(index, max_model=max_model)
        reason = f"tier {tier.low}-{tier.high}"
        if notes:
            reason = f"{reason} ({', '.join(notes)})"
        return RouteDecision(
            model=tier.model,
            score=score,
            fallback=False,
            reason=reason,
            effort=tier.effort_for(effective),
            candidates=tuple(m for m in candidates if m != tier.model),
        )


def get_auto_router() -> AutoRouter:
    """The router, with the fallback and optional table from ``config.yaml``."""
    cfg = get_config().auto
    tiers = tuple(
        Tier(t.low, t.high, t.model, tuple(t.alternates), tuple(t.efforts)) for t in cfg.tiers
    ) or DEFAULT_TIERS
    return AutoRouter(fallback_model=cfg.model, tiers=tiers)
