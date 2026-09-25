"""
Model access policy: which key may call which model, and with what client.

:func:`open_model_access` is the single entry point ``model_chat`` uses. Given
the model the client asked for and the key it sent, it either returns a ready
:class:`LLMClient` plus everything billing needs, or raises an HTTP error that
says exactly why not.

Model and key policy
--------------------
============================  =================  ===================
requested model               MisterPilot key    own key (BYOK)
============================  =================  ===================
``misterpilot-auto[-mode]``   yes                **no** (403)
a model in ``llm.MODELS``     yes*               yes
anything else                 **no** (400)       **no** (400)
============================  =================  ===================

Only the models in ``llm.MODELS`` can be used, with any key — a BYOK key does
not unlock other models. Unsupported names are refused before any key check,
secret fetch or provider call.

``misterpilot-auto`` is our managed routing product — we pick the upstream
model and bill for it — so it only runs on a MisterPilot key. Every supported
model accepts either key.

\\* A MisterPilot key is billed from our price table, so the model needs a
price in ``cost_service._PRICING``. Without one the request is refused rather
than charged at a guessed rate.

MisterPilot Auto
----------------
Only Auto runs the scorer. Its flow::

    MisterPilot key verified (and our DeepSeek key obtained for the verifier)
      -> customer preferences: mode, max model, daily budget
      -> tool-loop step with a known session?  -> pinned: same model, no scoring
         otherwise:
           score_request() -> verify_score()   AI check only where it can
                                               change the tier; cached
           -> conversation floors               (not for side jobs)
           -> AutoRouter.select_model(score)    services/auto_router.py
           -> cache-aware downgrade check       keep the warm model unless a
                                                switch pays for itself
      -> first candidate that is priced, has a service key and a closed
         breaker (tier model, alternates, nearest tiers, fallback)
      -> LLMClient(model, key, provider)        same client an explicit
                                                model gets
Explicit models never touch the scorer or the router.

Modes: the ``misterpilot-auto-economy`` / ``misterpilot-auto-quality`` model
aliases, the ``X-MisterPilot-Auto-Mode`` header, or ``misterpilot.mode`` in
the body shift the score (``auto.mode_shifts``). ``X-MisterPilot-Auto-Max-Model``
/ ``misterpilot.max_model`` caps the tier. ``X-MisterPilot-Daily-Budget-INR``
/ ``misterpilot.daily_budget_inr`` steps Auto down as the key's spend today
approaches it — it never refuses a request.

Side jobs — non-streamed requests without tools, which editors send for chat
titles and commit messages — are scored on their own text: no conversation
floors, no AI check, and they don't touch the chat's session.

Failover: :func:`next_auto_access` gives the next candidate when the served
model fails before its first token (see ``api/routes/model.py``).

Auto is billed like any explicit request, at the price of the model that
actually served it (× the usual margin) — so every tier needs a price.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from fastapi import HTTPException

from .. import debug
from ..config import get_config
from ..llm.llm_client import MODELS, LLMClient, Provider, UnknownModelError, canonical_model, provider_for
from . import provider_health
from .auto_context import context_floors, estimate_context_tokens, session_sticky_floor
from .auto_router import AutoRouter, RouteDecision, get_auto_router, max_effort
from .auto_session import (
    TTLStore,
    AutoSession,
    get_session,
    is_tool_step,
    save_route,
    session_key,
    spent_today_usd,
    tenant_of,
    user_turns,
)
from .complexity_scorer import score_request
from .complexity_verifier import VerifiedScore, skip_verification, verify_score
from .cost_service import cached_inr_rate, estimate_turn_usd, has_pricing
from .key_service import (
    KEY_TYPE_BYOK,
    KEY_TYPE_MISTERPILOT,
    get_server_key,
    is_misterpilot_key,
    resolve_api_key,
)

log = logging.getLogger("model_access")

AUTO_MODEL = "misterpilot-auto"
# Auto and its mode aliases -> the mode each one selects (None: default).
AUTO_MODELS: dict[str, Optional[str]] = {
    AUTO_MODEL: None,
    f"{AUTO_MODEL}-economy": "economy",
    f"{AUTO_MODEL}-quality": "quality",
}


@dataclass(frozen=True, slots=True)
class ModelAccess:
    """A resolved, policy-checked route for one request."""

    client: LLMClient
    requested_model: str        # what the client asked for, e.g. "misterpilot-auto"
    upstream_model: str         # what is actually called — and priced
    provider: Provider
    raw_key: str                # as sent; billing identifies the account by it
    key_type: str               # KEY_TYPE_MISTERPILOT | KEY_TYPE_BYOK
    auto_score: Optional[int] = None    # Auto only: the score routed on
    auto_route: Optional[str] = None    # Auto only: why this model was chosen
    auto_fallback: bool = False         # Auto only: not the scored tier's model
    effort: Optional[str] = None        # Auto only: reasoning effort to run at
    fallback_models: tuple[str, ...] = ()   # Auto only: failover order
    session_key: Optional[str] = None   # Auto only: the chat, if tracked
    user_turns: int = 0                 # Auto only: user messages in the chat

    @property
    def is_auto(self) -> bool:
        return is_auto_model(self.requested_model)


def is_auto_model(model: str) -> bool:
    return (model or "").strip().lower() in AUTO_MODELS


def available_models(raw_key: str = "") -> list[dict[str, Any]]:
    """OpenAI ``/v1/models`` entries for the models ``raw_key`` may call.

    Mirrors :func:`open_model_access` without verifying the key or contacting
    a provider: a MisterPilot key sees the Auto models plus every priced
    model, a BYOK key every supported model, and no key the full catalogue.
    """
    misterpilot = is_misterpilot_key(raw_key)
    byok = bool(raw_key) and not misterpilot

    entries: list[dict[str, Any]] = []
    if not byok:
        entries.extend({"id": m, "object": "model", "created": 0, "owned_by": "misterpilot"} for m in AUTO_MODELS)
    for model, provider in MODELS.items():
        if misterpilot and not has_pricing(model):
            continue
        entries.append({"id": model, "object": "model", "created": 0, "owned_by": provider.value})
    return entries


async def open_model_access(
    requested_model: str,
    raw_key: str,
    payload: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> ModelAccess:
    """Apply the key policy for ``requested_model`` and build its client.

    ``payload`` is the request body as a dict and ``headers`` the request
    headers; only Auto reads them (to score the request and read customer
    preferences). Checks run cheapest-first, so a request that policy forbids
    never triggers a key-verification call, a Secrets Manager fetch, or a
    scoring call.
    """
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide it via Authorization: Bearer <key> header or body.apikey.",
        )

    misterpilot = is_misterpilot_key(raw_key)
    auto = is_auto_model(requested_model)

    # 1. misterpilot-auto is a managed product: MisterPilot keys only.
    if auto and not misterpilot:
        raise HTTPException(
            status_code=403,
            detail=(
                f"{AUTO_MODEL} requires a MisterPilot API key. "
                "Your own provider keys work with any specific model instead."
            ),
        )

    if auto:
        return await _open_auto(requested_model, raw_key, payload, headers)

    # 2. Only supported models, whatever the key. The canonical id is what is
    #    sent upstream and priced ("GPT-5.5" -> "gpt-5.5").
    try:
        upstream_model = canonical_model(requested_model)
    except UnknownModelError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported model: {requested_model!r}. Supported models: "
                f"{', '.join([*AUTO_MODELS, *MODELS])}"
            ),
        )
    provider = MODELS[upstream_model]

    # 3. A MisterPilot key is billed by us, so the model must have a price.
    if misterpilot and not has_pricing(upstream_model):
        raise HTTPException(
            status_code=403,
            detail=(
                f"{requested_model} is not yet available on a MisterPilot key. "
                "Use your own provider key for this model."
            ),
        )

    # 4. Resolve the key: verify + swap a MisterPilot key for ours for this
    #    provider; pass a BYOK key straight through.
    upstream_key = await resolve_api_key(raw_key, provider=provider.value)
    if not upstream_key:
        raise HTTPException(status_code=401, detail="Missing API key.")

    return ModelAccess(
        client=LLMClient(upstream_model, upstream_key, provider, tenant=tenant_of(raw_key)),
        requested_model=requested_model,
        upstream_model=upstream_model,
        provider=provider,
        raw_key=raw_key,
        key_type=KEY_TYPE_MISTERPILOT if misterpilot else KEY_TYPE_BYOK,
    )


# ── MisterPilot Auto: customer preferences ────────────────────────────

@dataclass(frozen=True, slots=True)
class AutoPreferences:
    mode: str
    shift: int
    max_model: Optional[str]
    note: str = ""              # e.g. "daily budget 92% used"


def _header(headers: Optional[Mapping[str, str]], name: str) -> Optional[str]:
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == name:
            return value.strip() or None
    return None


def _lower_cap(router: AutoRouter, a: Optional[str], b: Optional[str]) -> Optional[str]:
    """The stricter of two max-model caps."""
    if a is None or b is None:
        return a or b
    ia, ib = router.tier_index(a), router.tier_index(b)
    if ia is None or ib is None:
        return a if ia is not None else b
    return a if ia <= ib else b


def auto_preferences(
    requested_model: str,
    payload: Mapping[str, Any],
    headers: Optional[Mapping[str, str]],
    raw_key: str,
    router: AutoRouter,
) -> AutoPreferences:
    """Mode, tier cap and budget pressure for this request.

    Precedence: body ``misterpilot`` object, then headers, then the model
    alias, then ``auto.default_mode``. Unknown values are ignored, never an
    error — a preference must not fail a request.
    """
    cfg = get_config().auto
    body = payload.get("misterpilot") if isinstance(payload.get("misterpilot"), Mapping) else {}

    mode = next(
        (m for m in (
            body.get("mode") if isinstance(body.get("mode"), str) else None,
            _header(headers, "x-misterpilot-auto-mode"),
            AUTO_MODELS.get((requested_model or "").strip().lower()),
            cfg.default_mode,
        ) if m and m.lower() in cfg.mode_shifts),
        "balanced",
    ).lower()
    shift = cfg.mode_shifts.get(mode, 0)

    max_model: Optional[str] = None
    raw_cap = body.get("max_model") if isinstance(body.get("max_model"), str) else _header(headers, "x-misterpilot-auto-max-model")
    if raw_cap:
        try:
            max_model = canonical_model(raw_cap)
        except UnknownModelError:
            log.info("MisterPilot Auto: ignoring unknown max model %r", raw_cap[:40])

    note = ""
    budget: Optional[float] = None
    raw_budget = body.get("daily_budget_inr", _header(headers, "x-misterpilot-daily-budget-inr"))
    try:
        budget = float(raw_budget) if raw_budget is not None else None
    except (TypeError, ValueError):
        budget = None
    if budget and budget > 0:
        used = spent_today_usd(raw_key) * cached_inr_rate() / budget
        if used >= 1.0:
            max_model = _lower_cap(router, max_model, cfg.budget_cap_model)
            note = f"daily budget reached ({used:.0%})"
        elif used >= cfg.budget_warn_ratio:
            shift += cfg.budget_shift
            note = f"daily budget {used:.0%} used"
    return AutoPreferences(mode=mode, shift=shift, max_model=max_model, note=note)


# ── MisterPilot Auto: scoring ─────────────────────────────────────────

# Verified verdicts by request text: a regenerated reply, or the same
# question in another chat, doesn't pay for (or wait on) a second AI check.
_verdicts: TTLStore[VerifiedScore] = TTLStore(10_000, 3600)
_CACHEABLE_STATUSES = {"verified", "corrected", "clamped", "blocked"}


def _verdict_key(computed: Any) -> str:
    signals = getattr(computed, "signals", {}) or {}
    text = str(signals.get("prompt") or signals.get("prompt_excerpt") or "")
    return hashlib.sha256(f"{computed.complexity_score}\x00{text}".encode("utf-8", "surrogatepass")).hexdigest()


async def _auto_score(
    payload: Optional[Mapping[str, Any]],
    verifier_key: str,
    router: Optional[AutoRouter] = None,
    *,
    verify: bool = True,
) -> Optional[VerifiedScore]:
    """Run the existing scorer (compute, then AI verify). Never raises.

    The AI check runs only where it can matter: it is skipped when asked
    (tool steps, side jobs) and for a confident score more than
    ``auto.verify_boundary_margin`` points from a tier edge. Returns ``None``
    on any failure; the router treats a missing score as invalid and serves
    the fallback model.
    """
    try:
        computed = score_request(dict(payload or {}))
        if not verify:
            return skip_verification(computed, "not needed for this request")
        cfg = get_config().auto
        if (
            router is not None
            and computed.confidence >= cfg.verify_min_confidence
            and router.boundary_distance(computed.complexity_score) > cfg.verify_boundary_margin
        ):
            return skip_verification(computed, "confident and away from a tier edge")
        key = _verdict_key(computed)
        cached = _verdicts.get(key)
        if cached is not None:
            return cached
        verdict = await verify_score(computed, api_key=verifier_key)
        if verdict.status in _CACHEABLE_STATUSES:
            _verdicts.put(key, verdict)
        return verdict
    except Exception:  # noqa: BLE001 — a scoring failure must not fail the request
        log.exception("MisterPilot Auto: scoring failed; using fallback model")
        return None


# ── MisterPilot Auto: serving ─────────────────────────────────────────

def _server_key_or_none(provider: Provider) -> Optional[str]:
    """Our key for ``provider``, or ``None`` if it can't be fetched."""
    try:
        return get_server_key(provider.value)
    except RuntimeError:
        return None   # already logged by key_service


def _servable(model: str, deepseek_key: Optional[str]) -> tuple[Optional[Provider], Optional[str], str]:
    """Can Auto actually serve ``model`` right now? ``(provider, key, problem)``.

    A model needs (a) to be a supported model, (b) a price — Auto bills at
    the served model's own rates — and (c) our key for its provider. Any miss
    returns ``problem`` instead of raising, so the caller can fall back rather
    than fail the user's request.
    """
    try:
        provider = provider_for(model)
    except UnknownModelError:
        return None, None, "unknown model"
    if not has_pricing(model):
        return None, None, "no pricing configured"
    if provider is Provider.DEEPSEEK and deepseek_key:
        return provider, deepseek_key, ""
    key = _server_key_or_none(provider)
    if not key:
        return None, None, f"no {provider.value} service key"
    return provider, key, ""


def _first_servable(
    chain: list[str], deepseek_key: Optional[str]
) -> tuple[Optional[tuple[str, Provider, str, tuple[str, ...]]], list[str]]:
    """The first model in ``chain`` Auto can serve, and notes on those skipped.

    Models with an open breaker are passed over — unless nothing else can
    serve, in which case they are tried anyway rather than failing.
    """
    notes: list[str] = []
    for respect_breaker in (True, False):
        for i, model in enumerate(chain):
            if respect_breaker and not provider_health.is_available(model):
                if f"{model} breaker open" not in notes:
                    notes.append(f"{model} breaker open")
                continue
            provider, key, problem = _servable(model, deepseek_key)
            if provider is not None and key:
                return (model, provider, key, tuple(m for m in chain[i + 1:] if m != model)), notes
            note = f"{model} unavailable ({problem})"
            if note not in notes:
                notes.append(note)
    return None, notes


def _pinned(router: AutoRouter, session: AutoSession) -> RouteDecision:
    """A tool-loop step: stay on the model that started the turn."""
    index = router.tier_index(session.model)
    if index is None:
        index = router.tiers.index(router.tier_for_score(session.score))
    return RouteDecision(
        model=session.model,
        score=session.score,
        fallback=False,
        reason="pinned to this turn's model (tool loop)",
        effort=session.effort,
        candidates=tuple(m for m in router.candidates(index) if m != session.model),
    )


def _is_claude(model: str) -> bool:
    try:
        return provider_for(model) is Provider.CLAUDE
    except UnknownModelError:
        return False


def _cache_aware(
    router: AutoRouter,
    decision: RouteDecision,
    session: AutoSession,
    context_tokens: int,
    prefs: AutoPreferences,
) -> RouteDecision:
    """Keep the chat on its warm model when stepping down would cost more.

    Moving to another model re-sends the whole context uncached (caches are
    per model). Upgrades always go ahead — quality first. A downgrade happens
    only if its cold first turn plus cheaper warm turns beat staying, over
    ``auto.switch_horizon_turns``. On Claude, effort never drops within a
    chat on the same model: changing it invalidates the cached conversation.
    """
    if decision.fallback:
        return decision
    if decision.model == session.model:
        if _is_claude(decision.model):
            return replace(decision, effort=max_effort(decision.effort, session.effort))
        return decision

    new_i, old_i = router.tier_index(decision.model), router.tier_index(session.model)
    if new_i is None or old_i is None or new_i >= old_i:
        return decision
    cap = router.tier_index(prefs.max_model) if prefs.max_model else None
    if cap is not None and old_i > cap:
        return decision                         # the customer's cap wins

    cfg = get_config().auto
    turns = max(1, cfg.switch_horizon_turns)
    out = cfg.expected_output_tokens
    warm_old = estimate_turn_usd(session.model, context_tokens=context_tokens, output_tokens=out, warm=True)
    cold_new = estimate_turn_usd(decision.model, context_tokens=context_tokens, output_tokens=out, warm=False)
    warm_new = estimate_turn_usd(decision.model, context_tokens=context_tokens, output_tokens=out, warm=True)
    stay, switch = turns * warm_old, cold_new + (turns - 1) * warm_new
    if switch < stay:
        return decision

    effort = session.effort if _is_claude(session.model) else router.tiers[old_i].effort_for(
        max(router.tiers[old_i].low, min(router.tiers[old_i].high, decision.score or router.tiers[old_i].low)))
    return RouteDecision(
        model=session.model,
        score=decision.score,
        fallback=False,
        reason=f"{decision.reason}; kept warm {session.model} (switching re-sends ~{context_tokens // 1000}k tokens uncached)",
        effort=effort,
        candidates=tuple(m for m in router.candidates(old_i, max_model=prefs.max_model) if m != session.model),
    )


async def _open_auto(
    requested_model: str,
    raw_key: str,
    payload: Optional[Mapping[str, Any]],
    headers: Optional[Mapping[str, str]] = None,
) -> ModelAccess:
    """Score the request, pick a model, and build the same client an explicit
    selection of that model would get.

    The MisterPilot key is verified *before* scoring, so an invalid or empty
    wallet never costs us a verifier call. That verification is the existing
    ``resolve_api_key`` — it also returns our DeepSeek key, which the AI
    verifier uses and which serves any DeepSeek tier without a second fetch.
    """
    payload = dict(payload or {})
    deepseek_key = await resolve_api_key(raw_key, provider=Provider.DEEPSEEK.value)

    router = get_auto_router()
    prefs = auto_preferences(requested_model, payload, headers, raw_key, router)
    tool_step = is_tool_step(payload)
    side_job = not payload.get("stream") and not payload.get("tools")
    skey = None if side_job else session_key(raw_key, payload)
    session = get_session(skey)
    turns = user_turns(payload)
    context_tokens = estimate_context_tokens(payload)

    verdict: Optional[VerifiedScore] = None
    if tool_step and session is not None:
        decision = _pinned(router, session)
        verification = "pinned"
    else:
        verdict = await _auto_score(payload, deepseek_key, router, verify=not (tool_step or side_job))
        raw_score = verdict.complexity_score if verdict is not None else None
        raised = ""
        score = raw_score
        if not side_job:
            # The conversation can raise the score (never lower it): earlier
            # turns, the previous Auto route, and a long context each set a floor.
            sticky = (
                session_sticky_floor(session.score, turns - session.user_turns)
                if session is not None else None
            )
            score, raised = context_floors(payload, sticky=sticky).apply(raw_score)
        decision = router.select_model(score, shift=prefs.shift, max_model=prefs.max_model)
        if raised:
            decision = replace(decision, reason=f"{decision.reason}; {raised}")
        if session is not None:
            decision = _cache_aware(router, decision, session, context_tokens, prefs)
        verification = verdict.status if verdict is not None else "scoring_failed"

    chosen, skipped = _first_servable([decision.model, *decision.candidates], deepseek_key)
    if chosen is None:
        log.error("MisterPilot Auto: no servable model (%s)", "; ".join(skipped))
        raise HTTPException(status_code=503, detail=f"{AUTO_MODEL} is temporarily unavailable")
    model, provider, upstream_key, remaining = chosen

    route = decision.reason
    if model != decision.model:
        route = f"{route}; {', '.join(skipped)} -> {model}"
    if prefs.note:
        route = f"{route}; {prefs.note}"
    fallback = decision.fallback or model != decision.model

    if skey is not None and decision.score is not None:
        save_route(skey, model=model, score=decision.score, effort=decision.effort, user_turns=turns)

    # Structured, content-free: no key, no prompt text.
    log.info(
        "MisterPilot Auto: auto_routing=enabled score=%s model=%s provider=%s effort=%s mode=%s"
        " verification=%s fallback=%s tool_step=%s side_job=%s session=%s context_tokens~%s route=%r",
        decision.score if decision.score is not None else "invalid",
        model,
        provider.value,
        decision.effort or "-",
        prefs.mode,
        verification,
        fallback,
        tool_step,
        side_job,
        "hit" if session is not None else ("new" if skey else "none"),
        context_tokens,
        route,
    )
    if verdict is not None:
        debug.report_auto(verdict, payload, route=f"{model} via {provider.value} ({route})")

    return ModelAccess(
        client=LLMClient(model, upstream_key, provider, tenant=tenant_of(raw_key)),
        requested_model=requested_model,
        upstream_model=model,
        provider=provider,
        raw_key=raw_key,
        key_type=KEY_TYPE_MISTERPILOT,
        auto_score=decision.score,
        auto_route=route,
        auto_fallback=fallback,
        effort=decision.effort,
        fallback_models=remaining,
        session_key=skey,
        user_turns=turns,
    )


def next_auto_access(access: ModelAccess) -> Optional[ModelAccess]:
    """The next failover candidate after ``access``'s model failed, or ``None``.

    Only before any output reached the client — the caller guarantees that.
    The chat's session moves to the new model, so its tool loop stays there.
    """
    if not access.is_auto or not access.fallback_models:
        return None
    chosen, _ = _first_servable(list(access.fallback_models), None)
    if chosen is None:
        return None
    model, provider, key, remaining = chosen
    log.warning("MisterPilot Auto: failover %s -> %s", access.upstream_model, model)
    if access.session_key is not None and access.auto_score is not None:
        save_route(access.session_key, model=model, score=access.auto_score,
                   effort=access.effort, user_turns=access.user_turns)
    return replace(
        access,
        client=LLMClient(model, key, provider, tenant=tenant_of(access.raw_key)),
        upstream_model=model,
        provider=provider,
        auto_route=f"{access.auto_route}; failover {access.upstream_model} -> {model}",
        auto_fallback=True,
        fallback_models=remaining,
    )
