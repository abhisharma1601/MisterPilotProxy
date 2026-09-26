import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Coroutine, Dict, Optional

import httpx

from .key_service import is_misterpilot_key

log = logging.getLogger(__name__)

MARGIN = 1.20

# Fallback used only until a live rate is fetched (or when every provider fails).
_FALLBACK_INR_RATE = 96.0

# Free, keyless USD -> INR exchange-rate providers (tried in order).
_RATE_PROVIDERS = (
    "https://api.frankfurter.app/latest?from=USD&to=INR",
    "https://open.er-api.com/v6/latest/USD",
)

_RATE_TTL_SECONDS = 6 * 60 * 60   # refresh cached rate every 6 hours
_RATE_RETRY_SECONDS = 10 * 60     # retry sooner if the last fetch failed

_cached_inr_rate: float = _FALLBACK_INR_RATE
_rate_fetched_at: float = 0.0
_rate_next_attempt: float = 0.0

# USD per token. cache_hit / cache_miss are input tokens served from / not
# from the provider's prompt cache. Every model a MisterPilot key can be billed
# for must be here — see has_pricing().
#
# Claude: through the native Messages API (claude.native) input is split into
# cache reads (cache_hit), cache writes (cache_write — Anthropic charges 1.25x
# input for a 5-minute entry) and uncached tokens (cache_miss). Through the
# OpenAI-compatible fallback there is no caching and everything is cache_miss.
# Models without a cache_write rate have no write premium (OpenAI, DeepSeek
# cache automatically at the normal input price).
_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-5.5":         {"output": 0.00003000, "cache_hit": 0.00000050,  "cache_miss": 0.00000500},
    "gpt-5.4":         {"output": 0.00001500, "cache_hit": 0.00000025,  "cache_miss": 0.00000250},
    "gpt-5.4-mini":    {"output": 0.00000450, "cache_hit": 0.000000075, "cache_miss": 0.00000075},
    "gpt-5.4-nano":    {"output": 0.00000100, "cache_hit": 0.00000002,  "cache_miss": 0.00000020},
    "gpt-5.3-codex":   {"output": 0.00002800, "cache_hit": 0.00000035,  "cache_miss": 0.00000350},
    # Anthropic
    "claude-opus-5-5":  {"output": 0.00002000, "cache_hit": 0.00000020, "cache_miss": 0.00000400, "cache_write": 0.00000500},
    "claude-sonnet-5":  {"output": 0.00001000, "cache_hit": 0.00000020, "cache_miss": 0.00000200, "cache_write": 0.00000250},
    "claude-haiku-4-5": {"output": 0.00000500, "cache_hit": 0.00000010, "cache_miss": 0.00000100, "cache_write": 0.00000125},
    # DeepSeek
    "deepseek-v4-pro": {"output": 0.00000396, "cache_hit": 0.000000044, "cache_miss": 0.00000132},
    "deepseek-flash":  {"output": 0.00000120, "cache_hit": 0.000000006, "cache_miss": 0.00000030},
}
# Defensive only: every usable model (llm.MODELS) is priced above and anything
# else is refused before it is called, so no real request reaches this. It
# keeps calc_cost from crashing if a model is ever added to MODELS without a
# price — and model_access refuses to *bill* a MisterPilot key for such a model.
_FALLBACK = "deepseek-v4-pro"


def _get_rates(model: Optional[str]) -> Dict[str, float]:
    return _PRICING.get(model or "", _PRICING[_FALLBACK])


def price_usd(
    model: Optional[str], *, output: int, cache_hit: int, cache_miss: int, cache_write: int = 0
) -> float:
    """Raw price of the tokens at ``model``'s rates — no margin, no charge.

    For a provider model this is what the provider bills us.
    """
    rates = _get_rates(model)
    return (
        output        * rates["output"]
        + cache_hit   * rates["cache_hit"]
        + cache_miss  * rates["cache_miss"]
        + cache_write * rates.get("cache_write", rates["cache_miss"])
    )


# Share of a warm conversation's input that the provider's cache serves.
_WARM_CACHE_SHARE = 0.9


def estimate_turn_usd(model: str, *, context_tokens: int, output_tokens: int, warm: bool) -> float:
    """Rough raw cost of one turn on ``model`` — for routing decisions, never billing.

    ``warm``: the provider already caches this conversation's prefix (the
    same model served the previous turn). A cold turn pays full input, plus
    the write premium where the provider charges one.
    """
    if warm:
        hit = int(context_tokens * _WARM_CACHE_SHARE)
        return price_usd(model, output=output_tokens, cache_hit=hit, cache_miss=context_tokens - hit)
    if "cache_write" in _get_rates(model):
        return price_usd(model, output=output_tokens, cache_hit=0, cache_miss=0, cache_write=context_tokens)
    return price_usd(model, output=output_tokens, cache_hit=0, cache_miss=context_tokens)


def has_pricing(model: Optional[str]) -> bool:
    """True if ``model`` has its own rates (not just the fallback).

    Required before a MisterPilot key is billed for any model: charging at a
    fallback rate would be silently wrong in either direction.
    """
    return (model or "") in _PRICING


def _extract_inr_rate(data) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    rates = data.get("rates")
    if not isinstance(rates, dict):
        return None
    value = rates.get("INR")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


async def _fetch_inr_rate() -> Optional[float]:
    async with httpx.AsyncClient() as client:
        for url in _RATE_PROVIDERS:
            try:
                resp = await client.get(url, timeout=5.0)
                resp.raise_for_status()
                rate = _extract_inr_rate(resp.json())
                if rate:
                    return rate
            except Exception as exc:
                log.warning("Exchange-rate fetch failed (%s): %s", url, exc)
    return None


async def get_inr_rate() -> float:
    """Return the live USD->INR rate, cached and with a hardcoded fallback."""
    global _cached_inr_rate, _rate_fetched_at, _rate_next_attempt

    override = os.environ.get("USD_INR_RATE")
    if override:
        try:
            return float(override)
        except ValueError:
            log.warning("Ignoring invalid USD_INR_RATE=%r", override)

    now = time.monotonic()
    if now < _rate_next_attempt or now - _rate_fetched_at < _RATE_TTL_SECONDS:
        return _cached_inr_rate

    rate = await _fetch_inr_rate()
    if rate:
        _cached_inr_rate = rate
        _rate_fetched_at = now
        _rate_next_attempt = now + _RATE_TTL_SECONDS
    else:
        # Keep the fallback rate but back off to avoid hammering the API.
        _rate_next_attempt = now + _RATE_RETRY_SECONDS

    return _cached_inr_rate


def cached_inr_rate() -> float:
    """The last known USD->INR rate, without fetching — for display only."""
    override = os.environ.get("USD_INR_RATE")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return _cached_inr_rate


# ── background tasks ──────────────────────────────────────────────────
#
# asyncio keeps only a weak reference to a task, so a fire-and-forget task
# that nothing holds can be garbage-collected before it runs. Every billing
# task goes through spawn(), which holds it until it finishes, and drain()
# gives in-flight charges a chance to complete at shutdown.

_background: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Run ``coro`` in the background, strongly referenced until done.

    Safe to call from a task that is being cancelled (e.g. a stream whose
    client disconnected): creating a task does not await anything.
    """
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task


async def drain(timeout: float = 20.0) -> None:
    """At shutdown: wait for in-flight billing, then cancel what's left.

    A charge cancelled here is written to the dead-letter file by
    :func:`_fire_charge`, so shutdown loses nothing.
    """
    pending = [t for t in _background if not t.done()]
    if not pending:
        return
    _, unfinished = await asyncio.wait(pending, timeout=timeout)
    for task in unfinished:
        task.cancel()
    if unfinished:
        await asyncio.gather(*unfinished, return_exceptions=True)
        log.error("Shutdown: %d billing task(s) unfinished after %.0fs", len(unfinished), timeout)


# ── charge delivery ───────────────────────────────────────────────────
#
# The wallet endpoint has no idempotency key, so a retry is only safe when
# the request provably never reached it (connection refused / connect
# timeout). If it was sent but no reliable answer came back (read timeout,
# dropped connection), the wallet may already have applied it — retrying
# could charge the user twice. Those, and outright refusals (non-2xx), are
# written to a dead-letter file for reconciliation instead of being lost.

_CHARGE_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
_CHARGE_RETRY_DELAYS = (0.5, 2.0, 5.0)     # up to 4 attempts in total


def _charge_url() -> str:
    url = os.environ.get("USAGE_CHARGE_URL")
    if not url:
        raise RuntimeError("USAGE_CHARGE_URL is not set in .env")
    return url


def _dead_letter_path() -> Path:
    configured = os.environ.get("CHARGE_DEADLETTER_PATH", "").strip()
    return Path(configured) if configured else Path(__file__).resolve().parent.parent / "failed_charges.jsonl"


def _dead_letter(charge: Dict[str, Any], reason: str) -> None:
    """Persist an undelivered charge so it can be reconciled, never dropped.

    The file holds the full record, including the MisterPilot key the wallet
    needs to identify the account — it is git-ignored and must be treated as
    a secret. The log line masks the key.
    """
    path = _dead_letter_path()
    entry = {"failedAt": datetime.now(timezone.utc).isoformat(), "reason": reason, **charge}
    key_tail = str(charge.get("apiKey", ""))[-4:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        log.error(
            "Usage charge NOT delivered (%s): model=%s costInr=%.6f key=…%s — saved to %s",
            reason, charge.get("model"), charge.get("costInr", 0.0), key_tail, path,
        )
    except OSError:
        # Last resort: everything except the key, so the amount is not lost.
        log.critical(
            "Usage charge NOT delivered (%s) AND could not be saved to %s: %s",
            reason, path, json.dumps({**entry, "apiKey": f"…{key_tail}"}),
        )


async def _fire_charge(charge: Dict[str, Any]) -> None:
    """Deliver one charge to the wallet: retry only where it can't double-charge."""
    try:
        try:
            url = _charge_url()
        except RuntimeError as exc:
            _dead_letter(charge, str(exc))
            return

        async with httpx.AsyncClient(timeout=_CHARGE_TIMEOUT) as client:
            for attempt in range(len(_CHARGE_RETRY_DELAYS) + 1):
                try:
                    resp = await client.post(url, json=charge)
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    # Never reached the wallet: retrying cannot double-charge.
                    if attempt < len(_CHARGE_RETRY_DELAYS):
                        await asyncio.sleep(_CHARGE_RETRY_DELAYS[attempt])
                        continue
                    _dead_letter(charge, f"wallet unreachable after {attempt + 1} attempts ({type(exc).__name__})")
                    return
                except httpx.HTTPError as exc:
                    # Sent, outcome unknown — the wallet may have applied it.
                    _dead_letter(charge, f"outcome unknown, not retried ({type(exc).__name__})")
                    return

                if resp.is_success:
                    return
                _dead_letter(charge, f"wallet refused: HTTP {resp.status_code}")
                return
    except asyncio.CancelledError:
        _dead_letter(charge, "cancelled at shutdown; may or may not have been applied")
        raise


class CostService:
    async def calc_cost(
        self,
        *,
        model: Optional[str],
        output: int,
        cache_hit: int,
        cache_miss: int,
        api_key: str,
        cache_write: int = 0,
    ) -> Dict:
        """Price a request; for a MisterPilot key, also charge the wallet.

        The charge is delivered in the background (see :func:`_fire_charge`)
        so it never delays or fails the chat response. BYOK requests are only
        priced, for the log — nothing is charged, so no exchange rate is
        needed and ``costInr`` is ``None``.

        ``model`` is the model actually called — for Auto, the model the
        router picked, never ``misterpilot-auto``.
        """
        raw_usd = price_usd(model, output=output, cache_hit=cache_hit, cache_miss=cache_miss,
                            cache_write=cache_write)
        billed = is_misterpilot_key(api_key)
        final_usd = raw_usd * MARGIN if billed else raw_usd
        resolved_model = model or _FALLBACK
        cost_inr: Optional[float] = None

        if billed:
            cost_inr = final_usd * await get_inr_rate()
            spawn(_fire_charge({
                "apiKey": api_key,
                "model": resolved_model,
                "costInr": cost_inr,
                "outputTokens": output,
                "cacheHitTokens": cache_hit,
                # Cache writes are uncached input (priced at the write rate in
                # costInr); the wallet record keeps its existing shape.
                "cacheMissTokens": cache_miss + cache_write,
            }))

        return {
            "costUsd": final_usd,
            "costInr": cost_inr,
            "model": resolved_model,
        }


_service: Optional[CostService] = None


def get_cost_service() -> CostService:
    global _service
    if _service is None:
        _service = CostService()
    return _service
