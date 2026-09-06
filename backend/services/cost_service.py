import asyncio
import logging
import os
import time
from typing import Dict, Optional

import httpx

from .key_service import is_misterpilot_key, _load_env

log = logging.getLogger(__name__)

MARGIN = 1.30

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

_PRICING = {
    "deepseek-v4-pro": {
        "output":    0.00000396,
        "cache_hit": 0.000000044,
        "cache_miss": 0.00000132,
    },
    "deepseek-v4-flash": {
        "output":    0.00000132,
        "cache_hit": 0.000000014,
        "cache_miss": 0.00000044,
    },
}
_FALLBACK = "deepseek-v4-pro"


def _get_rates(model: Optional[str]) -> Dict[str, float]:
    return _PRICING.get(model or "", _PRICING[_FALLBACK])


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

    _load_env()
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


def _charge_url() -> str:
    _load_env()
    url = os.environ.get("USAGE_CHARGE_URL")
    if not url:
        raise RuntimeError("USAGE_CHARGE_URL is not set in .env")
    return url


async def _fire_charge(
    api_key: str,
    model: str,
    cost_inr: float,
    output: int,
    cache_hit: int,
    cache_miss: int,
) -> None:
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                _charge_url(),
                json={
                    "apiKey": api_key,
                    "model": model,
                    "costInr": cost_inr,
                    "outputTokens": output,
                    "cacheHitTokens": cache_hit,
                    "cacheMissTokens": cache_miss,
                },
                timeout=10.0,
            )
    except Exception as exc:
        log.warning("Usage charge failed (non-blocking): %s", exc)


class CostService:
    async def calc_cost(
        self,
        *,
        model: Optional[str],
        output: int,
        cache_hit: int,
        cache_miss: int,
        api_key: str,
    ) -> Dict:
        rates = _get_rates(model)
        raw_usd = (
            output     * rates["output"]
            + cache_hit  * rates["cache_hit"]
            + cache_miss * rates["cache_miss"]
        )
        final_usd = raw_usd * MARGIN if is_misterpilot_key(api_key) else raw_usd
        cost_inr = final_usd * await get_inr_rate()
        resolved_model = model or _FALLBACK

        if is_misterpilot_key(api_key):
            asyncio.create_task(_fire_charge(
                api_key=api_key,
                model=resolved_model,
                cost_inr=cost_inr,
                output=output,
                cache_hit=cache_hit,
                cache_miss=cache_miss,
            ))

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
