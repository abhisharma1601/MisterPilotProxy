import asyncio
import logging
import os
from typing import Dict, Optional

import httpx

from .key_service import is_misterpilot_key, _load_env

log = logging.getLogger(__name__)

INR_RATE = 96.0
MARGIN = 1.30

_PRICING = {
    "deepseek-v4-pro": {
        "output":    0.00000087,
        "cache_hit": 0.000000003625,
        "cache_miss": 0.000000435,
    },
    "deepseek-v4-flash": {
        "output":    0.00000028,
        "cache_hit": 0.0000000028,
        "cache_miss": 0.00000014,
    },
}
_FALLBACK = "deepseek-v4-pro"


def _get_rates(model: Optional[str]) -> Dict[str, float]:
    return _PRICING.get(model or "", _PRICING[_FALLBACK])


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
        cost_inr = final_usd * INR_RATE
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
