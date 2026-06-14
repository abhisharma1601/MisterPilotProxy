import logging
import os
from typing import Dict, Optional

import httpx

from .key_service import _load_env
from ..models import DEFAULT_MODEL

log = logging.getLogger(__name__)


def _usage_url() -> str:
    _load_env()
    url = os.environ.get("USAGE_CHARGE_URL")
    if not url:
        raise RuntimeError("USAGE_CHARGE_URL is not set in .env")
    return url
    


class CostService:
    """Delegates cost calculation to the external usage API.

    Two endpoints are used, transparently chosen by key type:

    * ``/api/usage/charge``  — MisterPilot keys (wallet + balance).
    * ``/api/usage/calculate`` — user-supplied DeepSeek keys (cost-only).
    """

    async def calc_cost(
        self,
        *,
        model: Optional[str],
        output: int,
        cache_hit: int,
        cache_miss: int,
        api_key: str,
    ) -> Dict:
        """Call the usage API and return the full response dict.

        The external API is expected to return at least::

            {"costUsd": float, "costInr": float, ...}

        Any extra fields (balanceAfter, breakdown, transactionId, …) are
        forwarded unchanged to the extension so the UI can expose them.
        """
        url = _usage_url()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={
                    "apiKey": api_key,
                    "outputTokens": output,
                    "cacheHitTokens": cache_hit,
                    "cacheMissTokens": cache_miss,
                    "model": model or DEFAULT_MODEL,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()


_service: Optional[CostService] = None


def get_cost_service() -> CostService:
    global _service
    if _service is None:
        _service = CostService()
    return _service
