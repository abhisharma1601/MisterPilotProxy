from typing import Dict, Optional

# Models we expose to clients and can price.
AVAILABLE_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]

DEFAULT_MODEL = "deepseek-v4-pro"

# Per-token prices in USD, derived from DeepSeek billing data.
PRICING: Dict[str, Dict[str, float]] = {
    "deepseek-v4-pro": {
        "output":     0.00000087,
        "cache_hit":  0.000000003625,
        "cache_miss": 0.000000435,
    },
    "deepseek-v4-flash": {
        "output":     0.00000028,
        "cache_hit":  0.0000000028,
        "cache_miss": 0.00000014,
    },
}


class CostService:
    """Turns token usage into USD using per-model pricing."""

    # Percentage markup applied on top of the raw provider cost.
    # e.g. profit_margin = 30 means a raw cost of 100 is billed as 130.
    profit_margin: float = 30.0

    def __init__(
        self,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        default_model: str = DEFAULT_MODEL,
        profit_margin: Optional[float] = None,
    ) -> None:
        self._pricing = pricing if pricing is not None else PRICING
        self._default_model = default_model
        if profit_margin is not None:
            self.profit_margin = profit_margin

    def calc_cost(
        self,
        model: Optional[str],
        output: int,
        cache_hit: int,
        cache_miss: int,
    ) -> float:
        p = self._pricing.get(model or "", self._pricing[self._default_model])
        raw_cost = (
            output * p["output"]
            + cache_hit * p["cache_hit"]
            + cache_miss * p["cache_miss"]
        )
        return raw_cost * (1 + self.profit_margin / 100)


_service: Optional[CostService] = None


def get_cost_service() -> CostService:
    global _service
    if _service is None:
        _service = CostService()
    return _service
