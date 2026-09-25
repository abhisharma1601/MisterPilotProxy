from pathlib import Path
from typing import Dict, List
from functools import lru_cache

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Transport settings for one OpenAI-compatible provider.

    Keys are not configured here: every request resolves its own (a BYOK key
    passed through, or a MisterPilot key swapped for ours from AWS Secrets
    Manager). Models are not configured here either: each request names one.
    """

    base_url: str
    timeout: int = 60
    max_retries: int = 3
    # Ask for token usage on the final stream chunk. Without it a streamed
    # request produces no usage, so it is never costed or charged. All three
    # providers support it (Anthropic's compat endpoint lists stream_options
    # as fully supported).
    include_usage: bool = True
    # Sent as a system message ahead of every request's messages (empty: off).
    # Fixed text at the very start, so it never disturbs the prompt cache.
    system_instruction: str = ""
    # Claude only: call Anthropic's native Messages API (prompt caching,
    # adaptive thinking, effort) instead of its OpenAI-compatible endpoint,
    # which has no prompt caching. Falls back to the compatible endpoint if
    # the native API rejects a request.
    native: bool = False
    # Claude only: base URL for the native Messages API (SDK default if empty).
    native_base_url: str = ""


class TierConfig(BaseModel):
    """One row of the Auto routing table (services/auto_router.py)."""

    low: int
    high: int
    model: str
    # Same-class models to serve this tier when ``model`` is down.
    alternates: List[str] = []
    # Reasoning effort by position within the tier, lowest score first.
    efforts: List[str] = []


class AutoConfig(BaseModel):
    # Fallback model for "misterpilot-auto" when the router can't use a score
    # (invalid or failed scoring) or the selected tier can't be served.
    model: str = "deepseek-v4-pro"
    # Start each Auto reply with a line naming the model it was routed to,
    # e.g. "_MisterPilot Auto · `claude-sonnet-5` · complexity 12/20_".
    show_route: bool = True
    # Conversation floors (services/auto_context.py). Earlier user turns carry
    # their score forward, minus history_decay per turn of age.
    history_turns: int = 5
    history_decay: int = 1
    # The previous Auto reply's score can fall by at most this per user turn.
    max_drop_per_turn: int = 3
    # Past this many context tokens, never score below long_context_min_score.
    long_context_tokens: int = 100_000
    long_context_min_score: int = 7
    # Routing table override; empty = the built-in DEFAULT_TIERS.
    tiers: List[TierConfig] = []

    # Customer modes: "economy" | "balanced" | "quality". Each shifts the
    # score by the given amount before the tier lookup. Chosen per request by
    # model alias (misterpilot-auto-economy / -quality), header or body.
    default_mode: str = "balanced"
    mode_shifts: Dict[str, int] = {"economy": -2, "balanced": 0, "quality": 2}

    # Cache-aware downgrades: stepping down to a cheaper model mid-chat
    # re-sends the whole context uncached. Only switch when it pays off over
    # this many turns, assuming this many output tokens per turn.
    switch_horizon_turns: int = 3
    expected_output_tokens: int = 1500
    # Seconds a conversation's routing state is kept after its last request.
    session_ttl_seconds: int = 3600

    # AI verification is skipped when the computed score is confident and
    # more than verify_boundary_margin points from a tier edge.
    verify_min_confidence: float = 0.75
    verify_boundary_margin: int = 1

    # Daily budget (per key, when the client sends one): past warn_ratio the
    # score shifts down by budget_shift; past the budget, Auto is capped at
    # budget_cap_model. Never refuses a request.
    budget_warn_ratio: float = 0.8
    budget_shift: int = -2
    budget_cap_model: str = "gpt-5.4-mini"


    # Failover health: a model with this many consecutive upstream failures is
    # skipped for breaker_cooldown_seconds.
    breaker_failures: int = 3
    breaker_cooldown_seconds: int = 30


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]


class AppConfig(BaseModel):
    deepseek: ProviderConfig = ProviderConfig(base_url="https://api.deepseek.com/v1")
    openai: ProviderConfig = ProviderConfig(base_url="https://api.openai.com/v1")
    # Anthropic's OpenAI-compatible endpoint (used via the OpenAI SDK).
    claude: ProviderConfig = ProviderConfig(base_url="https://api.anthropic.com/v1/")
    auto: AutoConfig = AutoConfig()
    server: ServerConfig = ServerConfig()


@lru_cache
def get_config() -> AppConfig:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
