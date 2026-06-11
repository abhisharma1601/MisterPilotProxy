"""
API key resolution service.

The extension may send one of two kinds of keys in the request:

1. A real DeepSeek API key  — forwarded to DeepSeek unchanged.
2. A MisterPilot key (prefixed ``ms_``, e.g. ``ms_34982349343``) — an internal
   token. It does not work against DeepSeek directly; instead it maps to *our*
   DeepSeek key, which is read from the environment (``.env``).

Routes must never hand a raw header key to the LLM client directly. They resolve
it through :func:`resolve_api_key` first, e.g.::

    client = get_deepseek_client(resolve_api_key(x_api_key))
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import get_config

# MisterPilot-issued keys look like "mp-34982349343".
MISTERPILOT_KEY_PREFIX = "mp-"

# Environment variable holding our own DeepSeek key (used for MisterPilot keys).
DEEPSEEK_ENV_VAR = "DEEPSEEK_API_KEY"

# Key types — what kind of key the client sent. Drives both key resolution and
# cost calculation (MisterPilot keys are billed with a profit margin).
KEY_TYPE_MISTERPILOT = "misterpilot"
KEY_TYPE_DEEPSEEK = "deepseek"


def _load_env(path: str | None = None) -> None:
    """Load variables from .env into the environment (no-op if already set)."""
    env_path = path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".env"
    )
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and key not in os.environ:
                os.environ[key] = value


def is_misterpilot_key(key: Optional[str]) -> bool:
    """True if ``key`` is a MisterPilot-issued token (``mp-…``)."""
    return bool(key) and key.startswith(MISTERPILOT_KEY_PREFIX)


def key_type(key: Optional[str]) -> str:
    """Classify an inbound key as ``KEY_TYPE_MISTERPILOT`` or ``KEY_TYPE_DEEPSEEK``."""
    return KEY_TYPE_MISTERPILOT if is_misterpilot_key(key) else KEY_TYPE_DEEPSEEK


def _deepseek_key_from_env() -> str:
    """Our own DeepSeek key, read from .env (falls back to config)."""
    _load_env()
    return os.environ.get(DEEPSEEK_ENV_VAR, "") or get_config().deepseek.api_key


def resolve_api_key(key: Optional[str]) -> str:
    """Resolve an inbound header key to an actual DeepSeek key.

    - MisterPilot key (``mp-…``) → our DeepSeek key from the environment.
    - Anything else → used as-is (assumed to already be a real DeepSeek key).
    """
    if is_misterpilot_key(key):
        return _deepseek_key_from_env()
    return key or ""
