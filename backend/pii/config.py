"""
Configuration for the PII redaction pipeline.

Reads from .env — see .env.example for available variables.
PROXY_HMAC_KEY is auto-generated and persisted to .env on first run.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Literal


def _load_env(path: str | None = None) -> None:
    """Load variables from .env into the environment."""
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


def _persist_key(key: str, env_path: str | None = None) -> None:
    path = env_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".env"
    )
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\nPROXY_HMAC_KEY={key}\n")
    except OSError:
        pass


@dataclass
class PseudonymConfig:
    """
    Central configuration for the PII redaction pipeline.

    suffix_length controls HMAC truncation length (hex chars).
    Default 4 → 65536 slots. Use 6 for high-cardinality production deployments.
    """

    hmac_key: bytes
    suffix_length: int = 4
    restore_in_response: bool = True
    max_input_bytes: int = 4 * 1024 * 1024
    store_backend: Literal["memory", "file", "redis", "database"] = "memory"
    store_path: str = ".pii_map.json"
    redis_url: str | None = None
    db_url: str | None = None
    log_context_chars: int = 40

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "PseudonymConfig":
        """Build config from environment variables, with .env fallback."""
        _load_env(env_path)

        raw_key = os.environ.get("PROXY_HMAC_KEY", "")
        if not raw_key:
            raw_key = secrets.token_hex(32)
            _persist_key(raw_key, env_path)

        return cls(
            hmac_key=raw_key.encode("utf-8"),
            store_backend=os.environ.get("PII_STORE_BACKEND", "memory"),  # type: ignore[arg-type]
            store_path=os.environ.get("PII_STORE_PATH", ".pii_map.json"),
            redis_url=os.environ.get("PII_REDIS_URL"),
            db_url=os.environ.get("PII_DB_URL"),
        )
