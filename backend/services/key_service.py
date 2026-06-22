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

import json
import os
import time
from typing import Optional

import boto3
import requests
from botocore.exceptions import ClientError
from fastapi import HTTPException

# MisterPilot-issued keys look like "mp-34982349343".
MISTERPILOT_KEY_PREFIX = "mp"

# Key types — what kind of key the client sent. Drives both key resolution and
# cost calculation (MisterPilot keys are billed with a profit margin).
KEY_TYPE_MISTERPILOT = "misterpilot"
KEY_TYPE_DEEPSEEK = "deepseek"

# AWS Secrets Manager config
_AWS_SECRET_KEY_FIELD = "key"
_SECRET_TTL_SECONDS = 300  # refresh cached secret every 5 minutes

# module-level cache: secret_name -> (api_key, expires_at_monotonic)
_secret_cache: dict[str, tuple[str, float]] = {}


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


def _aws_secret_name() -> str:
    """Return the Secrets Manager secret name based on APP_ENV (dev or prod)."""
    _load_env()
    env = os.environ.get("APP_ENV", "dev").lower()
    return "deepseek_prod" if env == "prod" else "deepseek_dev"


def _fetch_deepseek_key_from_aws(secret_name: str) -> str:
    """Fetch the DeepSeek key from AWS Secrets Manager with a TTL-based cache."""
    now = time.monotonic()
    cached = _secret_cache.get(secret_name)
    if cached and now < cached[1]:
        return cached[0]

    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response["SecretString"])
        api_key: str = secret[_AWS_SECRET_KEY_FIELD]
    except (ClientError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to fetch DeepSeek key from AWS Secrets Manager ({secret_name!r}): {exc}"
        ) from exc

    _secret_cache[secret_name] = (api_key, now + _SECRET_TTL_SECONDS)
    return api_key


def _get_deepseek_key() -> str:
    """Fetch our DeepSeek key from AWS Secrets Manager."""
    return _fetch_deepseek_key_from_aws(_aws_secret_name())

def _verify_url() -> str:
    _load_env()
    url = os.environ.get("MISTERPILOT_VERIFY_URL")
    if not url:
        raise RuntimeError("MISTERPILOT_VERIFY_URL is not set in .env")
    return url

def verify_misterpilot_key(key: Optional[str]) -> bool:
    try:
        response = requests.post(_verify_url(), json={"apiKey": key}, timeout=5)
        response.raise_for_status()
        return bool(response.json().get("valid", False))
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Key verification service is unreachable")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=503, detail="Key verification service timed out")
    except requests.exceptions.HTTPError:
        return False
    except Exception:
        raise HTTPException(status_code=503, detail="Key verification service error")


def resolve_api_key(key: Optional[str]) -> str:
    """Resolve an inbound header key to an actual DeepSeek key.

    - MisterPilot key (``mp-…``) → our DeepSeek key from the environment.
    - Anything else → used as-is (assumed to already be a real DeepSeek key).
    """
    if is_misterpilot_key(key):
        if(not verify_misterpilot_key(key)):
            raise HTTPException(status_code=401, detail="Invalid or Low Balance in MisterPilot API key")
        return _get_deepseek_key()
    return key or ""
