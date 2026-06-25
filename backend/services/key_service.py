"""
API key resolution service.

The extension may send one of two kinds of keys in the request:

1. A real DeepSeek API key  — forwarded to DeepSeek unchanged.
2. A MisterPilot key (prefixed ``ms_``, e.g. ``ms_34982349343``) — an internal
   token. It does not work against DeepSeek directly; instead it maps to *our*
   DeepSeek key, which is read from the environment (``.env``).

Routes must never hand a raw header key to the LLM client directly. They resolve
it through :func:`resolve_api_key` first, e.g.::

    client = get_deepseek_client(await resolve_api_key(x_api_key))
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import boto3
import httpx
from botocore.exceptions import ClientError
from fastapi import HTTPException

logger = logging.getLogger(__name__)

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


def _load_env() -> None:
    """Load variables from backend/.env into the environment (no-op if already set)."""
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if not os.path.exists(env_path):
        logger.debug("No .env file found at %s, skipping", env_path)
        return
    load_dotenv(env_path, override=False)
    logger.info("Loaded environment from %s", env_path)


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
    name = "deepseek_prod" if env == "prod" else "deepseek_dev"
    logger.debug("Resolved AWS secret name: %s (APP_ENV=%s)", name, env)
    return name


def _fetch_deepseek_key_from_aws(secret_name: str) -> str:
    """Fetch the DeepSeek key from AWS Secrets Manager with a TTL-based cache."""
    now = time.monotonic()
    cached = _secret_cache.get(secret_name)
    if cached and now < cached[1]:
        logger.debug("Using cached DeepSeek key from secret %s", secret_name)
        return cached[0]

    logger.info("Fetching DeepSeek key from AWS Secrets Manager: %s", secret_name)
    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response["SecretString"])
        api_key: str = secret[_AWS_SECRET_KEY_FIELD]
    except (ClientError, KeyError, json.JSONDecodeError) as exc:
        logger.exception("Failed to fetch DeepSeek key from AWS Secrets Manager (%s)", secret_name)
        raise RuntimeError(
            f"Failed to fetch DeepSeek key from AWS Secrets Manager ({secret_name!r}): {exc}"
        ) from exc

    _secret_cache[secret_name] = (api_key, now + _SECRET_TTL_SECONDS)
    logger.info("Successfully fetched and cached DeepSeek key from %s", secret_name)
    return api_key


def _get_deepseek_key() -> str:
    """Fetch our DeepSeek key from AWS Secrets Manager."""
    return _fetch_deepseek_key_from_aws(_aws_secret_name())

def _verify_url() -> str:
    _load_env()
    url = os.environ.get("MISTERPILOT_VERIFY_URL")
    if not url:
        logger.error("MISTERPILOT_VERIFY_URL is not set in .env")
        raise RuntimeError("MISTERPILOT_VERIFY_URL is not set in .env")
    return url

async def verify_misterpilot_key(key: Optional[str]) -> bool:
    try:
        url = _verify_url()
    except RuntimeError as exc:
        logger.error("Server misconfiguration: %s", exc)
        raise HTTPException(status_code=500, detail="Server misconfiguration: key verification URL not configured")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"apiKey": key}, timeout=5.0)
        response.raise_for_status()
        return bool(response.json().get("valid", False))
    except httpx.ConnectError:
        logger.error("Key verification service is unreachable")
        raise HTTPException(status_code=503, detail="Key verification service is unreachable")
    except httpx.TimeoutException:
        logger.error("Key verification service timed out")
        raise HTTPException(status_code=503, detail="Key verification service timed out")
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            logger.error("Key verification service returned %s", e.response.status_code)
            raise HTTPException(status_code=503, detail="Key verification service error")
        logger.warning("Key verification rejected (HTTP %s)", e.response.status_code)
        return False
    except Exception:
        logger.exception("Unexpected error during key verification")
        raise HTTPException(status_code=503, detail="Key verification service error")


async def resolve_api_key(key: Optional[str]) -> str:
    """Resolve an inbound header key to an actual DeepSeek key.

    - MisterPilot key (``mp-…``) → our DeepSeek key from the environment.
    - Anything else → used as-is (assumed to already be a real DeepSeek key).
    """
    if is_misterpilot_key(key):
        if not await verify_misterpilot_key(key):
            raise HTTPException(status_code=401, detail="Invalid or Low Balance in MisterPilot API key")
        return _get_deepseek_key()
    return key or ""
