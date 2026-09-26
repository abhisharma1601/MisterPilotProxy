"""
API key resolution service.

The extension may send one of two kinds of keys in the request:

1. A provider API key (BYOK) — a real DeepSeek / OpenAI / Anthropic key,
   forwarded to that provider unchanged. Not billed by us.
2. A MisterPilot key (``mp-…``, e.g. ``mp-34982349343``) — an internal token.
   It does not work against any provider directly; after verification it is
   swapped for *our* key for the provider being called, fetched from AWS
   Secrets Manager as ``{provider}_{dev|prod}`` (``deepseek_prod``,
   ``openai_prod``, ``claude_prod`` …). Billed with a margin.

Routes must never hand a raw header key to an LLM client directly. They resolve
it through :func:`resolve_api_key` first — normally via
:func:`services.model_access.open_model_access`, which also applies the
per-model key policy (e.g. ``misterpilot-auto`` refuses BYOK).
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
KEY_TYPE_BYOK = "byok"          # the user's own provider key, any provider

# The provider whose key is used when a caller doesn't say.
DEFAULT_PROVIDER = "deepseek"

# AWS Secrets Manager config
_AWS_SECRET_KEY_FIELD = "key"
_SECRET_TTL_SECONDS = 300  # refresh cached secret every 5 minutes

# module-level cache: secret_name -> (api_key, expires_at_monotonic)
_secret_cache: dict[str, tuple[str, float]] = {}


def is_misterpilot_key(key: Optional[str]) -> bool:
    """True if ``key`` is a MisterPilot-issued token (``mp-…``)."""
    return bool(key) and key.startswith(MISTERPILOT_KEY_PREFIX)


def key_type(key: Optional[str]) -> str:
    """Classify an inbound key as ``KEY_TYPE_MISTERPILOT`` or ``KEY_TYPE_BYOK``."""
    return KEY_TYPE_MISTERPILOT if is_misterpilot_key(key) else KEY_TYPE_BYOK


def _aws_secret_name(provider: str = DEFAULT_PROVIDER) -> str:
    """Secrets Manager secret holding our key for ``provider``, by APP_ENV.

    ``deepseek_prod`` / ``deepseek_dev`` exactly as before; other providers
    follow the same pattern (``openai_prod``, ``claude_dev`` …).
    """
    env = os.environ.get("APP_ENV", "dev").lower()
    name = f"{provider}_{'prod' if env == 'prod' else 'dev'}"
    logger.debug("Resolved AWS secret name: %s (APP_ENV=%s)", name, env)
    return name


def _fetch_key_from_aws(secret_name: str) -> str:
    """Fetch a provider key from AWS Secrets Manager with a TTL-based cache."""
    now = time.monotonic()
    cached = _secret_cache.get(secret_name)
    if cached and now < cached[1]:
        logger.debug("Using cached provider key from secret %s", secret_name)
        return cached[0]

    logger.info("Fetching provider key from AWS Secrets Manager: %s", secret_name)
    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response["SecretString"])
        api_key: str = secret[_AWS_SECRET_KEY_FIELD]
    except (ClientError, KeyError, json.JSONDecodeError) as exc:
        logger.exception("Failed to fetch provider key from AWS Secrets Manager (%s)", secret_name)
        raise RuntimeError(
            f"Failed to fetch provider key from AWS Secrets Manager ({secret_name!r}): {exc}"
        ) from exc

    _secret_cache[secret_name] = (api_key, now + _SECRET_TTL_SECONDS)
    logger.info("Successfully fetched and cached provider key from %s", secret_name)
    return api_key


def get_server_key(provider: str = DEFAULT_PROVIDER) -> str:
    """Our own key for ``provider`` — what a MisterPilot key is swapped for."""
    return _fetch_key_from_aws(_aws_secret_name(provider))

def _verify_url() -> str:
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


async def resolve_api_key(key: Optional[str], provider: str = DEFAULT_PROVIDER) -> str:
    """Resolve an inbound header key to a real key for ``provider``.

    - MisterPilot key (``mp-…``) → verified, then swapped for our key for
      ``provider`` from AWS Secrets Manager.
    - Anything else → used as-is (BYOK: assumed to be a real key for
      ``provider``; the provider rejects it if it isn't).

    This only resolves keys. Which key types a *model* accepts is policy, and
    lives in :mod:`services.model_access`.
    """
    if is_misterpilot_key(key):
        if not await verify_misterpilot_key(key):
            raise HTTPException(status_code=401, detail="Invalid or Low Balance in MisterPilot API key")
        try:
            return get_server_key(provider)
        except RuntimeError:
            # Missing secret for this provider, or AWS unreachable. Logged in
            # _fetch_key_from_aws; the client gets no infrastructure detail.
            raise HTTPException(
                status_code=503,
                detail=f"MisterPilot keys are currently unavailable for {provider} models",
            )
    return key or ""
