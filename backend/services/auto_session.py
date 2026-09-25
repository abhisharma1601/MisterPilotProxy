"""
Per-conversation memory for MisterPilot Auto.

The OpenAI protocol is stateless: every request re-sends the whole chat, and
nothing says which model served the last turn. Auto needs that to route well:

  - **pinning** — an agent's tool loop stays on the model that started the
    turn. Re-scoring every tool step cost a verifier call and risked a
    mid-task model switch, and a switch re-sends the whole context uncached;
  - **cache-aware downgrades** — whether stepping down to a cheaper model
    pays for the cache it throws away (see ``model_access``);
  - **sticky score and effort** — without relying on the route line echoed
    back in the chat (which ``auto.show_route: false`` removes);
  - **spend** — per key per day, for the optional daily budget.

A conversation is identified by the customer's key and its first user
message, which the client re-sends unchanged on every turn. Everything lives
in process memory with a TTL: a restart or another instance only loses the
optimisations for one turn, never correctness. Keys are stored as hashes.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generic, Mapping, Optional, Tuple, TypeVar

from ..config import get_config

T = TypeVar("T")


class TTLStore(Generic[T]):
    """A small thread-safe LRU map whose entries expire."""

    def __init__(self, max_items: int, ttl_seconds: float) -> None:
        self._items: "OrderedDict[str, Tuple[float, T]]" = OrderedDict()
        self._max = max_items
        self.ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: Optional[str]) -> Optional[T]:
        if key is None:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored, value = item
            if time.monotonic() - stored > self.ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: Optional[str], value: T) -> None:
        if key is None:
            return
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def tenant_of(raw_key: str) -> str:
    """An opaque, stable id for the customer behind ``raw_key``."""
    return hashlib.sha256(f"mp-tenant:{raw_key}".encode()).hexdigest()[:32]


# ── conversation state ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AutoSession:
    model: str                  # the model that served the last turn
    score: int                  # the complexity it was routed on
    effort: Optional[str]       # the reasoning effort it ran at
    user_turns: int             # user messages in the chat at that turn


_sessions: TTLStore[AutoSession] = TTLStore(50_000, get_config().auto.session_ttl_seconds)


def _messages(payload: Mapping[str, Any]) -> list:
    messages = payload.get("messages")
    return messages if isinstance(messages, list) else []


def user_turns(payload: Mapping[str, Any]) -> int:
    return sum(1 for m in _messages(payload) if isinstance(m, Mapping) and m.get("role") == "user")


def is_tool_step(payload: Mapping[str, Any]) -> bool:
    """Is this request a step inside an agent's tool loop (not a new user turn)?"""
    messages = _messages(payload)
    last = messages[-1] if messages else None
    return isinstance(last, Mapping) and last.get("role") == "tool"


def session_key(raw_key: str, payload: Mapping[str, Any]) -> Optional[str]:
    """The conversation's id: the customer plus its first user message."""
    for msg in _messages(payload):
        if isinstance(msg, Mapping) and msg.get("role") == "user":
            content = msg.get("content")
            text = content if isinstance(content, str) else repr(content)
            return hashlib.sha256(f"{tenant_of(raw_key)}\x00{text}".encode("utf-8", "surrogatepass")).hexdigest()
    return None


def get_session(key: Optional[str]) -> Optional[AutoSession]:
    return _sessions.get(key)


def save_route(
    key: Optional[str], *, model: str, score: int, effort: Optional[str], user_turns: int
) -> None:
    """Record the route this turn took."""
    if key is None:
        return
    _sessions.put(key, AutoSession(model=model, score=score, effort=effort, user_turns=user_turns))


# ── daily spend per key ───────────────────────────────────────────────

_spend: Dict[str, Tuple[str, float]] = {}
_spend_lock = threading.Lock()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_spend(raw_key: str, usd: float) -> None:
    """Add a billed Auto request to the key's spend today (UTC)."""
    tenant, day = tenant_of(raw_key), _today()
    with _spend_lock:
        seen_day, total = _spend.get(tenant, (day, 0.0))
        _spend[tenant] = (day, (total if seen_day == day else 0.0) + usd)


def spent_today_usd(raw_key: str) -> float:
    """This key's Auto spend today (UTC) on this server instance."""
    with _spend_lock:
        day, total = _spend.get(tenant_of(raw_key), ("", 0.0))
    return total if day == _today() else 0.0
