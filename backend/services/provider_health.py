"""
Upstream health for MisterPilot Auto failover: a per-model circuit breaker.

When a model keeps failing (rate limits, overload, outages), every Auto
request routed to it would wait out the retries before failing over. The
breaker remembers: after ``auto.breaker_failures`` consecutive failures the
model is skipped for ``auto.breaker_cooldown_seconds``, then one request is
let through to probe it. Any success closes the breaker.

In-process state: each server instance learns on its own, which is fine for
a signal that only decides the order in which models are tried.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict

from ..config import get_config

log = logging.getLogger("provider_health")


@dataclass
class _State:
    failures: int = 0
    open_until: float = 0.0


_states: Dict[str, _State] = {}
_lock = threading.Lock()


def is_available(model: str) -> bool:
    """False while ``model``'s breaker is open."""
    with _lock:
        state = _states.get(model)
        return state is None or time.monotonic() >= state.open_until


def record_success(model: str) -> None:
    with _lock:
        state = _states.get(model)
        if state is not None and (state.failures or state.open_until):
            log.info("breaker closed: %s is serving again", model)
        _states.pop(model, None)


def record_failure(model: str) -> None:
    cfg = get_config().auto
    with _lock:
        state = _states.setdefault(model, _State())
        state.failures += 1
        if state.failures >= max(1, cfg.breaker_failures):
            state.open_until = time.monotonic() + cfg.breaker_cooldown_seconds
            # Half-open after the cooldown: one more failure re-opens it.
            state.failures = max(1, cfg.breaker_failures) - 1
            log.warning("breaker open: skipping %s for %ss", model, cfg.breaker_cooldown_seconds)
