"""
Conversation-aware floors for the MisterPilot Auto score.

The scorer judges the latest user request on its own, which under-routes a
short prompt that continues a hard task ("start with the impl" after an OAuth
design scores 2). Scoring the whole context window instead would over-route:
assistant replies, tool output and file bodies are full of words like
"security" and "architecture". So the conversation raises the score only
through three floors, each read from a narrow, trustworthy part of it:

  history       Earlier user turns, scored the same way as the current one,
                minus ``auto.history_decay`` per turn of age. A task carries
                into its follow-ups and fades once the user moves on.
  sticky        The score of the previous Auto reply — from the server-side
                session (services/auto_session.py), else read back from its
                route line ("complexity 13/20") — minus ``auto.max_drop_per_turn``
                per user turn since. Stops a mid-task flip to a cheaper model —
                every switch re-sends the whole context uncached — and never
                drops inside a tool loop (no new user turn). Needs
                a session or ``auto.show_route``; with neither it is absent.
  long_context  A minimum score once the context passes
                ``auto.long_context_tokens``: cheap tiers get unreliable on
                very long context whatever the task.

The final score is the highest of the verified score and the floors. Floors
only raise; nothing here can lower what the scorer and verifier decided.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..config import AutoConfig, get_config
from .auto_router import MAX_SCORE, MIN_SCORE
from .complexity_scorer import _is_followup, _strip_scaffolding, _user_turns, score_request

# Matches the route line model.py prepends to Auto replies:
#   _MisterPilot Auto · `claude-sonnet-5` · complexity 13/20_
_ROUTE_SCORE_RE = re.compile(r"\A\s*_MisterPilot Auto · [^\n]*?\bcomplexity (\d{1,2})/20")

_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class ContextFloors:
    history: Optional[int] = None
    sticky: Optional[int] = None
    long_context: Optional[int] = None
    context_tokens: int = 0

    def apply(self, score: Optional[int]) -> tuple[Optional[int], str]:
        """``(final score, note)``. The note names the floor that raised it, if any.

        An invalid score (``None``) stays invalid unless a floor exists: the
        conversation is then a better guide than the blind fallback model.
        """
        floors = {
            name: value
            for name, value in (("history", self.history), ("sticky", self.sticky),
                                ("long_context", self.long_context))
            if value is not None
        }
        if not floors:
            return score, ""
        name, floor = max(floors.items(), key=lambda kv: kv[1])
        if score is not None and score >= floor:
            return score, ""
        final = max(MIN_SCORE, min(MAX_SCORE, floor))
        return final, f"raised {score if score is not None else 'invalid'}->{final} by {name} floor"


def _history_floor(payload: Mapping[str, Any], cfg: AutoConfig) -> Optional[int]:
    """Best decayed score among the earlier substantive user turns."""
    earlier = _user_turns(payload)[:-1][-cfg.history_turns:] if cfg.history_turns > 0 else []
    best: Optional[int] = None
    for age, raw in enumerate(reversed(earlier), start=1):
        text, _ = _strip_scaffolding(raw)
        if not text or _is_followup(text):
            continue
        score = score_request({"messages": [{"role": "user", "content": raw}]}).complexity_score
        decayed = score - cfg.history_decay * age
        if decayed >= MIN_SCORE and (best is None or decayed > best):
            best = decayed
    return best


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, Mapping))
    return ""


def _sticky_floor(payload: Mapping[str, Any], cfg: AutoConfig) -> Optional[int]:
    """Previous Auto score, less the allowed drop per user turn since it."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    user_turns_since = 0
    for msg in reversed(messages):
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if role == "user":
            user_turns_since += 1
        elif role == "assistant":
            m = _ROUTE_SCORE_RE.match(_text(msg.get("content")))
            if m:
                floor = int(m.group(1)) - cfg.max_drop_per_turn * user_turns_since
                return floor if floor >= MIN_SCORE else None
    return None


def estimate_context_tokens(payload: Mapping[str, Any]) -> int:
    """Rough token count of what is sent upstream: messages plus tool schemas."""
    size = 0
    for key in ("messages", "tools"):
        value = payload.get(key)
        if value:
            size += len(json.dumps(value, ensure_ascii=False, default=str))
    return size // _CHARS_PER_TOKEN


def session_sticky_floor(prev_score: int, user_turns_since: int, cfg: Optional[AutoConfig] = None) -> Optional[int]:
    """The sticky floor from server-side session state instead of the route line."""
    cfg = cfg or get_config().auto
    floor = prev_score - cfg.max_drop_per_turn * max(0, user_turns_since)
    return floor if floor >= MIN_SCORE else None


def context_floors(
    payload: Mapping[str, Any],
    cfg: Optional[AutoConfig] = None,
    *,
    sticky: Optional[int] = None,
    use_route_line: bool = True,
) -> ContextFloors:
    """All three floors for ``payload``. Never raises; a failed floor is absent.

    ``sticky``: the floor from the conversation's session, when the server
    remembers it; otherwise it is read from the route line in the history.
    """
    cfg = cfg or get_config().auto
    try:
        history = _history_floor(payload, cfg)
    except Exception:  # noqa: BLE001 — a floor must not fail the request
        history = None
    if sticky is None and use_route_line:
        try:
            sticky = _sticky_floor(payload, cfg)
        except Exception:  # noqa: BLE001
            sticky = None
    tokens = estimate_context_tokens(payload)
    long_context = cfg.long_context_min_score if tokens >= cfg.long_context_tokens > 0 else None
    return ContextFloors(history=history, sticky=sticky, long_context=long_context, context_tokens=tokens)
