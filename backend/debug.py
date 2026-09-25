"""
Console verdict printer, for calibrating the complexity scorer.

Off by default. Set ``DEBUG_ROUTE_REQUEST=1`` to print one block per request:
the prompt as the scorer extracted it, the computed score and its math, the AI
check, and — for ``misterpilot-auto`` — the model the router picked.

It costs nothing extra where it can:

- **misterpilot-auto** is already scored on the request path to pick a model;
  :mod:`services.model_access` hands that verdict to :func:`report_auto`.
  Nothing is scored twice.
- **Explicit models** are not scored at all in normal operation, so when this
  is on they are scored in the background (:func:`score_in_background`), off
  the request path. The AI check runs on OUR DeepSeek key — never the user's
  own key, which must not pay for our diagnostics.

Printing never raises: a broken printout must not break a chat.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import sys
from typing import Any, Mapping, Optional

from .services.complexity_scorer import score_request
from .services.complexity_verifier import VerifiedScore, normalize_untrusted, verify_score
from .services.key_service import get_server_key

log = logging.getLogger("router.debug")

_seq = itertools.count(1)

# Strong references to in-flight background tasks; the event loop only keeps
# weak ones, so an unreferenced task can be garbage-collected mid-flight.
_tasks: set[asyncio.Task] = set()


def enabled() -> bool:
    return os.environ.get("DEBUG_ROUTE_REQUEST", "0").strip().lower() in ("1", "true", "yes")


# ── formatting ────────────────────────────────────────────────────────

def _console_safe(text: str) -> str:
    """Printable on this console, with user text defanged.

    The block echoes user-written text: control characters are stripped so a
    raw ESC sequence can't recolour, hide or overwrite the operator's
    terminal. Then re-encoded for consoles that can't print everything
    (Windows cp1252 etc.).
    """
    text = normalize_untrusted(text)
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _payload_shape(payload: Mapping[str, Any]) -> str:
    """Payload size, for context only — the scorer deliberately ignores it."""
    messages = payload.get("messages") or []
    chars = sum(len(m.get("content")) for m in messages
                if isinstance(m, Mapping) and isinstance(m.get("content"), str))
    return f"{len(messages)} msgs | {chars:,} ctx chars | {len(payload.get('tools') or [])} tools"


def _ai_line(v: VerifiedScore) -> str:
    """One line on what the AI verifier concluded."""
    if v.status == "blocked":
        who = "reported by AI" if v.injection == ("reported_by_ai",) else f"rules: {', '.join(v.injection)}"
        return f"BLOCKED - prompt injection ({who}) -> computed score is final  {v.latency_ms}ms"
    if v.ai is None:
        return f"{v.status} ({v.error or 'no reply'}) -> computed score is final  {v.latency_ms}ms"
    ai = v.ai
    delta = ai.complexity_score - v.computed.complexity_score
    if v.status == "clamped":
        return (
            f"CLAMPED  ai={ai.complexity_score} ({delta:+d} vs computed) is outside the guard band"
            f" -> held to {v.complexity_score}  ai_conf={ai.confidence}  {v.latency_ms}ms"
        )
    return (
        f"{v.status}  ai={ai.complexity_score} ({delta:+d} vs computed)"
        f"  ai_conf={ai.confidence}  {v.latency_ms}ms  -> AI score is final"
    )


def print_verdict(
    verdict: VerifiedScore,
    payload: Mapping[str, Any],
    *,
    requested_model: str,
    route: Optional[str] = None,
) -> None:
    """Print one verdict block. Never raises."""
    try:
        s = verdict.signals
        b = s.get("breakdown", {})
        c = verdict.computed

        evidence = []
        for label, key in (
            ("task", "task_evidence"), ("scope", "scope_evidence"),
            ("arch", "architecture_concepts"), ("sec", "security_families"),
            ("deps", "components"), ("depth", "depth_evidence"),
        ):
            values = s.get(key) or []
            if values:
                evidence.append(f"{label}=[{', '.join(map(str, values))}]")

        bar = "=" * 78
        lines = [
            bar,
            f"ROUTER #{next(_seq)}  model={requested_model}",
            f"  prompt   : {s.get('prompt_excerpt') or '(no user prompt found)'}",
            f"  source   : {s.get('prompt_source')}",
            f"  payload  : {_payload_shape(payload)}"
            f" | stripped {s.get('scaffolding_stripped_chars', 0):,} chars of scaffolding",
            f"  verdict  : score={verdict.complexity_score}/20  type={verdict.task_type.value}"
            f"  scope={verdict.scope.value}  depth={verdict.reasoning_depth.value}"
            f"  confidence={verdict.confidence}",
            f"  computed : score={c.complexity_score}  type={c.task_type.value}"
            f"  scope={c.scope.value}  depth={c.reasoning_depth.value}",
            f"  math     : 1 + task {b.get('task', 0)} + security {b.get('security', 0)}"
            f" + deps {b.get('dependency', 0)} + depth {b.get('depth', 0)}"
            f" = {b.get('raw_total', 0)}",
            f"  evidence : {'  '.join(evidence) or '(none)'}",
            f"  ai check : {_ai_line(verdict)}",
        ]
        if verdict.ai is not None:
            if verdict.ai.reason:
                lines.append(f"  ai says  : {verdict.ai.reason}")
            lines.extend(f"  ai issue : {issue}" for issue in verdict.ai.issues)
        if route:
            lines.append(f"  routed   : {route}")
        lines += [bar, ""]
        print(_console_safe("\n".join(lines)), flush=True)
    except Exception:  # noqa: BLE001 — diagnostics must never break a request
        log.exception("verdict printout failed")


# ── entry points ──────────────────────────────────────────────────────

def report_auto(
    verdict: VerifiedScore, payload: Mapping[str, Any], *, route: str
) -> None:
    """Print the verdict misterpilot-auto already computed to route with."""
    if enabled():
        print_verdict(verdict, payload, requested_model="misterpilot-auto", route=route)


async def _score_and_print(payload: Mapping[str, Any], requested_model: str) -> None:
    try:
        computed = score_request(dict(payload))
        try:
            # Blocking boto3 call (cached 5 min) — keep it off the event loop.
            api_key: Optional[str] = await asyncio.to_thread(get_server_key, "deepseek")
        except Exception:  # noqa: BLE001 — no key: the verifier reports "unverified"
            api_key = None
        verified = await verify_score(computed, api_key=api_key)
        print_verdict(verified, payload, requested_model=requested_model)
    except Exception:  # noqa: BLE001
        log.exception("background scoring failed")


def score_in_background(payload: Mapping[str, Any], requested_model: str) -> None:
    """Score an explicit-model request for the printout, off the request path."""
    if not enabled():
        return
    task = asyncio.create_task(_score_and_print(payload, requested_model))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
