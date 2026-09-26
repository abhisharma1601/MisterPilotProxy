"""
MisterPilot complexity scorer.

Given a Copilot Chat / VS Code chat request payload, estimate the *engineering
difficulty* of what the user asked for, on a 1-20 scale.

The one idea this module is built around
---------------------------------------
**Score the task, not the payload.**

Copilot Chat wraps every request in a large harness: tool definitions, a
terminal listing, editor context, workspace summaries, diagnostics, git state,
attached file bodies and the whole chat history. A request to rename a variable
can arrive as 150k characters that mention terminals, databases, security,
architecture and errors — none of which the user asked about.

Any scorer that reads that blob will call every request an expert task. So this
one works in two steps:

  1. ``extract_user_prompt`` digs out the words the user actually typed and
     strips the editor scaffolding around them.
  2. Every detector below reads ONLY that prompt. Payload-level context
     (attachments, diagnostics, terminal, git, workspace, history size) is used
     only when the prompt explicitly points at it — "fix the errors in these
     files" makes the attached file count relevant; "add a prop" does not.

Calibration anchors (from the spec)::

    "Explain this function"                          ->  2
    "Add pagination to this endpoint"                ->  5
    "Implement OAuth login with refresh tokens"      -> 12
    "Design a payment system for 10 million users"   -> 19

Scoring model
-------------
Start at 1, then add independent components::

    task type          explanation +0..2   edit +1..3      bugfix +2..5
                       refactor    +3..6   feature +4..8   security +3..6
                       architecture +5..8  system design +7..10
                       (debugging / optimization: +3..7, see TASK_POINTS)
    security           +3..6   when the task touches security concepts
    dependencies       +0..5   components that must work together, minus one
    reasoning depth    +0..5   low / medium / high, scaled by evidence

The position *within* a task range comes from scope (function … cross-repo),
extra requested actions and explicit scale. The sum is clamped to 1-20.

Pure standard library, Python 3.11+. No I/O, no global state — every helper is
a plain function of its inputs, so each can be tested on its own.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

__all__ = [
    "ComplexityScore",
    "ReasoningDepth",
    "Scope",
    "TaskType",
    "compute_score",
    "detect_architecture_signals",
    "detect_reasoning_depth",
    "detect_scope",
    "detect_security_signals",
    "detect_task_type",
    "estimate_dependency_complexity",
    "extract_user_prompt",
    "score_request",
]


# ════════════════════════════════════════════════════════════════════════
# Types
# ════════════════════════════════════════════════════════════════════════

class TaskType(StrEnum):
    """What kind of engineering work the user is asking for.

    ``UNKNOWN`` is not in the spec's detection list; it is returned only when
    no prompt could be extracted at all, so callers can tell "trivial" apart
    from "we could not see the request".
    """

    EXPLANATION = "explanation"
    EDIT = "edit"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    FEATURE = "feature"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    SYSTEM_DESIGN = "system_design"
    UNKNOWN = "unknown"


class Scope(StrEnum):
    """How much of the codebase the task reaches, narrowest first.

    ``MULTI_MODULE`` sits between module and service: it is what "frontend and
    backend" or "these three files" means, and the spec's example output uses it.
    """

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    MULTI_MODULE = "multi_module"
    SERVICE = "service"
    REPOSITORY = "repository"
    CROSS_REPOSITORY = "cross_repository"


class ReasoningDepth(StrEnum):
    """How hard the thinking is, independent of how much typing it takes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ComplexityScore:
    """The scorer's verdict.

    ``signals`` always carries the three keys from the spec example
    (``architecture``, ``security``, ``dependency_count``) plus the evidence and
    the per-component breakdown, so a surprising score can be traced back to
    the exact words that produced it.
    """

    complexity_score: int
    task_type: TaskType
    reasoning_depth: ReasoningDepth
    scope: Scope
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form matching the spec's output schema."""
        out = asdict(self)
        out["task_type"] = self.task_type.value
        out["reasoning_depth"] = self.reasoning_depth.value
        out["scope"] = self.scope.value
        return out


@dataclass(frozen=True, slots=True)
class ExtractedPrompt:
    """The user's request, separated from the editor's scaffolding.

    ``source`` records where the text came from, which feeds confidence:
    a dedicated ``prompt`` field or ``<userRequest>`` tag is unambiguous; a
    fallback to an earlier turn ("yes, go ahead") is a guess.
    """

    text: str
    source: str
    raw_text: str = ""                    # before scaffolding was stripped
    attachment_paths: tuple[str, ...] = ()
    is_followup: bool = False


@dataclass(frozen=True, slots=True)
class Detection:
    """A detector's answer plus the phrases that produced it."""

    value: Any
    evidence: tuple[str, ...] = ()
    strength: float = 0.0                 # 0..1, how sure the detector is


# ════════════════════════════════════════════════════════════════════════
# 1. Prompt extraction
# ════════════════════════════════════════════════════════════════════════

# Fields that, when present, hold the user's request directly. The Copilot
# chat participant API and several proxies put it here rather than in messages.
_PROMPT_FIELDS = ("prompt", "request", "query", "userPrompt", "user_prompt", "message", "input")

# Tags whose content IS the request. If one is present, everything outside it
# is scaffolding by definition — this is the most reliable signal Copilot gives.
_INTENT_TAGS = (
    "userRequest", "user_request", "userQuery", "user_query",
    "request", "prompt", "query", "question",
)

# Pasted code and attached file bodies are context, not a request. They are
# removed so a snippet that happens to contain "password" or "cache" cannot
# fire the security or architecture detectors.
_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_ATTACHMENT_PATH_RE = re.compile(
    r"<attachment\b[^>]*?\b(?:filePath|path|uri|id)\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
# Any paired tag block, of any name. Copilot's wrappers are XML-ish
# (<context>, <editorContext>, <reminderInstructions>, <attachments>, …) and
# the set changes between releases, so rather than chase a list of names we
# drop every paired block and keep what the user typed between them.
_PAIRED_BLOCK_RE = re.compile(r"<([A-Za-z][\w.:-]*)\b[^>]*>.*?</\1\s*>", re.DOTALL)
_ANY_TAG_RE = re.compile(r"</?[A-Za-z][\w.:-]*(?:\s[^>]*)?/?>")
_WS_RE = re.compile(r"\s+")

# Short replies whose meaning lives in an earlier turn. Scoring "yes do it" as
# a 1 would under-route the very task it approves, so these fall back to the
# most recent substantive request.
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|ok(?:ay)?|sure|go ahead|proceed|continue|do it|"
    r"please do|sounds good|lgtm|try again|retry|again|same|keep going|"
    r"next|go on|carry on|still (?:failing|broken|not working)|"
    r"that didn'?t work|didn'?t work|not working)\b[\s.!,]*",
    re.IGNORECASE,
)
_FOLLOWUP_MAX_WORDS = 8


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI-style ``content`` (string or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, Mapping):
        text = content.get("text") or content.get("content")
        return text if isinstance(text, str) else ""
    return ""


def _strip_scaffolding(raw: str) -> tuple[str, tuple[str, ...]]:
    """Reduce one raw user message to the words the user typed.

    Returns ``(clean_text, attachment_paths)``. Attachment *paths* are kept
    because a prompt like "refactor these files" makes them meaningful; their
    *bodies* are always dropped.
    """
    if not raw:
        return "", ()

    attachments = tuple(dict.fromkeys(_ATTACHMENT_PATH_RE.findall(raw)))

    # 1. An explicit intent tag wins outright.
    for tag in _INTENT_TAGS:
        found = re.findall(
            rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>", raw, flags=re.IGNORECASE | re.DOTALL
        )
        if found:
            inner = "\n".join(found)
            inner = _FENCE_RE.sub(" ", inner)
            inner = _ANY_TAG_RE.sub(" ", inner)
            return _WS_RE.sub(" ", inner).strip(), attachments

    # 2. Otherwise drop every paired block (repeat to peel nesting), then
    #    code fences, then any stray tag markers.
    text = raw
    for _ in range(4):
        stripped = _PAIRED_BLOCK_RE.sub(" ", text)
        if stripped == text:
            break
        text = stripped
    text = _FENCE_RE.sub(" ", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()

    # 3. If that removed everything, the user's words were *inside* an
    #    unrecognised wrapper. Take the LAST top-level block: editors put
    #    context first and the request after it. Re-admitting the whole raw
    #    text here would bring back exactly the scaffolding (terminals, tool
    #    instructions, workspace info) this function exists to remove.
    if not text:
        blocks = [m.group(0) for m in _PAIRED_BLOCK_RE.finditer(raw)]
        if blocks:
            last = _FENCE_RE.sub(" ", blocks[-1])
            text = _WS_RE.sub(" ", _ANY_TAG_RE.sub(" ", last)).strip()

    return text, attachments


def _user_turns(payload: Mapping[str, Any]) -> list[str]:
    """Raw text of every user turn, oldest first, from wherever it lives.

    Handles OpenAI ``messages`` and Copilot-style ``chatHistory`` entries,
    which may use ``role``/``content`` or ``request``/``prompt`` keys.
    """
    turns: list[str] = []
    for key in ("chatHistory", "history", "messages"):
        entries = payload.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if isinstance(entry, str):
                turns.append(entry)
                continue
            if not isinstance(entry, Mapping):
                continue
            role = str(entry.get("role", "user")).lower()
            if role not in ("user", "human"):
                continue
            text = _content_to_text(entry.get("content"))
            if not text:
                for alt in ("prompt", "request", "message", "text"):
                    if isinstance(entry.get(alt), str):
                        text = entry[alt]
                        break
            if text:
                turns.append(text)
    return turns


def _is_followup(text: str) -> bool:
    """True for a short reply that only makes sense given an earlier turn."""
    words = text.split()
    return bool(words) and len(words) <= _FOLLOWUP_MAX_WORDS and bool(_FOLLOWUP_RE.match(text))


def extract_user_prompt(payload: Mapping[str, Any]) -> ExtractedPrompt:
    """Find the user's actual request inside a chat payload.

    Order of preference:

    1. A top-level prompt field (``prompt``, ``request``, ``query`` …). The
       Copilot participant API sends the typed text here, already clean.
    2. The latest user turn in ``messages`` / ``chatHistory``, with Copilot's
       XML-ish scaffolding stripped (``<userRequest>`` extracted if present).
    3. If that latest turn is a bare follow-up ("yes, go ahead", "still
       failing"), the most recent substantive user turn is prepended, because
       the follow-up inherits that task's difficulty. Chat history is used here
       for its *meaning*, never its size.

    Never raises; returns an empty prompt with ``source="none"`` if nothing
    usable is found.
    """
    if not isinstance(payload, Mapping):
        return ExtractedPrompt(text="", source="none")

    # 1. Direct prompt fields.
    for key in _PROMPT_FIELDS:
        value = payload.get(key)
        text = _content_to_text(value) if not isinstance(value, str) else value
        if text and text.strip():
            clean, attachments = _strip_scaffolding(text)
            if clean:
                return ExtractedPrompt(
                    text=clean, source=f"field:{key}", raw_text=text,
                    attachment_paths=attachments,
                )

    # 2. Conversation turns.
    turns = _user_turns(payload)
    if not turns:
        return ExtractedPrompt(text="", source="none")

    latest_raw = turns[-1]
    latest, attachments = _strip_scaffolding(latest_raw)
    has_tag = any(re.search(rf"<{t}\b", latest_raw, re.IGNORECASE) for t in _INTENT_TAGS)
    source = "messages:userRequest" if has_tag else "messages:last_user"

    # 3. Follow-up: borrow the task from the nearest substantive earlier turn.
    if _is_followup(latest):
        for earlier_raw in reversed(turns[:-1]):
            earlier, earlier_attachments = _strip_scaffolding(earlier_raw)
            if earlier and not _is_followup(earlier):
                return ExtractedPrompt(
                    text=f"{earlier}\n{latest}",
                    source="history_fallback",
                    raw_text=latest_raw,
                    attachment_paths=tuple(dict.fromkeys(attachments + earlier_attachments)),
                    is_followup=True,
                )

    return ExtractedPrompt(
        text=latest, source=source, raw_text=latest_raw,
        attachment_paths=attachments, is_followup=_is_followup(latest),
    )


# ════════════════════════════════════════════════════════════════════════
# Pattern helpers
# ════════════════════════════════════════════════════════════════════════

def _rx(*patterns: str) -> re.Pattern[str]:
    """Compile alternatives as one case-insensitive, word-bounded pattern."""
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


def _hits(pattern: re.Pattern[str], text: str) -> list[str]:
    """Distinct matched phrases, lower-cased, in order of appearance."""
    return list(dict.fromkeys(m.group(0).strip().lower() for m in pattern.finditer(text)))


# ════════════════════════════════════════════════════════════════════════
# 2. Task type
# ════════════════════════════════════════════════════════════════════════
#
# Each task type has a pattern set and a weight. The weight encodes how
# decisive a match is: "why does X deadlock" matches both an explanation
# phrase ("why does") and a debugging one ("deadlock"), and the debugging
# match should win. Ties fall back to _TASK_PRECEDENCE (more demanding first).

_TASK_PATTERNS: dict[TaskType, tuple[re.Pattern[str], float]] = {
    TaskType.EXPLANATION: (_rx(
        r"explain\w*", r"what (?:does|is|are|was)", r"what's", r"how does",
        r"how do(?:es)? (?:this|it|that)", r"walk (?:me )?through",
        r"describe", r"summari[sz]e", r"tell me (?:about|what)",
        r"meaning of", r"help me understand", r"understand",
        r"what happens (?:when|if)", r"difference between",
    ), 1.0),
    TaskType.EDIT: (_rx(
        r"rename", r"change", r"update", r"modify", r"tweak", r"adjust",
        r"re-?format", r"format", r"move", r"replace", r"remove", r"delete",
        r"bump", r"convert", r"add (?:a |an )?(?:comment|comments|docstring|"
        r"type hints?|types|log(?:ging)?|import|prop|field|param(?:eter)?|"
        r"argument|label|tooltip|placeholder|class ?name|style|margin|padding)",
        r"fix (?:the )?typo", r"typo", r"wording", r"sort (?:the )?imports",
    ), 1.2),
    TaskType.BUGFIX: (_rx(
        r"fix\w*", r"bug\w*", r"broken", r"doesn'?t work", r"not working",
        r"error", r"exception", r"crash\w*", r"null ?pointer", r"undefined",
        r"fail(?:s|ing|ed|ure)?", r"incorrect", r"wrong (?:output|result|value)",
        r"type ?error", r"regression", r"stopped working",
    ), 1.4),
    TaskType.DEBUGGING: (_rx(
        r"debug\w*", r"investigate", r"root cause", r"diagnos\w+",
        r"intermittent\w*", r"flaky", r"race condition", r"deadlock\w*",
        r"memory leak", r"leak\w*", r"hang(?:s|ing)?", r"heisenbug",
        r"stack ?trace", r"reproduce", r"why (?:does|is|do) .{0,60}"
        r"(?:fail|crash|hang|slow|leak|break|error|timeout)\w*",
        r"segfault", r"core dump", r"only (?:happens|fails) (?:in|on|under)",
    ), 1.8),
    TaskType.REFACTOR: (_rx(
        r"refactor\w*", r"restructur\w+", r"clean ?up", r"extract",
        r"decouple", r"split (?:this |it )?into", r"reorgani[sz]\w+",
        r"simplif\w+", r"de-?duplicate", r"dry (?:this|it) up",
        r"consolidate", r"modulari[sz]e", r"rewrite", r"port (?:this|it) to",
        r"migrate (?:this|the)? ?(?:code|component|module)s? to",
        r"convert (?:this |the )?(?:project |codebase )?to typescript",
    ), 1.6),
    TaskType.FEATURE: (_rx(
        r"implement\w*", r"add (?:support|a new|an? \w+ (?:endpoint|page|feature|"
        r"screen|view|form|api|button|flow|service|integration|command|option))",
        r"add \w+(?:ing|ion)", r"build", r"create", r"develop", r"introduce",
        r"integrate", r"set ?up", r"wire (?:up|in)", r"support for",
        r"new (?:feature|endpoint|page|screen|service|module|component)",
        r"write (?:a|an) (?:service|module|api|endpoint|script|cli|tool|"
        r"component|hook|worker|job|scheduler|parser)",
        r"pagination", r"add pagination", r"login", r"sign ?up", r"checkout",
        r"export (?:to|as)", r"upload", r"notifications?",
    ), 1.5),
    TaskType.OPTIMIZATION: (_rx(
        r"optimi[sz]\w+", r"speed (?:up|it up)", r"faster", r"performance",
        r"slow\w*", r"latency", r"reduce (?:memory|cpu|allocations|load time|"
        r"bundle size|latency)", r"throughput", r"n ?\+ ?1", r"bottleneck",
        r"profil\w+", r"too much memory", r"make (?:it|this) scale",
    ), 1.7),
    TaskType.SECURITY: (_rx(
        r"security (?:review|audit|issue|hole|fix)", r"audit", r"vulnerab\w+",
        r"harden\w*", r"secure (?:this|the|it|our)", r"sanitiz\w+",
        r"xss", r"csrf", r"sql injection", r"injection", r"cve-?\d*",
        r"pen ?test\w*", r"threat model\w*", r"owasp", r"exploit\w*",
        r"leak(?:ing|ed)? (?:secrets?|credentials?|tokens?|keys?)",
    ), 2.0),
    TaskType.ARCHITECTURE: (_rx(
        r"architect\w*", r"design (?:the|a|an|our) (?:architecture|structure|"
        r"module|layer|data model|schema|domain|api|interface|abstraction)",
        r"microservices?", r"event[- ]driven", r"cqrs", r"event sourcing",
        r"hexagonal", r"clean architecture", r"domain[- ]driven", r"ddd",
        r"split (?:the )?monolith", r"decompose", r"service boundar\w+",
        r"bounded contexts?", r"plugin (?:system|architecture)",
    ), 2.2),
    TaskType.SYSTEM_DESIGN: (_rx(
        r"system design",
        r"design (?:a|an|the) .{0,40}(?:system|platform|infrastructure|backend)",
        r"(?:scalable|distributed|highly available|fault[- ]tolerant) "
        r"(?:system|platform|architecture|service)",
        r"\d[\d,.]*\s*(?:k|m|b|thousand|million|billion)\+?\s+(?:users|"
        r"requests|rps|qps|tps|transactions|events|messages|customers)",
        r"(?:millions|billions) of (?:users|requests|events|transactions)",
        r"multi[- ]region", r"globally distributed", r"high availability",
    ), 3.0),
}

# Tie-break order: the more demanding reading wins a tie, because
# under-routing a hard task costs more than over-routing an easy one.
_TASK_PRECEDENCE: tuple[TaskType, ...] = (
    TaskType.SYSTEM_DESIGN, TaskType.ARCHITECTURE, TaskType.SECURITY,
    TaskType.DEBUGGING, TaskType.OPTIMIZATION, TaskType.REFACTOR,
    TaskType.FEATURE, TaskType.BUGFIX, TaskType.EDIT, TaskType.EXPLANATION,
)


def detect_task_type(prompt: str) -> Detection:
    """Classify the primary kind of work requested.

    Each task type accumulates ``weight × distinct_phrases_matched``; the
    highest total wins, precedence breaks ties.

    Two deliberate rules on top of the raw tallies:

    * Security and architecture are also *dimensions* scored separately (see
      ``detect_security_signals`` / ``detect_architecture_signals``). They win
      the task type only when the user's *action* is a security or design
      action — "audit the auth flow", "design the architecture". "Implement
      OAuth" stays a FEATURE with a security dimension, which is what the
      spec's anchor (12) expects.
    * Explanation phrases lose to any action. "How do I add pagination?" in an
      editor chat is a request to add pagination, not a lecture.
    """
    if not prompt.strip():
        return Detection(TaskType.UNKNOWN, (), 0.0)

    tallies: dict[TaskType, float] = {}
    evidence: dict[TaskType, list[str]] = {}
    for task, (pattern, weight) in _TASK_PATTERNS.items():
        found = _hits(pattern, prompt)
        if found:
            tallies[task] = weight * len(found)
            evidence[task] = found

    if not tallies:
        # Nothing recognisable. A question reads as explanation; an
        # imperative with no known verb reads as a small edit.
        guess = TaskType.EXPLANATION if prompt.rstrip().endswith("?") else TaskType.EDIT
        return Detection(guess, (), 0.25)

    # Explanation only wins when nothing else matched at all.
    if len(tallies) > 1:
        tallies.pop(TaskType.EXPLANATION, None)

    # "fix" matches BUGFIX, but "fix the typo" is an EDIT and "fix the race
    # condition" is DEBUGGING — defer to the more specific reading when the
    # only bugfix evidence is the bare verb.
    bug = evidence.get(TaskType.BUGFIX, [])
    if bug and all(b.startswith("fix") for b in bug) and (
        TaskType.EDIT in tallies or TaskType.DEBUGGING in tallies
        or TaskType.SECURITY in tallies or TaskType.OPTIMIZATION in tallies
    ):
        tallies.pop(TaskType.BUGFIX, None)

    best = max(tallies.values())
    winners = [t for t in _TASK_PRECEDENCE if tallies.get(t) == best]
    chosen = winners[0]

    # Strength: how far the winner stands above the runner-up. Used by
    # confidence — a clear winner is a confident classification.
    ordered = sorted(tallies.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) / ordered[0] if len(ordered) > 1 else 1.0
    strength = min(1.0, 0.55 + 0.45 * margin)

    return Detection(chosen, tuple(evidence.get(chosen, ())), strength)


# ════════════════════════════════════════════════════════════════════════
# 3. Scope
# ════════════════════════════════════════════════════════════════════════
#
# Scope is read from the prompt's own words, widest match first. Payload
# context (attachments, diagnostics) can widen it — but only when the prompt
# points at that context ("these files", "all the errors"). A Copilot request
# that merely *carries* twelve attached files is still about one function if
# the user said "this function".

_SCOPE_PATTERNS: tuple[tuple[Scope, re.Pattern[str]], ...] = (
    (Scope.CROSS_REPOSITORY, _rx(
        r"cross[- ]repo\w*", r"across (?:all )?(?:the )?(?:repos|repositories)",
        r"(?:multiple|several|both|other|another|all) (?:repos|repositories)",
        r"(?:sibling|upstream|downstream) (?:repo|repository|project)s?",
        r"across (?:services|projects)", r"multi[- ]repo",
    )),
    (Scope.REPOSITORY, _rx(
        r"(?:entire|whole|full) (?:codebase|repo|repository|project|app|application)",
        r"(?:across|throughout) (?:the )?(?:codebase|repo|repository|project)",
        r"(?:repo|repository|project|codebase)[- ]wide", r"everywhere",
        r"all (?:the )?(?:files|modules|components|endpoints|services|usages|callers)",
        r"every (?:file|module|component|endpoint|usage|caller)",
    )),
    (Scope.SERVICE, _rx(
        r"system", r"platform", r"(?:micro)?service", r"backend", r"back-end",
        r"server", r"infrastructure", r"pipeline", r"application",
    )),
    (Scope.MULTI_MODULE, _rx(
        r"frontend and backend", r"front-?end and back-?end",
        r"(?:client|ui) and (?:server|api)", r"(?:several|multiple|these|those) "
        r"(?:files|modules|components|classes|packages)",
        r"across (?:modules|files|components|layers)", r"full[- ]stack",
        r"end[- ]to[- ]end", r"both (?:the )?\w+ and (?:the )?\w+",
    )),
    (Scope.MODULE, _rx(
        r"module", r"file", r"package", r"endpoint", r"route", r"controller",
        r"page", r"screen", r"view", r"component", r"hook", r"api",
    )),
    (Scope.CLASS, _rx(r"class", r"struct", r"interface", r"model", r"type")),
    (Scope.FUNCTION, _rx(
        r"function", r"method", r"this line", r"these lines", r"snippet",
        r"selection", r"selected (?:code|text)", r"expression", r"variable",
        r"lambda", r"regex", r"query",
    )),
)

# Scope order, for "wider than" comparisons.
_SCOPE_ORDER = tuple(Scope)

# Default scope per task type when the prompt names none.
_DEFAULT_SCOPE: dict[TaskType, Scope] = {
    TaskType.EXPLANATION: Scope.FUNCTION,
    TaskType.EDIT: Scope.FUNCTION,
    TaskType.BUGFIX: Scope.MODULE,
    TaskType.DEBUGGING: Scope.MODULE,
    TaskType.REFACTOR: Scope.MODULE,
    TaskType.FEATURE: Scope.MODULE,
    TaskType.OPTIMIZATION: Scope.MODULE,
    TaskType.SECURITY: Scope.MODULE,
    TaskType.ARCHITECTURE: Scope.SERVICE,
    TaskType.SYSTEM_DESIGN: Scope.SERVICE,
    TaskType.UNKNOWN: Scope.FUNCTION,
}

# A floor the task type itself implies: you cannot design a system at
# function scope, whatever words were used.
_MIN_SCOPE: dict[TaskType, Scope] = {
    TaskType.ARCHITECTURE: Scope.MULTI_MODULE,
    TaskType.SYSTEM_DESIGN: Scope.SERVICE,
}

# Phrases showing the user is pointing at the payload's own context.
_REFERS_TO_ATTACHMENTS = _rx(
    r"(?:these|those|the attached|attached|all (?:the|these|of these)?) ?files?",
    r"the (?:selected|open|referenced) files",
)
_REFERS_TO_DIAGNOSTICS = _rx(
    r"(?:all|these|those|the) (?:\w+ )?(?:errors|warnings|problems|diagnostics|lint(?:s| errors)?)",
    r"problems (?:panel|tab)", r"type errors", r"compile errors", r"build errors",
)

_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\.{0,2}[\\/])?(?:[\w.@-]+[\\/])*[\w.@-]+"
    r"\.(?:py|ts|tsx|js|jsx|mjs|cjs|java|go|rs|rb|cs|cpp|cc|c|h|hpp|kt|swift|"
    r"php|scala|sql|sh|ps1|ya?ml|json|toml|md|html|css|scss|vue|svelte|dart|ex|exs)\b"
)


# Library and framework names that look like file paths ("next.js") but are
# not files the user is pointing at.
_NOT_FILES = {
    "node.js", "next.js", "nuxt.js", "vue.js", "react.js", "express.js",
    "nest.js", "three.js", "d3.js", "chart.js", "ember.js", "backbone.js",
    "angular.js", "socket.js", "p5.js", "tensorflow.js", "moment.js",
}


def _typed_paths(text: str) -> set[str]:
    """File paths the user typed, minus framework names that merely look like one."""
    return {p for p in _PATH_RE.findall(text) if p.lower() not in _NOT_FILES}


def _wider(a: Scope, b: Scope) -> Scope:
    return a if _SCOPE_ORDER.index(a) >= _SCOPE_ORDER.index(b) else b


def _payload_file_count(payload: Mapping[str, Any], prompt: ExtractedPrompt) -> int:
    """Distinct files the payload carries as references/attachments."""
    paths: set[str] = set(prompt.attachment_paths)
    for key in ("references", "attachments", "files"):
        items = payload.get(key)
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        for item in items:
            if isinstance(item, str):
                paths.add(item)
            elif isinstance(item, Mapping):
                for k in ("path", "filePath", "uri", "file", "name", "id"):
                    v = item.get(k)
                    if isinstance(v, str):
                        paths.add(v)
                        break
    return len(paths)


def _payload_diagnostic_files(payload: Mapping[str, Any]) -> int:
    """Distinct files that have diagnostics attached to the payload."""
    diags = payload.get("diagnostics")
    if not isinstance(diags, Sequence) or isinstance(diags, (str, bytes)):
        return 0
    files: set[str] = set()
    for d in diags:
        if isinstance(d, Mapping):
            for k in ("file", "path", "uri", "filePath", "source"):
                v = d.get(k)
                if isinstance(v, str):
                    files.add(v)
                    break
    return len(files) or (1 if diags else 0)


def _files_to_scope(n: int) -> Scope | None:
    if n >= 10:
        return Scope.REPOSITORY
    if n >= 2:
        return Scope.MULTI_MODULE
    if n == 1:
        return Scope.MODULE
    return None


def detect_scope(
    prompt: ExtractedPrompt,
    task_type: TaskType,
    payload: Mapping[str, Any] | None = None,
) -> Detection:
    """How far the change or analysis reaches.

    1. The widest scope phrase in the prompt ("this function" < "frontend and
       backend" < "across the codebase" < "across repositories").
    2. File paths the user typed: two or more distinct paths is multi-module.
    3. Payload context — ONLY if the prompt points at it. "Fix all the type
       errors" + diagnostics in 14 files is repository scope; "add a prop"
       with 14 attached files is not.
    4. Otherwise a per-task default, and a per-task minimum (system design is
       never function-scoped).
    """
    text = prompt.text
    payload = payload or {}
    evidence: list[str] = []
    found: Scope | None = None

    for scope, pattern in _SCOPE_PATTERNS:
        hit = _hits(pattern, text)
        if hit:
            found = scope
            evidence.extend(hit)
            break  # patterns are ordered widest-first

    typed_paths = _typed_paths(text)
    by_paths = _files_to_scope(len(typed_paths))
    if by_paths is not None:
        evidence.append(f"{len(typed_paths)} path(s) named")
        found = by_paths if found is None else _wider(found, by_paths)

    if _REFERS_TO_ATTACHMENTS.search(text):
        n = _payload_file_count(payload, prompt)
        by_attached = _files_to_scope(n)
        if by_attached is not None:
            evidence.append(f"{n} referenced file(s)")
            found = by_attached if found is None else _wider(found, by_attached)

    if _REFERS_TO_DIAGNOSTICS.search(text):
        n = _payload_diagnostic_files(payload)
        by_diag = _files_to_scope(n)
        if by_diag is not None:
            evidence.append(f"diagnostics in {n} file(s)")
            found = by_diag if found is None else _wider(found, by_diag)

    explicit = found is not None
    scope = found if found is not None else _DEFAULT_SCOPE.get(task_type, Scope.MODULE)
    minimum = _MIN_SCOPE.get(task_type)
    if minimum is not None:
        scope = _wider(scope, minimum)

    return Detection(scope, tuple(evidence), 1.0 if explicit else 0.4)


# ════════════════════════════════════════════════════════════════════════
# 4. Architectural and security signals
# ════════════════════════════════════════════════════════════════════════
#
# These are *dimensions*, not task types: any task can carry them. They are
# grouped into concept families so that "redis cache" and "caching layer" count
# once, not twice — the score should reflect how many distinct hard problems
# the task involves, not how many synonyms the user typed.

_ARCH_CONCEPTS: dict[str, re.Pattern[str]] = {
    # Bare "scale" is avoided: "scale the image" is a CSS transform, not a
    # capacity problem.
    "scalability": _rx(
        r"scalab(?:le|ility)", r"scaling", r"(?:to|at|will|must|should) scale",
        r"scale (?:up|out|horizontally|to \d\w*)", r"horizontal(?:ly)? scal\w+",
        r"high (?:traffic|load|throughput)",
        r"\d[\d,.]*\s*(?:k|m|b|thousand|million|billion)\+?\s+(?:users|requests|"
        r"rps|qps|tps|transactions|events|messages|customers)",
        r"(?:millions|billions) of",
    ),
    "microservices": _rx(r"micro-?services?", r"service mesh", r"service boundar\w+"),
    "distributed_systems": _rx(
        r"distributed", r"consensus", r"raft", r"paxos", r"multi[- ]region",
        r"eventual(?:ly)? consisten\w+", r"partition toleran\w+", r"cap theorem",
    ),
    "event_driven": _rx(
        r"event[- ]driven", r"event bus", r"pub ?/ ?sub", r"publish[- ]subscribe",
        r"webhooks?", r"event stream\w*",
    ),
    "caching": _rx(r"cach(?:e|es|ing|ed)", r"redis", r"memcached?", r"cdn"),
    "cqrs_event_sourcing": _rx(r"cqrs", r"event sourcing", r"event store"),
    "messaging": _rx(
        r"kafka", r"rabbit ?mq", r"sqs", r"sns", r"nats", r"pulsar",
        r"message (?:queue|broker)", r"queues?", r"background (?:jobs?|workers?)",
    ),
    "sharding": _rx(r"shard\w*", r"partition(?:ing|ed)? (?:the )?(?:data|table|db|database)"),
    "replication": _rx(r"replica\w*", r"read replicas?", r"failover", r"leader election"),
    "high_availability": _rx(
        r"high(?:ly)? availab\w+", r"fault[- ]toleran\w+", r"zero[- ]downtime",
        r"disaster recovery", r"redundan\w+", r"\bsla\b", r"99\.9+",
    ),
    "transactions": _rx(
        r"distributed transactions?", r"sagas?", r"two[- ]phase commit", r"2pc",
        r"idempoten\w+", r"exactly[- ]once", r"outbox",
    ),
    "migration": _rx(
        r"(?:data|database|schema|zero[- ]downtime) migrations?",
        r"migrat(?:e|ing|ion) (?:from|off|to) \w+", r"backfill\w*",
    ),
    "concurrency": _rx(
        r"concurren\w+", r"race conditions?", r"deadlocks?", r"locking",
        r"mutex\w*", r"thread[- ]safe\w*", r"parallel(?:ism|ize)?", r"async(?:io)? (?:race|ordering)",
    ),
}

_SECURITY_FAMILIES: dict[str, re.Pattern[str]] = {
    "authentication": _rx(
        r"auth(?!or)\w*", r"log ?in", r"sign ?in", r"sign ?up", r"sso",
        r"oauth\s?2?", r"openid", r"oidc", r"saml", r"sessions?", r"mfa", r"2fa",
        r"passwords?", r"passkeys?", r"webauthn",
    ),
    "tokens": _rx(
        r"jwts?", r"refresh tokens?", r"access tokens?", r"bearer tokens?",
        r"token (?:rotation|refresh|expiry|revocation)", r"id tokens?",
    ),
    # Bare "role" is avoided: "what role does this play" is not authorization.
    "authorization": _rx(
        r"rbac", r"abac", r"permissions?", r"role[- ]based",
        r"(?:user|admin|access) roles?", r"roles? and permissions",
        r"access control", r"acls?", r"authori[sz]\w+", r"multi[- ]tenan\w+",
    ),
    "cryptography": _rx(
        r"encrypt\w*", r"decrypt\w*", r"hash(?:ing)? (?:passwords?|secrets?)",
        r"bcrypt", r"argon2?", r"scrypt", r"tls", r"ssl", r"certificates?",
        r"hmac", r"signing", r"signatures?", r"crypto\w*", r"key rotation",
    ),
    "secrets": _rx(
        r"secrets?", r"api[- ]keys?", r"private keys?", r"credentials?",
        r"vault", r"kms", r"dotenv", r"env (?:file|secrets?)",
    ),
    "payments": _rx(
        r"payments?", r"billing", r"checkout", r"pci", r"credit cards?",
        r"stripe", r"invoic\w+", r"refunds?", r"subscriptions?",
    ),
    "vulnerabilities": _rx(
        r"xss", r"csrf", r"sql injection", r"injection", r"cve", r"vulnerab\w+",
        r"sanitiz\w+", r"owasp", r"ssrf", r"rce", r"exploit\w*",
    ),
    "privacy": _rx(r"pii", r"gdpr", r"hipaa", r"personal data", r"data retention"),
}


def _concepts(prompt: str, table: Mapping[str, re.Pattern[str]]) -> dict[str, list[str]]:
    """Concept families present in the prompt, with the phrases that matched."""
    out: dict[str, list[str]] = {}
    for name, pattern in table.items():
        hit = _hits(pattern, prompt)
        if hit:
            out[name] = hit
    return out


def detect_architecture_signals(prompt: str) -> Detection:
    """Distinct architectural concepts the task involves.

    Value: sorted list of concept families (``scalability``, ``caching``,
    ``messaging`` …). Each family is one hard problem the implementer has to
    get right; "Redis cache in front of a Kafka consumer" is two.
    """
    found = _concepts(prompt, _ARCH_CONCEPTS)
    evidence = tuple(p for phrases in found.values() for p in phrases)
    return Detection(sorted(found), evidence, min(1.0, len(found) / 3))


def detect_security_signals(prompt: str) -> Detection:
    """Distinct security concept families the task touches.

    Value: sorted list of families (``authentication``, ``tokens``,
    ``payments`` …). "OAuth login with refresh tokens" is two families —
    authentication and token handling — which is exactly where the subtle
    bugs live.
    """
    found = _concepts(prompt, _SECURITY_FAMILIES)
    evidence = tuple(p for phrases in found.values() for p in phrases)
    return Detection(sorted(found), evidence, min(1.0, len(found) / 3))


# ════════════════════════════════════════════════════════════════════════
# 5. Dependency complexity
# ════════════════════════════════════════════════════════════════════════
#
# How many components must work together for the task to be done. The spec's
# scale: one function = 1, controller + service = 2, frontend + backend + DB
# = 3, frontend + backend + auth + DB + queue = 5.
#
# Components are detected as families from the prompt. Some phrases imply more
# than one: OAuth is an auth component AND an external identity provider;
# "payment" implies an external payment gateway.

_COMPONENTS: dict[str, re.Pattern[str]] = {
    "frontend": _rx(
        r"front-?end", r"ui", r"client[- ]side", r"react", r"vue", r"angular",
        r"svelte", r"next\.?js", r"browser", r"page", r"screen", r"form",
    ),
    "api_layer": _rx(
        r"back-?end", r"api", r"endpoints?", r"controllers?", r"routes?",
        r"handlers?", r"rest", r"graphql", r"grpc", r"server",
    ),
    "service_layer": _rx(
        r"service (?:layer|class)", r"business logic", r"domain (?:layer|logic)",
        r"use[- ]cases?", r"(?<!micro)services?",
    ),
    "database": _rx(
        r"database", r"db", r"sql", r"postgres\w*", r"mysql", r"sqlite",
        r"mongo\w*", r"dynamo\w*", r"schema", r"tables?", r"orm", r"prisma",
        r"migrations?", r"persist\w*", r"storage layer",
    ),
    "cache": _rx(r"cach(?:e|es|ing)", r"redis", r"memcached?"),
    "queue": _rx(
        r"queues?", r"kafka", r"rabbit ?mq", r"sqs", r"pub ?/ ?sub",
        r"event bus", r"message broker", r"background (?:jobs?|workers?)",
        r"cron", r"scheduler",
    ),
    "auth": _rx(
        r"auth(?!or)\w*", r"log ?in", r"sign ?in", r"oauth\s?2?", r"sso",
        r"jwts?", r"sessions?", r"permissions?", r"rbac",
    ),
    "external_service": _rx(
        r"third[- ]party", r"external (?:api|service)", r"webhooks?",
        r"oauth\s?2?", r"openid", r"stripe", r"paypal", r"twilio", r"sendgrid",
        r"payment (?:gateway|provider)", r"payments?", r"identity provider",
        r"s3", r"openai", r"slack", r"github api",
    ),
    "file_storage": _rx(r"uploads?", r"file storage", r"blob", r"object storage", r"s3"),
    "search": _rx(r"elastic ?search", r"open ?search", r"search index", r"full[- ]text search", r"algolia"),
    "notifications": _rx(r"email\w*", r"sms", r"push notifications?", r"notifications?"),
    "infrastructure": _rx(
        r"docker\w*", r"kubernetes", r"k8s", r"terraform", r"helm",
        r"ci ?/ ?cd", r"load balanc\w+", r"cdn", r"nginx", r"deploy\w*",
    ),
    "observability": _rx(r"logging", r"metrics", r"tracing", r"monitoring", r"telemetry", r"alerting"),
}

# Minimum components a task type implies even when none are named:
# architecture and system design are by definition several parts together.
_MIN_COMPONENTS: dict[TaskType, int] = {
    TaskType.ARCHITECTURE: 3,
    TaskType.SYSTEM_DESIGN: 3,
}


def estimate_dependency_complexity(prompt: str, task_type: TaskType) -> Detection:
    """Estimate how many components must cooperate. Value is an int >= 1.

    Counts distinct component families named in the prompt, floored by the
    task type's implied minimum. Payload file counts are deliberately NOT used:
    twelve attached files in one module are still one component.
    """
    found = _concepts(prompt, _COMPONENTS)
    count = max(1, len(found), _MIN_COMPONENTS.get(task_type, 1))
    evidence = tuple(f"{k}: {', '.join(v)}" for k, v in sorted(found.items()))
    return Detection(count, evidence, min(1.0, len(found) / 5))


# ════════════════════════════════════════════════════════════════════════
# 6. Reasoning depth
# ════════════════════════════════════════════════════════════════════════
#
# Separate from task type and scope: a one-line fix to a race condition is
# narrow in scope but deep in reasoning. Depth is HIGH when the task involves
# architecture, security, distributed systems, migrations or hard debugging;
# MEDIUM for ordinary engineering (bugfixes, CRUD, API integration); LOW for
# explanation, formatting and simple edits.

_HIGH_DEPTH_TASKS = {TaskType.ARCHITECTURE, TaskType.SYSTEM_DESIGN, TaskType.SECURITY}
_MEDIUM_DEPTH_TASKS = {
    TaskType.BUGFIX, TaskType.FEATURE, TaskType.REFACTOR,
    TaskType.DEBUGGING, TaskType.OPTIMIZATION,
}

# Debugging that is genuinely hard: non-deterministic, emergent or systemic.
_HARD_DEBUG = _rx(
    r"race conditions?", r"deadlocks?", r"memory leaks?", r"intermittent\w*",
    r"flaky", r"heisenbug", r"only (?:happens|fails) (?:in|on|under)",
    r"under load", r"in production", r"non-?deterministic", r"corrupt\w*",
    r"segfault", r"undefined behaviou?r",
)

# Factors that make an ordinary (MEDIUM) task harder without making it an
# architecture problem.
_COMPLICATORS = _rx(
    r"backwards?[- ]compatib\w+", r"without breaking", r"edge cases?",
    r"validation", r"error handling", r"retr(?:y|ies)", r"rollback",
    r"transactions?", r"timeouts?", r"versioning", r"pagination with (?:filter|sort)\w*",
    r"cursor[- ]based", r"rate[- ]limit\w*", r"localization|i18n",
    r"accessibility|a11y", r"streaming", r"real[- ]?time", r"offline",
)


def detect_reasoning_depth(
    prompt: str,
    task_type: TaskType,
    architecture: Sequence[str],
    security: Sequence[str],
) -> Detection:
    """Classify reasoning depth; ``strength`` carries how much evidence backs it.

    ``evidence`` lists the *high-depth concepts* (or complicators, for MEDIUM)
    so ``compute_score`` can scale the depth points by how many distinct hard
    problems the task actually has.
    """
    hard_debug = _hits(_HARD_DEBUG, prompt)

    # High-depth concepts: each architectural family, security as ONE concept
    # (it already earns its own points in compute_score — counting each family
    # here too would double-charge it), and each hard-debugging phrase family.
    high: list[str] = list(architecture)
    if security:
        high.append("security")
    if hard_debug:
        high.append("hard_debugging")

    if task_type in _HIGH_DEPTH_TASKS or high:
        return Detection(ReasoningDepth.HIGH, tuple(high), min(1.0, len(high) / 3))

    if task_type in _MEDIUM_DEPTH_TASKS:
        complicators = _hits(_COMPLICATORS, prompt)
        return Detection(
            ReasoningDepth.MEDIUM, tuple(complicators), min(1.0, len(complicators) / 2)
        )

    return Detection(ReasoningDepth.LOW, (), 0.0)


# ════════════════════════════════════════════════════════════════════════
# 7. Score composition
# ════════════════════════════════════════════════════════════════════════

# (min, max) points per task type, straight from the spec's scoring model.
# Debugging and optimization are not in the spec's table; they sit between
# bugfix and refactor at the low end (a clear stack trace, an obvious N+1) and
# reach feature territory at the top (a heisenbug, a latency budget).
TASK_POINTS: dict[TaskType, tuple[float, float]] = {
    TaskType.EXPLANATION: (0, 2),
    TaskType.EDIT: (1, 3),
    TaskType.BUGFIX: (2, 5),
    TaskType.DEBUGGING: (3, 7),
    TaskType.OPTIMIZATION: (3, 7),
    TaskType.REFACTOR: (3, 6),
    TaskType.FEATURE: (4, 8),
    TaskType.SECURITY: (3, 6),
    TaskType.ARCHITECTURE: (5, 8),
    TaskType.SYSTEM_DESIGN: (7, 10),
    TaskType.UNKNOWN: (0, 0),
}

# Where in its task range a request sits, driven by scope. Narrow scopes sit
# at the bottom of the range; the range's own width does the rest. Monotonic:
# a wider scope never places a task lower.
_SCOPE_POSITION: dict[Scope, float] = {
    Scope.FUNCTION: 0.0,
    Scope.CLASS: 0.05,
    Scope.MODULE: 0.10,
    Scope.MULTI_MODULE: 0.35,
    Scope.SERVICE: 0.60,
    Scope.REPOSITORY: 0.80,
    Scope.CROSS_REPOSITORY: 1.0,
}

# Extra position within the range.
_POS_PER_EXTRA_ACTION = 0.15   # "add X, then Y, and also Z"
_POS_SCALE = 0.30              # an explicit scale target ("10 million users")
_POS_BREADTH = 0.15            # "complete", "production-ready", "end-to-end"
_POS_CODE_BOUND_EXPLAIN = 0.5  # explaining concrete code vs. a general question

_ACTION_VERBS = _rx(
    r"add", r"implement", r"create", r"build", r"write", r"fix", r"refactor",
    r"update", r"remove", r"delete", r"rename", r"migrate", r"optimi[sz]e",
    r"test", r"document", r"integrate", r"replace", r"convert", r"deploy",
    r"design", r"extract", r"move", r"validate", r"secure", r"cache",
)
_BREADTH_WORDS = _rx(
    r"complete", r"full(?:y)?", r"production[- ]ready", r"end[- ]to[- ]end",
    r"robust", r"comprehensive", r"from scratch", r"entire",
)
_CODE_REFERENCE = _rx(
    r"this", r"these", r"the (?:code|function|method|class|file|component|snippet)",
    r"selected", r"selection", r"above", r"below",
)
_SCALE_RE = re.compile(
    r"\d[\d,.]*\s*(?:k|m|b|thousand|million|billion)\+?\s+(?:users|requests|rps|"
    r"qps|tps|transactions|events|messages|customers)|(?:millions|billions) of",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Every component of the final score, for tracing a verdict."""

    base: float
    task: float
    security: float
    dependency: float
    depth: float
    raw_total: float
    final: int

    def as_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _task_points(
    task: TaskType, scope: Scope, prompt: str, *, scale: bool
) -> float:
    """Points for the task type, positioned within its range.

    Position = scope position + extra requested actions + explicit scale +
    breadth words, clamped to [0, 1]. Explanation additionally gets half its
    range when it is about concrete code ("explain *this* function") rather
    than a general question — reading code is work; recalling syntax is not.
    """
    lo, hi = TASK_POINTS[task]
    position = _SCOPE_POSITION[scope]

    extra_actions = max(0, len(_hits(_ACTION_VERBS, prompt)) - 1)
    position += _POS_PER_EXTRA_ACTION * min(extra_actions, 3)
    if scale:
        position += _POS_SCALE
    if _BREADTH_WORDS.search(prompt):
        position += _POS_BREADTH
    if task is TaskType.EXPLANATION and _CODE_REFERENCE.search(prompt):
        position += _POS_CODE_BOUND_EXPLAIN

    position = max(0.0, min(1.0, position))
    return lo + (hi - lo) * position


def _security_points(task: TaskType, families: Sequence[str]) -> float:
    """+3 for the first security family, +1 for each further one, max +6.

    Skipped when the task type IS security: its task range already charges
    for it, and charging twice would push every audit to the top band.
    """
    if not families or task is TaskType.SECURITY:
        return 0.0
    return float(min(6, 3 + (len(families) - 1)))


def _dependency_points(count: int) -> float:
    """Components beyond the first, max +5 (1 → 0, 2 → 1, 3 → 2, 5 → 4)."""
    return float(max(0, min(5, count - 1)))


def _depth_points(depth: Detection) -> float:
    """LOW +0; MEDIUM +0..2 by complicators; HIGH +2..5 by distinct hard concepts."""
    n = len(depth.evidence)
    if depth.value is ReasoningDepth.HIGH:
        return float(2 + min(3, max(0, n - 1)))
    if depth.value is ReasoningDepth.MEDIUM:
        return float(min(2, n))
    return 0.0


def compute_score(
    task: TaskType,
    scope: Scope,
    depth: Detection,
    security_families: Sequence[str],
    dependency_count: int,
    prompt: str,
) -> ScoreBreakdown:
    """Combine the components into a 1-20 score.

    ``1 + task + security + dependency + depth``, rounded half-up and clamped.
    Every component is independent and capped, so no single signal — least of
    all payload size, which is not an input at all — can run the score to 20
    by itself.
    """
    base = 1.0
    if task is TaskType.UNKNOWN:
        return ScoreBreakdown(base, 0.0, 0.0, 0.0, 0.0, base, 1)

    scale = bool(_SCALE_RE.search(prompt))
    task_pts = _task_points(task, scope, prompt, scale=scale)
    security_pts = _security_points(task, security_families)
    dependency_pts = _dependency_points(dependency_count)
    depth_pts = _depth_points(depth)

    raw = base + task_pts + security_pts + dependency_pts + depth_pts
    final = max(1, min(20, int(raw + 0.5)))   # half-up, not banker's rounding
    return ScoreBreakdown(base, task_pts, security_pts, dependency_pts, depth_pts, raw, final)


# ════════════════════════════════════════════════════════════════════════
# 8. Confidence
# ════════════════════════════════════════════════════════════════════════

def _confidence(
    prompt: ExtractedPrompt,
    task: Detection,
    scope: Detection,
) -> float:
    """How much to trust the score, 0..1.

    High when the prompt came from a clean source, the task type won clearly
    and scope was stated rather than defaulted. Low for empty or one-word
    prompts, follow-ups resolved through history, and near-tie classifications.
    """
    if not prompt.text:
        return 0.05

    conf = 0.35
    conf += 0.35 * task.strength                     # clear task-type winner
    conf += 0.15 * scope.strength                    # scope stated, not assumed

    words = len(prompt.text.split())
    if words >= 4:
        conf += 0.10
    elif words <= 1:
        conf -= 0.15

    if prompt.source.startswith("field:") or prompt.source == "messages:userRequest":
        conf += 0.05                                 # unambiguous extraction
    if prompt.is_followup:
        conf -= 0.20                                 # task inferred from history

    return round(max(0.05, min(0.99, conf)), 2)


# ════════════════════════════════════════════════════════════════════════
# Main API
# ════════════════════════════════════════════════════════════════════════

def score_request(payload: Mapping[str, Any]) -> ComplexityScore:
    """Score a Copilot Chat / VS Code chat request payload for difficulty.

    Resilient to missing or oddly-shaped fields: any payload, including an
    empty dict, yields a valid ``ComplexityScore`` (score 1, confidence ~0,
    task ``unknown`` if no prompt can be found).
    """
    if not isinstance(payload, Mapping):
        payload = {}

    prompt = extract_user_prompt(payload)
    text = prompt.text

    task = detect_task_type(text)
    scope = detect_scope(prompt, task.value, payload)
    architecture = detect_architecture_signals(text)
    security = detect_security_signals(text)
    dependency = estimate_dependency_complexity(text, task.value)
    depth = detect_reasoning_depth(text, task.value, architecture.value, security.value)

    breakdown = compute_score(
        task=task.value,
        scope=scope.value,
        depth=depth,
        security_families=security.value,
        dependency_count=dependency.value,
        prompt=text,
    )

    signals: dict[str, Any] = {
        # The three keys from the spec's example output.
        "architecture": bool(architecture.value),
        "security": bool(security.value),
        "dependency_count": dependency.value,
        # Evidence, for tracing a verdict back to the words that caused it.
        "architecture_concepts": list(architecture.value),
        "security_families": list(security.value),
        "components": [e.split(":", 1)[0] for e in dependency.evidence],
        "scale": bool(_SCALE_RE.search(text)),
        "task_evidence": list(task.evidence),
        "scope_evidence": list(scope.evidence),
        "depth_evidence": list(depth.evidence),
        "breakdown": breakdown.as_dict(),
        "prompt_source": prompt.source,
        "prompt": text[:4000],            # the extracted request, scaffolding stripped
        "prompt_excerpt": text[:200],
        "scaffolding_stripped_chars": max(0, len(prompt.raw_text) - len(text)),
    }

    return ComplexityScore(
        complexity_score=breakdown.final,
        task_type=task.value,
        reasoning_depth=depth.value,
        scope=scope.value,
        confidence=_confidence(prompt, task, scope),
        signals=signals,
    )
