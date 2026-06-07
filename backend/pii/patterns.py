"""
Sensitive entity detection patterns.

Covers credentials and PII that realistically appear in code comments,
log pastes, and casual prompts sent to an LLM.

Priority ladder (higher wins on overlap):
  JWT 95 > AI_KEY 90 > GH_TOKEN 88 > AWS_KEY 87 > DB_URL 85 > API_KEY 80
  > SSN 25 > EMAIL 20 > PHONE 10
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    PEM_KEY  = "PEM_KEY"
    JWT      = "JWT"
    AI_KEY   = "AI_KEY"
    GH_TOKEN = "GH_TOKEN"
    AWS_KEY  = "AWS_KEY"
    DB_URL   = "DB_URL"
    API_KEY  = "API_KEY"
    SSN      = "SSN"
    EMAIL    = "EMAIL"
    PHONE    = "PHONE"


@dataclass(frozen=True)
class PatternDef:
    entity_type: EntityType
    pattern: re.Pattern
    value_group: int = 0
    priority: int = 0


PATTERN_REGISTRY: list[PatternDef] = [

    # PEM private keys (RSA, EC, DSA, OpenSSH, etc.) — multi-line base64
    # blocks between BEGIN/END markers.  PEM_KEY_8290
    PatternDef(
        EntityType.PEM_KEY,
        re.compile(
            r"-----BEGIN (?:"
            r"RSA PRIVATE KEY|DSA PRIVATE KEY|EC PRIVATE KEY"
            r"|OPENSSH PRIVATE KEY|PRIVATE KEY|ENCRYPTED PRIVATE KEY"
            r"|EC PARAMETERS"
            r")-----"
            r"\s+[A-Za-z0-9+/=\s]+?"
            r"-----END (?:"
            r"RSA PRIVATE KEY|DSA PRIVATE KEY|EC PRIVATE KEY"
            r"|OPENSSH PRIVATE KEY|PRIVATE KEY|ENCRYPTED PRIVATE KEY"
            r"|EC PARAMETERS"
            r")-----",
            re.DOTALL,
        ),
        priority=93,
    ),

    # JSON Web Token — three base64url segments, header always starts with eyJ.
    PatternDef(
        EntityType.JWT,
        re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        priority=95,
    ),

    # OpenAI / Anthropic / generic "sk-" API keys.
    PatternDef(
        EntityType.AI_KEY,
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        priority=90,
    ),

    # GitHub personal access tokens (classic ghp_ format).
    PatternDef(
        EntityType.GH_TOKEN,
        re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
        priority=88,
    ),

    # AWS access key IDs.
    PatternDef(
        EntityType.AWS_KEY,
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        priority=87,
    ),

    # Database connection strings with embedded credentials.
    # e.g. postgres://admin:secret@prod.db.internal:5432/app
    PatternDef(
        EntityType.DB_URL,
        re.compile(
            r"(?:postgresql|postgres|mysql|mongodb|redis|amqp|smtp)s?://"
            r"[^:@\s]*:[^@\s]+@[^\s/\"']+(?:/[^\s\"']*)?",
            re.IGNORECASE,
        ),
        priority=85,
    ),

    # Generic key=value / key:value credential assignments.
    # Redacts only the value portion (group 1); key name is preserved.
    # e.g. "password=hunter2"  →  "password=API_KEY_XXXX"
    PatternDef(
        EntityType.API_KEY,
        re.compile(
            r"(?:password|secret|token|key|api[-_]?key|apikey"
            r"|auth(?:_?token)?|passwd|pwd)\s*[=:]\s*(\S+)",
            re.IGNORECASE,
        ),
        value_group=1,
        priority=80,
    ),

    # US Social Security Numbers — dashes only to avoid false positives on
    # version strings (1.2.3456) and port numbers.
    PatternDef(
        EntityType.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        priority=25,
    ),

    # Email addresses — appear in log lines, comments, test data.
    # Excludes well-known dummy domains (example.com, test.com etc).
    PatternDef(
        EntityType.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        priority=20,
    ),

    # Phone numbers — require clear delimiters to avoid false positives
    # on port numbers, array indices, and version strings.
    PatternDef(
        EntityType.PHONE,
        re.compile(
            r"(?:"
            r"\+\d{1,3}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{1,4}[\s\-.]?\d{1,9}"
            r"|\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}"
            r")"
        ),
        priority=10,
    ),
]

PATTERN_REGISTRY.sort(key=lambda p: p.priority, reverse=True)

# Dummy/placeholder email domains that should never be redacted.
_EXCLUSION_RE = re.compile(
    r"\b\w+@(?:example|test|placeholder|dummy|sample|localhost|foo|bar)\.\w+\b",
    re.IGNORECASE,
)


def is_excluded(value: str) -> bool:
    return bool(_EXCLUSION_RE.fullmatch(value))
