"""
Sensitive entity detection patterns.

Covers every major credential / secret format that realistically appears in
code comments, log pastes, config snippets, and casual LLM prompts.

Priority ladder (higher number wins on overlap):
  JWT 95 > PEM_KEY 93 > AI_KEY 92 > GH_TOKEN 91 > AWS_KEY 89
  > STRIPE_KEY 88 > DB_URL 85 > SLACK_TOKEN 84 > GCP_KEY 83
  > HTTP_AUTH 82 > API_KEY 80 > GITLAB_TOKEN 79 > SERVICE_TOKEN 76
  > GENERIC_TOKEN 72 > OAUTH_TOKEN 70
  > SSN 25 > EMAIL 20 > PHONE 10
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    PEM_KEY       = "PEM_KEY"
    JWT           = "JWT"
    AI_KEY        = "AI_KEY"
    GH_TOKEN      = "GH_TOKEN"
    AWS_KEY       = "AWS_KEY"
    STRIPE_KEY    = "STRIPE_KEY"
    DB_URL        = "DB_URL"
    SLACK_TOKEN   = "SLACK_TOKEN"
    GCP_KEY       = "GCP_KEY"
    HTTP_AUTH     = "HTTP_AUTH"
    API_KEY       = "API_KEY"
    GITLAB_TOKEN  = "GITLAB_TOKEN"
    SERVICE_TOKEN = "SERVICE_TOKEN"
    GENERIC_TOKEN = "GENERIC_TOKEN"
    OAUTH_TOKEN   = "OAUTH_TOKEN"
    SSN           = "SSN"
    EMAIL         = "EMAIL"
    PHONE         = "PHONE"


@dataclass(frozen=True)
class PatternDef:
    entity_type: EntityType
    pattern: re.Pattern
    value_group: int = 0   # 0 = redact the whole match; N = redact only group N
    priority: int = 0


PATTERN_REGISTRY: list[PatternDef] = [

    # ── JSON Web Token ────────────────────────────────────────────────────────
    # Three base64url segments; header always starts with eyJ.
    PatternDef(
        EntityType.JWT,
        re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        priority=95,
    ),

    # ── PEM private keys (RSA, EC, DSA, OpenSSH) ─────────────────────────────
    # Matches both real newlines and literal \n (as stored in .env files).
    PatternDef(
        EntityType.PEM_KEY,
        re.compile(
            r"-----BEGIN (?:"
            r"RSA PRIVATE KEY|DSA PRIVATE KEY|EC PRIVATE KEY"
            r"|OPENSSH PRIVATE KEY|PRIVATE KEY|ENCRYPTED PRIVATE KEY"
            r"|EC PARAMETERS"
            r")-----"
            r".+?"
            r"-----END (?:"
            r"RSA PRIVATE KEY|DSA PRIVATE KEY|EC PRIVATE KEY"
            r"|OPENSSH PRIVATE KEY|PRIVATE KEY|ENCRYPTED PRIVATE KEY"
            r"|EC PARAMETERS"
            r")-----",
            re.DOTALL,
        ),
        priority=93,
    ),

    # ── AI / LLM service keys ─────────────────────────────────────────────────
    # OpenAI: sk-<20+>
    # Anthropic: sk-ant-api03-<93+>
    # OpenRouter: sk-or-v1-<64+>
    # Mistral / Together / Perplexity / etc. share the sk- prefix
    PatternDef(
        EntityType.AI_KEY,
        re.compile(r"\bsk-(?:ant-api\d+-|or-v\d+-|proj-)?[A-Za-z0-9_\-]{20,}\b"),
        priority=92,
    ),
    # HuggingFace: hf_<20+>
    PatternDef(
        EntityType.AI_KEY,
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        priority=92,
    ),
    # Cohere: API keys have a specific 40-char base64url look, but also served
    # under GENERIC_TOKEN; add explicit prefix just in case they add one later.
    # Replicate: r8_<40+>
    PatternDef(
        EntityType.AI_KEY,
        re.compile(r"\br8_[A-Za-z0-9]{40,}\b"),
        priority=92,
    ),
    # Stability AI: sk-<base64-like>
    # (already caught by the sk- pattern above)

    # ── GitHub tokens ──────────────────────────────────────────────────────────
    # Classic PAT (ghp_), OAuth (gho_), Server (ghs_), Refresh (ghr_)
    PatternDef(
        EntityType.GH_TOKEN,
        re.compile(r"\bgh[posr]_[A-Za-z0-9]{36,}\b"),
        priority=91,
    ),
    # Fine-grained PAT: github_pat_<22>_<59>
    PatternDef(
        EntityType.GH_TOKEN,
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22}_[A-Za-z0-9_]{59}\b"),
        priority=91,
    ),

    # ── AWS credentials ────────────────────────────────────────────────────────
    # All known AWS IAM key ID prefixes (AKIA, AGPA, AIPA, ANPA, ANVA, ASIA, …)
    PatternDef(
        EntityType.AWS_KEY,
        re.compile(
            r"\b(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b"
        ),
        priority=89,
    ),

    # ── Stripe keys ────────────────────────────────────────────────────────────
    # Secret (sk_), publishable (pk_), restricted (rk_); live + test; webhook secret
    PatternDef(
        EntityType.STRIPE_KEY,
        re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[0-9A-Za-z]{20,}\b"),
        priority=88,
    ),
    PatternDef(
        EntityType.STRIPE_KEY,
        re.compile(r"\bwhsec_[0-9A-Za-z]{32,}\b"),
        priority=88,
    ),

    # ── Database connection strings ────────────────────────────────────────────
    # Matches any driver://user:pass@host/db with embedded credentials.
    PatternDef(
        EntityType.DB_URL,
        re.compile(
            r"(?:postgresql|postgres|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss"
            r"|amqp|amqps|smtp|smtps|mssql|sqlserver|oracle|clickhouse|cassandra"
            r"|elasticsearch|couchdb|neo4j)s?://"
            r'[^:@\s]*:[^@\s]+@[^\s/"]+(?:/[^\s"]*)?',
            re.IGNORECASE,
        ),
        priority=85,
    ),

    # ── Slack tokens ──────────────────────────────────────────────────────────
    # xoxb (bot), xoxp (user), xoxa (app-level), xoxr (refresh), xoxs (service)
    PatternDef(
        EntityType.SLACK_TOKEN,
        re.compile(r"\bxox[baprs](-[0-9A-Za-z]{10,}){2,}\b"),
        priority=84,
    ),

    # ── Google Cloud / GCP ────────────────────────────────────────────────────
    # Google browser/server API key: AIzaSy + 33 chars
    PatternDef(
        EntityType.GCP_KEY,
        re.compile(r"\bAIzaSy[0-9A-Za-z_\-]{33}\b"),
        priority=83,
    ),

    # ── HTTP Authorization / API-key headers ──────────────────────────────────
    # Matches the credential value after the scheme keyword so the
    # scheme label is preserved: "Authorization: Bearer <REDACTED>"
    PatternDef(
        EntityType.HTTP_AUTH,
        re.compile(
            r"(?:Authorization|X-Api-Key|X-Auth-Token|X-Access-Token"
            r"|X-Amz-Security-Token|X-Service-Token|X-Token|Api-Key)\s*:\s*"
            r"(?:(?:Bearer|Token|Basic|Digest|API-?Key|ApiKey)\s+)?(\S{8,})",
            re.IGNORECASE,
        ),
        value_group=1,
        priority=82,
    ),

    # ── Generic key = value / key : value credential assignments ──────────────
    # Redacts only the value (group 1); key name is preserved in output.
    # e.g.  "password=hunter2"  →  "password=API_KEY_xxxx"
    PatternDef(
        EntityType.API_KEY,
        re.compile(
            r"(?:password|secret|token|key|api[-_]?key|apikey"
            r"|auth(?:orization)?(?:[-_]?token)?|passwd|pwd|passphrase"
            r"|private[-_]?key|access[-_]?key|client[-_]?secret"
            r"|app[-_]?secret|signing[-_]?key|encryption[-_]?key"
            r"|webhook[-_]?secret|deploy[-_]?key|service[-_]?account"
            r")\s*[=:]\s*(\S+)",
            re.IGNORECASE,
        ),
        value_group=1,
        priority=80,
    ),

    # ── GitLab tokens ─────────────────────────────────────────────────────────
    PatternDef(
        EntityType.GITLAB_TOKEN,
        re.compile(r"\bglpat-[0-9A-Za-z_\-]{20,}\b"),   # personal access token
        priority=79,
    ),
    PatternDef(
        EntityType.GITLAB_TOKEN,
        re.compile(r"\bgloas-[0-9A-Za-z_\-]{20,}\b"),   # OAuth application secret
        priority=79,
    ),
    PatternDef(
        EntityType.GITLAB_TOKEN,
        re.compile(r"\bgldeptoken[0-9A-Za-z_\-]{20,}\b"),  # deploy token (legacy)
        priority=79,
    ),
    PatternDef(
        EntityType.GITLAB_TOKEN,
        re.compile(r"\bgldt-[0-9A-Za-z_\-]{20,}\b"),    # deploy token (new format)
        priority=79,
    ),

    # ── Service-specific tokens ────────────────────────────────────────────────

    # npm registry token
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bnpm_[A-Za-z0-9]{36,}\b"),
        priority=76,
    ),
    # PyPI API token: pypi-AgE + long base64url
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bpypi-AgE[A-Za-z0-9_\-]{50,}\b"),
        priority=76,
    ),
    # SendGrid: SG.<22>.<43>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bSG\.[0-9A-Za-z_\-]{22}\.[0-9A-Za-z_\-]{43}\b"),
        priority=76,
    ),
    # Twilio Account SID: AC + 32 hex chars
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bAC[0-9a-fA-F]{32}\b"),
        priority=76,
    ),
    # Mailgun: key-<32 alnum>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bkey-[0-9a-zA-Z]{32}\b"),
        priority=76,
    ),
    # Mailchimp: <32 hex>-us<1-2 digits>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\b[0-9a-f]{32}-us[0-9]{1,2}\b"),
        priority=76,
    ),
    # Shopify access tokens / secrets
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bshp(?:at|ss|ca|pa|uat)_[0-9A-Fa-f]{32,}\b"),
        priority=76,
    ),
    # Telegram bot token: <8-10 digits>:<35 alnum+_->
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\b[0-9]{8,10}:[A-Za-z0-9_\-]{35}\b"),
        priority=76,
    ),
    # Discord bot token: <18-26 base64url>.<6 chars>.<27-40 chars>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\b[A-Za-z0-9_\-]{18,26}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,40}\b"),
        priority=76,
    ),
    # Docker Hub personal access token: dckr_pat_<43>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bdckr_pat_[A-Za-z0-9_\-]{43,}\b"),
        priority=76,
    ),
    # DigitalOcean personal access token: dop_v1_<64 hex>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bdop_v1_[0-9a-f]{64}\b"),
        priority=76,
    ),
    # Doppler service token: dp.st.<long base62>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bdp\.st\.[A-Za-z0-9_\-]{40,}\b"),
        priority=76,
    ),
    # Airtable personal access token: pat<14>.<64>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\bpat[A-Za-z0-9]{14}\.[A-Za-z0-9]{64}\b"),
        priority=76,
    ),
    # Linear API key: lin_api_<40+>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\blin_api_[A-Za-z0-9_\-]{40,}\b"),
        priority=76,
    ),
    # Firebase config snippet: 1:<project-number>:web:<hex>
    PatternDef(
        EntityType.SERVICE_TOKEN,
        re.compile(r"\b1:[0-9]{12}:web:[0-9a-f]{16,}\b"),
        priority=76,
    ),
    # Cloudflare API token / global key: often 40-char base62 (caught by GENERIC_TOKEN)
    # Vercel token: often 24-char hex (caught by GENERIC_TOKEN)
    # Netlify PAT: long alnum (caught by GENERIC_TOKEN)

    # ── Generic bare token ─────────────────────────────────────────────────────
    # Standalone alphanumeric string ≥32 chars that contains at least one
    # uppercase, one lowercase, and one digit — the fingerprint of a randomly
    # generated credential.  Delimiter-bounded to avoid matching long identifiers.
    PatternDef(
        EntityType.GENERIC_TOKEN,
        re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?=[A-Za-z0-9]*[A-Z])"   # at least one uppercase
            r"(?=[A-Za-z0-9]*[a-z])"   # at least one lowercase
            r"(?=[A-Za-z0-9]*[0-9])"   # at least one digit
            r"[A-Za-z0-9]{32,128}"
            r"(?![A-Za-z0-9])"
        ),
        priority=72,
    ),

    # ── Google OAuth access tokens ─────────────────────────────────────────────
    PatternDef(
        EntityType.OAUTH_TOKEN,
        re.compile(r"\bya29\.[A-Za-z0-9_\-]{40,}\b"),
        priority=70,
    ),

    # ── US Social Security Numbers ─────────────────────────────────────────────
    # Require dashes (xxx-xx-xxxx) to avoid false positives on version strings.
    PatternDef(
        EntityType.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        priority=25,
    ),

    # ── Email addresses ────────────────────────────────────────────────────────
    PatternDef(
        EntityType.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        priority=20,
    ),

    # ── Phone numbers ──────────────────────────────────────────────────────────
    # Require clear delimiters to avoid false positives on port/version numbers.
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

# ── Exclusion list for dummy / placeholder emails ─────────────────────────────
_EXCLUSION_RE = re.compile(
    r"\b\w+@(?:example|test|placeholder|dummy|sample|localhost|foo|bar)\.\w+\b",
    re.IGNORECASE,
)


def is_excluded(match_text: str) -> bool:
    """Return True if the match should be excluded (e.g. dummy emails)."""
    return bool(_EXCLUSION_RE.search(match_text))
