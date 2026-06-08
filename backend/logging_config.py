"""
Structured logging setup for MisterPilot backend.

Mirrors the DeepSeekProxy log format:
  YYYY-MM-DD HH:MM:SS [LEVEL] message

Call setup_logging() once at startup (main.py lifespan).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .pii.pipeline import Finding

_LOG_FILE = Path(__file__).parent / "misterpilot.log"

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with console + rotating file handlers."""
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    console = logging.StreamHandler()
    console.setFormatter(_fmt)
    root.addHandler(console)

    path = log_file or os.environ.get("LOG_FILE") or str(_LOG_FILE)
    try:
        file_h = logging.FileHandler(path, encoding="utf-8")
        file_h.setFormatter(_fmt)
        root.addHandler(file_h)
    except OSError:
        logging.getLogger("logging_config").warning(
            "Cannot open log file %s — file logging disabled", path
        )


def log_pii_findings(log: logging.Logger, route: str, findings: list[Finding]) -> None:
    """Log PII redaction summary — counts and entity types only, never the values."""
    type_counts: dict[str, int] = {}
    for f in findings:
        type_counts[f.entity_type] = type_counts.get(f.entity_type, 0) + 1
    summary = ", ".join(f"{t}:{c}" for t, c in sorted(type_counts.items()))
    log.warning("PII redacted from %s: %d finding(s) [%s]", route, len(findings), summary)
