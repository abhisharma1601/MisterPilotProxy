"""
RedactionPipeline: detects and replaces sensitive entities in text.

Single-pass design: all patterns run simultaneously, overlaps resolved by
priority, replacements applied left-to-right in one sweep.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import PseudonymConfig
from .metrics import Metrics
from .patterns import PATTERN_REGISTRY, PatternDef, is_excluded
from .pseudonymizer import Pseudonymizer
from .store import MappingStore, build_store

log = logging.getLogger("pii.pipeline")


@dataclass
class Finding:
    entity_type: str
    original: str
    placeholder: str
    context: str


@dataclass
class _Match:
    start: int
    end: int
    original: str
    full_text: str
    val_start: int
    val_end: int
    pdef: PatternDef


class RedactionPipeline:

    def __init__(
        self,
        config: PseudonymConfig,
        store: MappingStore | None = None,
    ) -> None:
        self._config = config
        self._store = store or build_store(config)
        self._pseudo = Pseudonymizer(config, self._store)
        self._metrics = Metrics()
        self._ctx = config.log_context_chars

    def redact(self, text: str) -> tuple[str, list[Finding]]:
        """Scan text, replace sensitive entities with placeholders."""
        if not text:
            return text, []

        byte_len = len(text.encode("utf-8"))
        if byte_len > self._config.max_input_bytes:
            log.warning(
                "Input %d bytes exceeds max_input_bytes %d — scanning truncated",
                byte_len, self._config.max_input_bytes,
            )

        with self._metrics.measure():
            raw = self._collect(text)
            resolved = self._resolve_overlaps(raw)
            sanitized, findings = self._apply(text, resolved)

        self._metrics.record_redaction(len(findings))
        return sanitized, findings

    def restore(self, text: str) -> str:
        """Replace all known placeholders in text with their original values."""
        return self._pseudo.restore(text)

    def redact_json(self, payload: Any) -> tuple[Any, list[Finding]]:
        """Recursively redact all string values in a JSON-compatible structure."""
        findings: list[Finding] = []
        result = self._redact_value(payload, findings)
        return result, findings

    def restore_json(self, payload: Any) -> Any:
        """Recursively restore all placeholders in a JSON-compatible structure."""
        return self._restore_value(payload)

    def _redact_value(self, value: Any, findings: list[Finding]) -> Any:
        if isinstance(value, str):
            sanitized, found = self.redact(value)
            findings.extend(found)
            return sanitized
        if isinstance(value, dict):
            return {k: self._redact_value(v, findings) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(item, findings) for item in value]
        return value

    def _restore_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._pseudo.restore(value)
        if isinstance(value, dict):
            return {k: self._restore_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._restore_value(item) for item in value]
        return value

    @property
    def store(self) -> MappingStore:
        return self._store

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    def close(self) -> None:
        self._store.close()

    def _collect(self, text: str) -> list[_Match]:
        matches: list[_Match] = []
        for pdef in PATTERN_REGISTRY:
            for m in pdef.pattern.finditer(text):
                vg = pdef.value_group
                sensitive = m.group(vg)
                if not sensitive or is_excluded(sensitive):
                    continue
                val_start = m.start(vg) - m.start()
                val_end   = m.end(vg)   - m.start()
                matches.append(_Match(
                    start=m.start(), end=m.end(),
                    original=sensitive, full_text=m.group(0),
                    val_start=val_start, val_end=val_end,
                    pdef=pdef,
                ))
        return matches

    def _resolve_overlaps(self, matches: list[_Match]) -> list[_Match]:
        if not matches:
            return []
        by_priority = sorted(
            matches,
            key=lambda m: (-m.pdef.priority, -(m.end - m.start), m.start),
        )
        accepted: list[_Match] = []
        covered: list[tuple[int, int]] = []
        for m in by_priority:
            if not any(m.start < e and m.end > s for s, e in covered):
                accepted.append(m)
                covered.append((m.start, m.end))
        accepted.sort(key=lambda m: m.start)
        return accepted

    def _apply(self, text: str, matches: list[_Match]) -> tuple[str, list[Finding]]:
        findings: list[Finding] = []
        parts: list[str] = []
        cursor = 0

        for m in matches:
            ctx_s = max(0, m.start - self._ctx)
            ctx_e = min(len(text), m.end + self._ctx)
            before = text[ctx_s : m.start].replace("\n", " ")
            after  = text[m.end : ctx_e].replace("\n", " ")
            trunc  = m.original[:80] + ("..." if len(m.original) > 80 else "")

            placeholder = self._pseudo.pseudonymize(m.pdef.entity_type, m.original)

            if m.pdef.value_group != 0:
                replacement = (
                    m.full_text[: m.val_start]
                    + placeholder
                    + m.full_text[m.val_end :]
                )
            else:
                replacement = placeholder

            parts.append(text[cursor : m.start])
            parts.append(replacement)
            cursor = m.end

            findings.append(Finding(
                entity_type=m.pdef.entity_type.value,
                original=trunc,
                placeholder=placeholder,
                context=f"...{before}[{trunc}]{after}...",
            ))

        parts.append(text[cursor:])
        return "".join(parts), findings
