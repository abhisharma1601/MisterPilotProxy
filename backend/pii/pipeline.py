"""
RedactionPipeline: detects and replaces sensitive entities in text.

Single-pass design: all patterns run simultaneously, overlaps resolved by
priority, replacements applied left-to-right in one sweep.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import re
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

        # Second pass: deobfuscated scan to catch encoding / whitespace /
        # comment / homoglyph bypasses that the normal pass missed.
        sanitized, extra = self._deobfuscated_redact(
            sanitized, {f.original for f in findings}
        )
        findings.extend(extra)

        self._metrics.record_redaction(len(findings))
        return sanitized, findings

    # ── deobfuscation pass ────────────────────────────────────────────────

    _HOMOGLYPH_MAP: dict[int, int] = {
        0x0410: 0x41, 0x0412: 0x42, 0x0415: 0x45, 0x041A: 0x4B,
        0x041C: 0x4D, 0x041D: 0x48, 0x041E: 0x4F, 0x0420: 0x50,
        0x0421: 0x43, 0x0422: 0x54, 0x0423: 0x59, 0x0425: 0x58,
        0x0406: 0x49, 0x0408: 0x4A, 0x0405: 0x53, 0x0404: 0x45,
        0xA0:    0x20,  # non-breaking space
    }

    @classmethod
    def _normalize_homoglyphs(cls, text: str) -> str:
        """Map Cyrillic/Unicode lookalikes to their ASCII equivalents."""
        return text.translate(cls._HOMOGLYPH_MAP)

    @classmethod
    def _deobfuscate(cls, text: str) -> str:
        """Normalise text to defeat common obfuscation tricks.

        Handles: HTML entities, URL encoding, C-style comments, whitespace
        runs, emoji / non-ASCII splitters, dash/dot separators, and Cyrillic
        homoglyphs.  Strips ALL non-alphanumeric-noise so split-secret
        bypasses are caught on re-scan.  The output is only used for pattern
        matching — the original text is preserved for display.
        """
        import html as _html
        from urllib.parse import unquote

        text = _html.unescape(unquote(text))
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = cls._normalize_homoglyphs(text)
        # Remove whitespace, emoji, dashes, dots — everything that could be
        # used to split a secret token without changing meaning.
        # Underscores are NOT stripped: they appear in real tokens.
        return re.sub(r"[\s\-.\u00a0\U0001F300-\U0001FFFF\u2600-\u27FF\u2000-\u206F]+",
                      "", text)

    @classmethod
    def _build_flexible_pattern(cls, core: str) -> str:
        """Build a regex that matches `core` allowing HTML entities, URL
        encoding, whitespace, dashes, dots, underscores, C-style comments,
        and emoji between each character — the most common obfuscation tricks."""
        _SEP = (
            r"(?:/\*.*?\*/)?"   # C-style /* comment */ between chars
            r"[\s\-.\u00a0\U0001F300-\U0001FFFF\u2600-\u27FF\u2000-\u206F]*"
        )
        parts: list[str] = []
        for ch in core:
            if ch.isascii() and (ch.isalnum() or ch in "-_"):
                code = ord(ch)
                parts.append(
                    r"(?:" + re.escape(ch)
                    + r"|&#" + str(code) + r";"
                    + r"|&#[Xx]" + format(code, "X") + r";"
                    + r"|%" + format(code, "02X") + r")"
                    + _SEP
                )
            else:
                parts.append(re.escape(ch) + _SEP)
        return "".join(parts)

    def _deobfuscated_redact(
        self, text: str, already: set[str]
    ) -> tuple[str, list[Finding]]:
        """Second-pass scan against deobfuscated text.

        Returns (sanitized_text, extra_findings).  Deobfuscates the input,
        scans for patterns in the clean (whitespace-free, HTML-unescaped,
        etc.) text, then maps findings back into the real text via flexible
        regex substitution that accounts for the original obfuscation.
        """
        cleaned = self._deobfuscate(text)
        if cleaned == text:
            return text, []

        extra: list[Finding] = []
        for pdef in PATTERN_REGISTRY:
            for m in pdef.pattern.finditer(cleaned):
                vg = pdef.value_group
                sensitive = m.group(vg)
                if not sensitive or is_excluded(sensitive):
                    continue
                core = re.sub(r"\s+", "", sensitive)
                if core in already or len(core) < 8:
                    continue
                # Skip pseudonymizer placeholders — they look like
                # ENTITY_TYPE_HEX and could be found by the patterns after
                # whitespace removal concatenates them with adjacent text.
                if self._pseudo.contains_placeholder(core):
                    continue
                # Verify core exists in the *deobfuscated* original
                if core not in cleaned:
                    continue

                already.add(core)
                placeholder = self._pseudo.pseudonymize(pdef.entity_type, core)

                # Replace in the real text using a flexible pattern that
                # accounts for HTML entities, URL encoding, C comments,
                # whitespace, dashes, dots, and emoji between every character.
                pattern = self._build_flexible_pattern(core)
                text, n = re.subn(
                    pattern, placeholder, text, count=1, flags=re.DOTALL,
                )
                if n:
                    extra.append(Finding(
                        entity_type=pdef.entity_type.value,
                        original=core[:80] + ("..." if len(core) > 80 else ""),
                        placeholder=placeholder,
                        context="[deobfuscated]",
                    ))

        return text, extra

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
