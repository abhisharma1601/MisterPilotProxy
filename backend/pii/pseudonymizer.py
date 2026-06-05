"""
Placeholder generation via HMAC-SHA256 pseudonymisation.

Format: {TYPE}_{SUFFIX}  e.g. EMAIL_B921  PHONE_3F1A  DB_URL_AC51

Same value always produces the same suffix (deterministic).
Different values with the same suffix get a counter: EMAIL_B921_2.

Entity linking: EMAIL values with the same local part (before @) share
the same HMAC suffix, so john@acme.com and john@other.org → EMAIL_XXXX
and EMAIL_XXXX_1 — the shared prefix signals it's the same person.
"""
from __future__ import annotations

import hashlib
import hmac
import re

from .config import PseudonymConfig
from .patterns import EntityType
from .store import MappingStore

# Matches any generated placeholder: TYPE_HEXSUFFIX or TYPE_HEXSUFFIX_N
_PLACEHOLDER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(e.value) for e in EntityType) + r")_[0-9A-F]{4,}(?:_\d+)?\b"
)


class Pseudonymizer:
    def __init__(self, config: PseudonymConfig, store: MappingStore) -> None:
        self._key   = config.hmac_key
        self._slen  = config.suffix_length
        self._store = store

    def pseudonymize(self, entity_type: EntityType, value: str) -> str:
        # Type-aware lookup: only reuse an existing placeholder if its type
        # prefix matches, so the same raw value under EMAIL vs API_KEY
        # produces distinct placeholders.
        existing = self._store.get_by_original(value)
        if existing and existing.startswith(f"{entity_type.value}_"):
            return existing

        normalized  = self._normalize_for_hmac(entity_type, value)
        suffix      = self._hmac_suffix(normalized)
        placeholder = f"{entity_type.value}_{suffix}"

        collision_original = self._store.get_by_placeholder(placeholder)
        if collision_original is not None and collision_original != value:
            placeholder = self._resolve_collision(placeholder, value)

        self._store.put(placeholder, value)
        return placeholder

    def restore(self, text: str) -> str:
        """Replace all known placeholders in text with their original values."""
        def _replace(m: re.Match) -> str:
            original = self._store.get_by_placeholder(m.group(0))
            return original if original is not None else m.group(0)

        return _PLACEHOLDER_RE.sub(_replace, text)

    def _normalize_for_hmac(self, entity_type: EntityType, value: str) -> str:
        # Entity linking for emails: normalise to local part so that
        # john@company.com and john@other.org share the same suffix.
        if entity_type == EntityType.EMAIL and "@" in value:
            return value.split("@")[0].lower()
        return value

    def _hmac_suffix(self, value: str) -> str:
        h = hmac.new(self._key, value.encode("utf-8"), hashlib.sha256)
        return h.hexdigest()[: self._slen].upper()

    def _resolve_collision(self, base: str, value: str) -> str:
        counter = 1
        while True:
            candidate = f"{base}_{counter}"
            existing  = self._store.get_by_placeholder(candidate)
            if existing is None or existing == value:
                return candidate
            counter += 1
