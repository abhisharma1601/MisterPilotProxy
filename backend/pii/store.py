"""
Mapping store: placeholder ↔ original value.

THREAT MODEL:
  This store is the inversion oracle. Anyone with read access to it can
  reverse all pseudonymization. In production:
  - FileStore: use full-disk encryption or replace with EncryptedFileStore
    (wrap content with AES-GCM keyed from the HMAC key).
  - RedisStore: use TLS (rediss://) + AUTH. The Redis keyspace stores
    plaintext original values — treat it as a secrets store.
  - DatabaseStore: apply column-level encryption (pgcrypto, SQLCipher).
  - InMemoryStore: fine for testing; mappings are lost on restart.

  All implementations must be thread-safe. The FileStore uses atomic
  os.replace() writes to prevent corruption under concurrent flushes.
"""
from __future__ import annotations

import abc
import json
import os
import threading
import time
from typing import Optional


class MappingStore(abc.ABC):
    """Abstract mapping store. Implementations must be thread-safe."""

    @abc.abstractmethod
    def get_by_original(self, original: str) -> Optional[str]:
        """Return placeholder for original, or None."""

    @abc.abstractmethod
    def get_by_placeholder(self, placeholder: str) -> Optional[str]:
        """Return original for placeholder, or None."""

    @abc.abstractmethod
    def put(self, placeholder: str, original: str) -> None:
        """Write a bidirectional mapping (idempotent if already present)."""

    @abc.abstractmethod
    def size(self) -> int:
        """Number of stored mappings."""

    def close(self) -> None:
        """Flush and release resources."""


# --------------------------------------------------------------------------- #
# In-memory                                                                    #
# --------------------------------------------------------------------------- #

class InMemoryStore(MappingStore):
    """
    Thread-safe in-memory store. No persistence — mappings lost on restart.
    Use for: unit tests, ephemeral sessions.
    """

    def __init__(self) -> None:
        self._ph2orig: dict[str, str] = {}
        self._orig2ph: dict[str, str] = {}
        self._lock = threading.RLock()

    def get_by_original(self, original: str) -> Optional[str]:
        with self._lock:
            return self._orig2ph.get(original)

    def get_by_placeholder(self, placeholder: str) -> Optional[str]:
        with self._lock:
            return self._ph2orig.get(placeholder)

    def put(self, placeholder: str, original: str) -> None:
        with self._lock:
            self._ph2orig[placeholder] = original
            self._orig2ph[original] = placeholder

    def size(self) -> int:
        with self._lock:
            return len(self._ph2orig)


# --------------------------------------------------------------------------- #
# File-backed                                                                  #
# --------------------------------------------------------------------------- #

class FileStore(MappingStore):
    """
    JSON file store with atomic writes and background periodic flush.

    Write path: dirty flag is set on every put(); a daemon thread flushes
    every _FLUSH_INTERVAL seconds using write-to-temp + os.replace() for
    atomicity. close() forces an immediate flush.

    THREAT MODEL: file is plaintext JSON. Restrict filesystem permissions
    to the proxy process user (chmod 600). For compliance environments,
    subclass and override _serialize/_deserialize to add AES-GCM encryption.
    """

    _FLUSH_INTERVAL = 5.0

    def __init__(self, path: str) -> None:
        self._path = path
        self._ph2orig: dict[str, str] = {}
        self._orig2ph: dict[str, str] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._load()
        self._start_flush_thread()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._ph2orig.update(data.get("ph2orig", {}))
                self._orig2ph.update(data.get("orig2ph", {}))
        except (OSError, json.JSONDecodeError):
            pass

    def _flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            snapshot = {
                "ph2orig": dict(self._ph2orig),
                "orig2ph": dict(self._orig2ph),
            }
            self._dirty = False

        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f)
            os.replace(tmp, self._path)   # atomic on POSIX; near-atomic on Windows
        except OSError:
            pass

    def _start_flush_thread(self) -> None:
        def _loop():
            while True:
                time.sleep(self._FLUSH_INTERVAL)
                self._flush()
        t = threading.Thread(target=_loop, daemon=True, name="pii-store-flush")
        t.start()

    def get_by_original(self, original: str) -> Optional[str]:
        with self._lock:
            return self._orig2ph.get(original)

    def get_by_placeholder(self, placeholder: str) -> Optional[str]:
        with self._lock:
            return self._ph2orig.get(placeholder)

    def put(self, placeholder: str, original: str) -> None:
        with self._lock:
            self._ph2orig[placeholder] = original
            self._orig2ph[original] = placeholder
            self._dirty = True

    def size(self) -> int:
        with self._lock:
            return len(self._ph2orig)

    def close(self) -> None:
        self._flush()


# --------------------------------------------------------------------------- #
# Redis                                                                        #
# --------------------------------------------------------------------------- #

class RedisStore(MappingStore):
    """
    Redis-backed store for multi-process / horizontally-scaled deployments.

    Key schema:
      pii:ph:{placeholder}       → original value (plaintext)
      pii:orig:{sha256(original)} → placeholder
      (original is hashed as a Redis key to avoid storing plaintext in SCAN output)

    THREAT MODEL:
    - Use TLS: rediss://
    - Use AUTH: redis://:<password>@host:port
    - The value side of pii:ph:* keys contains plaintext originals. Apply
      Redis ACLs to restrict key-space access to the proxy process only.
    - Consider encrypting values with AES-GCM before storing.

    Requires: pip install redis
    """

    _PH_PREFIX   = "pii:ph:"
    _ORIG_PREFIX = "pii:orig:"

    def __init__(self, redis_url: str) -> None:
        try:
            import redis as _redis
        except ImportError:
            raise ImportError("Redis backend requires: pip install redis")
        self._r = _redis.from_url(redis_url, decode_responses=True)

    def _orig_key(self, original: str) -> str:
        import hashlib
        return self._ORIG_PREFIX + hashlib.sha256(original.encode()).hexdigest()

    def get_by_original(self, original: str) -> Optional[str]:
        return self._r.get(self._orig_key(original))

    def get_by_placeholder(self, placeholder: str) -> Optional[str]:
        return self._r.get(self._PH_PREFIX + placeholder)

    def put(self, placeholder: str, original: str) -> None:
        pipe = self._r.pipeline()
        pipe.set(self._PH_PREFIX + placeholder, original)
        pipe.set(self._orig_key(original), placeholder)
        pipe.execute()

    def size(self) -> int:
        # Approximate: counts all Redis keys, not just ours.
        # For an accurate count, maintain a separate counter key.
        return self._r.dbsize()


# --------------------------------------------------------------------------- #
# Database (SQLite default, SQLAlchemy-compatible)                             #
# --------------------------------------------------------------------------- #

class DatabaseStore(MappingStore):
    """
    SQL-backed store. Defaults to SQLite (zero extra dependencies).
    Pass a SQLAlchemy-compatible URL for Postgres/MySQL.

    THREAT MODEL: original values are stored in plaintext. Apply column-level
    encryption (pgcrypto, SQLCipher) for PCI DSS / HIPAA compliance.
    The `created_at` timestamp enables TTL-based purging of stale mappings.
    """

    def __init__(self, db_url: str = "sqlite://.pii_map.db") -> None:
        self._db_url = db_url
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._create_schema()

    def _connect(self):
        if "sqlite" in self._db_url:
            import sqlite3
            path = self._db_url.replace("sqlite:///", "").replace("sqlite://", "")
            return sqlite3.connect(path, check_same_thread=False)
        raise NotImplementedError(
            "Non-SQLite databases require a SQLAlchemy subclass. "
            "Override _connect() with your engine."
        )

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS pii_map (
                    placeholder TEXT PRIMARY KEY,
                    original    TEXT NOT NULL,
                    created_at  INTEGER NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orig ON pii_map(original)"
            )
            self._conn.commit()

    def get_by_original(self, original: str) -> Optional[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT placeholder FROM pii_map WHERE original = ?", (original,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_by_placeholder(self, placeholder: str) -> Optional[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT original FROM pii_map WHERE placeholder = ?", (placeholder,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def put(self, placeholder: str, original: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO pii_map(placeholder, original, created_at) "
                "VALUES (?, ?, ?)",
                (placeholder, original, int(time.time())),
            )
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM pii_map")
            return cur.fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def build_store(config) -> MappingStore:
    """Instantiate the correct store from a PseudonymConfig."""
    backend = config.store_backend
    if backend == "memory":
        return InMemoryStore()
    if backend == "file":
        return FileStore(config.store_path)
    if backend == "redis":
        if not config.redis_url:
            raise ValueError("Set PII_REDIS_URL for redis store backend")
        return RedisStore(config.redis_url)
    if backend == "database":
        return DatabaseStore(config.db_url or "sqlite://.pii_map.db")
    raise ValueError(f"Unknown store backend: {backend!r}")
