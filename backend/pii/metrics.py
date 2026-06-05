"""
Thread-safe counters and latency histogram for the redaction pipeline.

Export snapshot() to Prometheus, Datadog, or your observability backend.
In production, replace the list-based histogram with a proper HDR histogram
(pip install hdrh) for accurate high-percentile latency reporting.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class Metrics:
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    requests_total: int = 0
    findings_total: int = 0
    redacted_requests: int = 0
    errors_total: int = 0
    _latencies_ms: list[float] = field(default_factory=list, repr=False)
    _MAX_LATENCY_SAMPLES: int = field(default=2000, repr=False)

    def record_redaction(self, n_findings: int) -> None:
        with self._lock:
            self.requests_total += 1
            self.findings_total += n_findings
            if n_findings > 0:
                self.redacted_requests += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def record_latency_ms(self, ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(ms)
            if len(self._latencies_ms) > self._MAX_LATENCY_SAMPLES:
                # Drop the oldest half to keep memory bounded
                self._latencies_ms = self._latencies_ms[self._MAX_LATENCY_SAMPLES // 2 :]

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager that records wall-clock latency in ms."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_latency_ms((time.perf_counter() - start) * 1000)

    def percentile_ms(self, p: float) -> float:
        """Return the p-th percentile latency (0–100)."""
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            s = sorted(self._latencies_ms)
            idx = max(0, int(len(s) * p / 100) - 1)
            return s[idx]

    def snapshot(self, store_size: int = 0) -> dict:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "findings_total": self.findings_total,
                "redacted_requests": self.redacted_requests,
                "errors_total": self.errors_total,
                "p50_latency_ms": self.percentile_ms(50),
                "p95_latency_ms": self.percentile_ms(95),
                "p99_latency_ms": self.percentile_ms(99),
                "store_size": store_size,
            }
