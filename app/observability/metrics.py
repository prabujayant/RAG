"""Lightweight in-memory stage-timing registry (no external deps).

Records per-stage wall-clock durations so slow uploads/queries can be
attributed (parse vs embed vs index vs rerank vs judge) instead of guessed.
Exposed via GET /metrics. Process-local: resets on restart, which is fine
for a dev/personal deployment.
"""

from __future__ import annotations

import threading
import time

_START = time.monotonic()
_lock = threading.Lock()
_counts: dict[str, int] = {}
_totals_ms: dict[str, float] = {}
_max_ms: dict[str, float] = {}
_samples_ms: dict[str, list[float]] = {}
_counters: dict[str, int] = {}
_LAST_N = 50


def record(stage: str, duration_ms: float) -> None:
    """Record one timing sample for *stage* (thread-safe, never raises)."""
    try:
        with _lock:
            _counts[stage] = _counts.get(stage, 0) + 1
            _totals_ms[stage] = _totals_ms.get(stage, 0.0) + duration_ms
            _max_ms[stage] = max(_max_ms.get(stage, 0.0), duration_ms)
            samples = _samples_ms.setdefault(stage, [])
            samples.append(duration_ms)
            if len(samples) > _LAST_N:
                del samples[0 : len(samples) - _LAST_N]
    except Exception:
        pass


def count(name: str, amount: int = 1) -> None:
    """Increment an event counter (safety blocks, redactions, …). Thread-safe."""
    try:
        with _lock:
            _counters[name] = _counters.get(name, 0) + amount
    except Exception:
        pass


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * pct / 100)))
    return sorted_vals[idx]


def summary() -> dict:
    """Return per-stage count/avg/p95/max/last + uptime (JSON-serializable)."""
    with _lock:
        stages: dict[str, dict] = {}
        for stage, count in _counts.items():
            total = _totals_ms.get(stage, 0.0)
            samples = sorted(_samples_ms.get(stage, []))
            stages[stage] = {
                "count": count,
                "avg_ms": round(total / count, 1) if count else 0.0,
                "p95_ms": round(_percentile(samples, 95), 1),
                "max_ms": round(_max_ms.get(stage, 0.0), 1),
                "last_ms": round(samples[-1], 1) if samples else 0.0,
            }
        return {
            "uptime_seconds": round(time.monotonic() - _START, 1),
            "stages": stages,
            "counters": dict(_counters),
        }


def reset() -> None:
    """Clear all samples (tests)."""
    with _lock:
        _counts.clear()
        _totals_ms.clear()
        _max_ms.clear()
        _samples_ms.clear()
        _counters.clear()


__all__ = ["count", "record", "summary", "reset"]
