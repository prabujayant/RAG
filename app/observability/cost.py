"""LLM cost tracking: pricing table + usage aggregation (no external deps).

Token counts are reported by every OpenRouter response, but nothing summed
them — this module closes that gap. :func:`record_llm_usage` is called by the
LLM client on every completion (generation, grounding judge, agent turns);
:func:`usage_summary` feeds ``GET /metrics`` so spend is visible next to
latency. Process-local like the timing registry: resets on restart.

Prices are USD per 1M tokens (prompt, completion) and are approximate —
OpenRouter adjusts them per provider. Unknown models record tokens with
``cost_usd=None`` (unpriced) rather than a made-up number.
"""

from __future__ import annotations

import threading

# (prompt $/1M, completion $/1M). Verified against OpenRouter listings;
# ":free" variants are $0. Revisit when switching default models.
MODEL_PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "nex-agi/nex-n2.5-mini": (0.0, 0.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-3-5-haiku": (0.80, 4.00),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
}


def normalize_model_id(model: str | None) -> str:
    """Lowercase + strip provider routing suffixes (``:free``, ``:nitro``…)."""
    if not model:
        return ""
    base = model.strip().lower()
    for suffix in (":free", ":nitro", ":floor", ":extended"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def estimate_cost_usd(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Return the estimated USD cost, or None when unpriced/unknown."""
    prices = MODEL_PRICES_USD_PER_M.get(normalize_model_id(model))
    if prices is None or prompt_tokens is None or completion_tokens is None:
        return None
    prompt_rate, completion_rate = prices
    return round(prompt_tokens * prompt_rate / 1_000_000 + completion_tokens * completion_rate / 1_000_000, 6)


_lock = threading.Lock()
_totals: dict[str, dict[str, float]] = {}


def _bucket(label: str, model: str | None) -> str:
    return f"{label}::{normalize_model_id(model) or 'unknown'}"


def record_llm_usage(
    label: str,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Aggregate one completion's tokens + cost (thread-safe, never raises).

    Returns the estimated cost so callers can attach it to responses.
    """
    try:
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        key = _bucket(label, model)
        with _lock:
            entry = _totals.setdefault(
                key,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            )
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt_tokens or 0
            entry["completion_tokens"] += completion_tokens or 0
            entry["cost_usd"] += cost or 0.0
        return cost
    except Exception:
        return None


def usage_summary() -> dict:
    """Return per-label/model token + cost totals (JSON-serializable)."""
    with _lock:
        entries = {
            key: {
                "calls": int(val["calls"]),
                "prompt_tokens": int(val["prompt_tokens"]),
                "completion_tokens": int(val["completion_tokens"]),
                "total_tokens": int(val["prompt_tokens"] + val["completion_tokens"]),
                "cost_usd": round(val["cost_usd"], 6),
            }
            for key, val in sorted(_totals.items())
        }
    totals = {
        "calls": sum(e["calls"] for e in entries.values()),
        "prompt_tokens": sum(e["prompt_tokens"] for e in entries.values()),
        "completion_tokens": sum(e["completion_tokens"] for e in entries.values()),
        "cost_usd": round(sum(e["cost_usd"] for e in entries.values()), 6),
    }
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return {"by_usage": entries, "totals": totals}


def reset_usage() -> None:
    """Clear all usage samples (tests)."""
    with _lock:
        _totals.clear()


__all__ = [
    "MODEL_PRICES_USD_PER_M",
    "estimate_cost_usd",
    "normalize_model_id",
    "record_llm_usage",
    "reset_usage",
    "usage_summary",
]
