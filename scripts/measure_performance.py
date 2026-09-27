"""Measure end-to-end query latency, cost, and throughput.

Produces the numbers the README quotes. Everything here is *measured* on the
local stack — nothing is estimated or copied from a vendor datasheet.

What it measures
----------------
- **Latency**: per-query wall time for the full pipeline (retrieval → rerank →
  evidence selection → generation → grounding), reported as p50/p95/p99.
- **Cost**: USD per query, from the token counts OpenRouter returns and the
  pricing table in ``app.observability.cost``.
- **Throughput**: queries per minute, measured by running a fixed number of
  queries sequentially and dividing.

Usage
-----
::

    python scripts/measure_performance.py --questions 20
    python scripts/measure_performance.py --questions 20 --output evals/reports/performance.json

Requires a running stack (Postgres + Qdrant) with an ingested corpus and a
valid ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.evaluation.datasets import load_golden_dataset  # noqa: E402
from app.generation.client import OpenRouterClient  # noqa: E402
from app.generation.service import GenerationService  # noqa: E402
from app.observability import cost  # noqa: E402
from app.retrieval.evidence import EvidenceSelector  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.reranker import Reranker  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure query latency, cost, and throughput.")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions to run.")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results here.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up queries excluded from stats.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    dataset = load_golden_dataset(settings.eval_dataset_path)
    answerable = [ex for ex in dataset.examples if ex.answerable]
    questions = [ex.question for ex in answerable[: args.questions]]
    if not questions:
        print("ERROR: no answerable questions in the golden dataset.", file=sys.stderr)
        return 1

    print(f"Model: {settings.openrouter_model}")
    print(f"Questions: {len(questions)} (+{args.warmup} warm-up)")
    print("Loading models and connecting to stores...")

    retriever = HybridRetriever(settings=settings)
    reranker = Reranker(settings=settings) if settings.enable_reranker else None
    selector = EvidenceSelector(settings=settings)
    service = GenerationService(client=OpenRouterClient(settings=settings))

    cost.reset_usage()

    latencies: list[float] = []
    failures = 0
    total = args.warmup + len(questions)

    for i in range(total):
        question = questions[i % len(questions)]
        is_warmup = i < args.warmup
        start = time.perf_counter()
        try:
            retrieved = retriever.retrieve(question, top_k=settings.hybrid_top_k)
            if reranker is not None:
                retrieved = reranker.rerank(question, retrieved, top_k=settings.rerank_top_k)
            evidence = selector.select(retrieved)
            output = service.generate(question, candidates=evidence)
            elapsed = time.perf_counter() - start
            failed = output.answer.startswith("Generation failed")
            if failed:
                failures += 1
            if not is_warmup:
                latencies.append(elapsed)
            label = "warmup" if is_warmup else f"{i - args.warmup + 1}/{len(questions)}"
            status = "FAIL" if failed else "ok"
            print(f"  [{label}] {elapsed:6.2f}s {status}  {question[:60]}")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            failures += 1
            if not is_warmup:
                latencies.append(elapsed)
            print(f"  [{i}] {elapsed:6.2f}s ERROR {type(exc).__name__}: {str(exc)[:80]}")

    if not latencies:
        print("ERROR: no successful measurements.", file=sys.stderr)
        return 1

    usage = cost.usage_summary()
    totals = usage["totals"]
    n = len(latencies)
    total_seconds = sum(latencies)

    results = {
        "measured_at": datetime.now(UTC).isoformat(),
        "model": settings.openrouter_model,
        "questions_measured": n,
        "failures": failures,
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "throughput_queries_per_minute": round(n / total_seconds * 60, 2) if total_seconds else 0.0,
        "tokens": {
            "prompt_total": totals["prompt_tokens"],
            "completion_total": totals["completion_tokens"],
            "per_query": round(totals["total_tokens"] / n, 1),
        },
        "cost_usd": {
            "total": totals["cost_usd"],
            "per_query": round(totals["cost_usd"] / n, 6),
        },
        "llm_calls": totals["calls"],
    }

    print()
    print("=" * 62)
    print("MEASURED RESULTS")
    print("=" * 62)
    lat = results["latency_seconds"]
    print(f"  Queries measured      : {n} ({failures} failures)")
    print(f"  Latency mean / median : {lat['mean']:.2f}s / {lat['median']:.2f}s")
    print(f"  Latency p95 / p99     : {lat['p95']:.2f}s / {lat['p99']:.2f}s")
    print(f"  Throughput            : {results['throughput_queries_per_minute']:.2f} queries/min")
    print(f"  Tokens per query      : {results['tokens']['per_query']:.0f}")
    print(f"  Cost per query        : ${results['cost_usd']['per_query']:.6f}")
    print(f"  Total cost            : ${results['cost_usd']['total']:.6f}")
    print("=" * 62)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Written to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
