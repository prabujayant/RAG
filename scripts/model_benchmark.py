"""Benchmark cheap OpenRouter models against AskMyDocs' real answer contract.

The app does not just need valid JSON — the grounding pipeline splits the answer
into claims and verifies each against its cited evidence, so the model MUST emit
inline ``[C1]``/``[C2]`` markers inside the answer text. A model that returns a
``citations`` array but no inline markers silently breaks grounding, which a
casual test will miss.

Scores each model on:
  * json_ok     — response parses as the expected JSON object
  * inline      — number of [Cn] markers inside `answer` (must be > 0)
  * claims      — extracted claims array length
  * latency     — seconds
  * cost        — USD for the run's token usage

Usage:
    python scripts/model_benchmark.py                # default shortlist
    python scripts/model_benchmark.py --top 12       # shortlist by price
    python scripts/model_benchmark.py --trials 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx
from dotenv import load_dotenv

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You answer strictly from the provided evidence. "
    'Respond with a single JSON object: {"answer": string, "citations": [string], '
    '"claims": [{"claim": string, "citation_ids": [string]}]}. '
    "Every factual sentence in `answer` must end with an inline marker like [C1] "
    "referring to the evidence item it came from. "
    "If the evidence is insufficient, refuse instead of guessing."
)

# Realistic multi-claim questions with evidence drawn from the project's own docs.
CASES = [
    {
        "question": "What dominates ingestion latency, and how is chunk identity preserved?",
        "evidence": (
            "[C1] The ingestion pipeline runs parsing, cleaning, chunking, "
            "embedding, vector upsert and keyword indexing.\n"
            "[C2] On CPU, embedding dominates: 46,977 ms of a 47.1 s run for a "
            "4-chunk document.\n"
            "[C3] Chunk IDs are deterministic hashes derived from the document id "
            "and chunk index, so they match across Postgres, Qdrant and the "
            "tsvector postings."
        ),
    },
    {
        "question": "How are API keys secured and what happens when they expire?",
        "evidence": (
            "[C1] API keys are prefixed with `amsk_` and are shown only once at "
            "creation time.\n"
            "[C2] Keys expire after 90 days and can be revoked at any time from "
            "the Admin Console.\n"
            "[C3] A key can be scoped to a set of permissions using the "
            "permissions claim; scoping is strongly recommended."
        ),
    },
]


def fetch_models(api_key: str) -> list[dict]:
    r = httpx.get("https://openrouter.ai/api/v1/models", timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def price_of(model: dict) -> tuple[float, float] | None:
    p = model.get("pricing") or {}
    try:
        pin = float(p.get("prompt", "0")) * 1_000_000
        pout = float(p.get("completion", "0")) * 1_000_000
    except (TypeError, ValueError):
        return None
    if pin < 0 or pout < 0:
        return None  # variable-priced router entries
    return pin, pout


def run_case(api_key: str, model_id: str, case: dict) -> dict:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {case['question']}\n\nEvidence:\n{case['evidence']}",
            },
        ],
        "max_tokens": 900,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    start = time.time()
    try:
        r = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "secs": time.time() - start}

    secs = time.time() - start
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}", "secs": secs}

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return {"ok": False, "error": "no choices", "secs": secs}

    content = (choices[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    result = {
        "ok": True,
        "secs": secs,
        "finish": choices[0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "empty": not content.strip(),
    }
    try:
        obj = json.loads(content)
        answer = obj.get("answer") or ""
        markers = sum(answer.count(m) for m in ("[C1]", "[C2]", "[C3]"))
        result.update(
            {
                "json": True,
                "inline": markers,
                "claims": len(obj.get("claims") or []),
                "answer_len": len(answer),
            }
        )
    except Exception:  # noqa: BLE001
        result.update({"json": False, "inline": 0, "claims": 0, "answer_len": 0})
    return result


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark cheap OpenRouter models.")
    parser.add_argument("--top", type=int, default=10, help="How many cheap paid models to test.")
    parser.add_argument("--trials", type=int, default=1, help="Runs per model per case.")
    parser.add_argument(
        "--max-out-price",
        type=float,
        default=0.45,
        help="Max completion price (USD per 1M) to consider 'dead cheap'.",
    )
    parser.add_argument("--only", nargs="*", help="Explicit model ids to test.")
    parser.add_argument(
        "--include-free",
        action="store_true",
        help="Also test free (:free) variants.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return 1

    models = fetch_models(api_key)

    if args.only:
        candidates = [m for m in models if m["id"] in args.only]
    else:
        priced = []
        for m in models:
            pr = price_of(m)
            if pr is None:
                continue
            pin, pout = pr
            if pin == 0 and pout == 0 and not args.include_free:
                continue
            if pout > args.max_out_price:
                continue
            if (m.get("context_length") or 0) < 32_000:
                continue
            priced.append((pout, pin, m["id"], m))
        priced.sort(key=lambda t: (t[0], t[1], t[2]))
        candidates = [t[3] for t in priced[: args.top]]

    print(f"Testing {len(candidates)} model(s), {len(CASES)} case(s), {args.trials} trial(s)\n")

    rows = []
    for model in candidates:
        mid = model["id"]
        pr = price_of(model) or (0.0, 0.0)
        runs = []
        for _ in range(args.trials):
            for case in CASES:
                runs.append(run_case(api_key, mid, case))
        good = [r for r in runs if r.get("ok")]
        total_cost = sum(
            (r.get("prompt_tokens", 0) / 1e6) * pr[0]
            + (r.get("completion_tokens", 0) / 1e6) * pr[1]
            for r in good
        )
        rows.append(
            {
                "model": mid,
                "in": pr[0],
                "out": pr[1],
                "runs": len(runs),
                "ok": len(good),
                "json": sum(1 for r in good if r.get("json")),
                "cited": sum(1 for r in good if (r.get("inline") or 0) > 0),
                "empty": sum(1 for r in good if r.get("empty")),
                "claims": sum(r.get("claims", 0) for r in good),
                "avg_s": round(sum(r["secs"] for r in good) / len(good), 1) if good else None,
                "cost": total_cost,
            }
        )

    hdr = f"{'model':44s} {'$/1M in/out':>14s} {'json':>5s} {'cited':>6s} {'empty':>6s} {'claims':>7s} {'avg s':>6s} {'$/query':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:44s} {r['in']:6.3f}/{r['out']:<7.3f} "
            f"{r['json']:>3d}/{r['runs']:<2d} {r['cited']:>4d}/{r['runs']:<2d} "
            f"{r['empty']:>4d}/{r['runs']:<2d} {r['claims']:>7d} "
            f"{str(r['avg_s']):>6s} {r['cost']/max(1, r['runs']):>9.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
