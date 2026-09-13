"""Does LLM_REASONING_EFFORT=none reduce answer quality?

Isolates the one variable that matters: given IDENTICAL evidence, does the model
produce a less correct answer with reasoning disabled?

Each case defines the required facts a correct answer must state. A case scores
1.0 if every required fact appears and no forbidden (hallucinated) fact appears,
otherwise a partial score. Run both settings, compare mean accuracy.

Usage:
    python scripts/reasoning_quality_test.py
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = (
    "You answer strictly from the provided evidence. Respond with a single JSON "
    'object: {"answer": string, "citations": [string], "claims": '
    '[{"claim": string, "citation_ids": [string]}]}. '
    "Every factual sentence in `answer` must end with an inline marker like [C1]. "
    "If the evidence is insufficient, refuse instead of guessing."
)

# `must` = facts a correct answer contains; `forbidden` = claims NOT supported by
# the evidence (appearing means the model hallucinated).
CASES = [
    {
        "q": "How long do API keys last and what happens at expiry?",
        "evidence": "[C1] API keys are prefixed with amsk_ and shown only once.\n"
        "[C2] Keys expire after 90 days and can be revoked from the Admin Console.",
        "must": ["90 day", "revoke"],
        "forbidden": ["never expire", "365 day", "one year"],
    },
    {
        "q": "What dominates ingestion latency on CPU?",
        "evidence": "[C1] Pipeline stages: parse, clean, chunk, embed, upsert, keyword index.\n"
        "[C2] Embedding took 46,977 ms of a 47.1 s run.",
        "must": ["embed"],
        "forbidden": ["parsing dominates", "chunking dominates", "qdrant dominates"],
    },
    {
        "q": "How is chunk identity preserved across stores?",
        "evidence": "[C1] Chunk IDs are deterministic hashes of the document id and "
        "chunk index, so they match across Postgres, Qdrant and tsvector.",
        "must": ["deterministic", "hash"],
        "forbidden": ["random", "uuid"],
    },
    {
        "q": "What does the safety question gate block?",
        "evidence": "[C1] The deterministic question gate refuses disallowed requests "
        "(explosives, malware, hacking how-tos) and instruction-takeover phrasing, "
        "before any retrieval or model spend.",
        "must": ["malware", "refus"],
        "forbidden": ["calls the llm first", "always allows"],
    },
    {
        "q": "Why does the app refuse instead of answering when evidence is missing?",
        "evidence": "[C1] The answer contract requires every claim to carry a citation "
        "the judge can verify; unsupported claims make the status ungrounded, so the "
        "system refuses rather than guessing.",
        "must": ["citation", "refus"],
        "forbidden": ["always answers", "guesses"],
    },
    {
        "q": "What is the final context budget and why is it bounded?",
        "evidence": "[C1] FINAL_CONTEXT_K selects 5 chunks and MAX_CONTEXT_TOKENS caps "
        "the prompt at 3000 tokens so generation cost and latency stay bounded.",
        "must": ["5 chunk", "3000"],
        "forbidden": ["10 chunk", "unbounded"],
    },
]


def score(answer: str, case: dict) -> float:
    low = answer.lower()
    hits = sum(1 for m in case["must"] if m.lower() in low)
    penalties = sum(1 for f in case["forbidden"] if f.lower() in low)
    base = hits / len(case["must"])
    return max(0.0, base - 0.5 * penalties)


def run(api_key: str, model: str, effort: str, case: dict) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {case['q']}\n\nEvidence:\n{case['evidence']}"},
        ],
        "max_tokens": 900,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    if effort != "default":
        payload["reasoning"] = {"effort": effort}
    start = time.time()
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    secs = time.time() - start
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "score": 0.0, "secs": secs}
    d = r.json()
    ch = (d.get("choices") or [{}])[0]
    content = (ch.get("message") or {}).get("content") or ""
    usage = d.get("usage") or {}
    det = usage.get("completion_tokens_details") or {}
    try:
        obj = json.loads(content)
        answer = obj.get("answer") or ""
        s = score(answer, case)
        return {
            "score": s,
            "secs": secs,
            "reasoning": det.get("reasoning_tokens"),
            "completion": usage.get("completion_tokens"),
            "markers": len(re.findall(r"\[C\d\]", answer)),
            "answer": answer[:120],
        }
    except Exception:  # noqa: BLE001
        return {"score": 0.0, "secs": secs, "error": "not json", "answer": content[:80]}


def main() -> int:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-mini:free")
    if not key:
        print("OPENROUTER_API_KEY not set")
        return 1

    print(f"model: {model}\n")
    summary = {}
    for effort in ("high", "none"):
        scores, secs, reas, marks = [], [], [], []
        print(f"=== reasoning effort: {effort} ===")
        for case in CASES:
            r = run(key, model, effort, case)
            scores.append(r.get("score", 0.0))
            secs.append(r.get("secs", 0))
            if r.get("reasoning") is not None:
                reas.append(r["reasoning"])
            marks.append(r.get("markers", 0))
            flag = "OK " if r.get("score", 0) >= 1.0 else "PARTIAL"
            print(f"  [{flag}] score={r.get('score',0):.2f} {r.get('secs',0):5.1f}s "
                  f"reasoning={r.get('reasoning')} markers={r.get('markers')} :: {str(r.get('answer'))[:70]}")
        summary[effort] = {
            "mean": sum(scores) / len(scores),
            "perfect": sum(1 for s in scores if s >= 1.0),
            "avg_s": sum(secs) / len(secs),
            "avg_reasoning": (sum(reas) / len(reas)) if reas else 0,
            "avg_markers": sum(marks) / len(marks),
        }
        print()

    print("=== SUMMARY ===")
    print(f"{'effort':8s} {'mean acc':>9s} {'perfect':>8s} {'avg s':>7s} {'reasoning tok':>14s} {'markers':>8s}")
    for eff, s in summary.items():
        print(f"{eff:8s} {s['mean']:9.2f} {s['perfect']:>5d}/6 {s['avg_s']:7.1f} "
              f"{s['avg_reasoning']:14.0f} {s['avg_markers']:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
