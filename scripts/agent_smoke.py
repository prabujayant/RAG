"""Live smoke test for the agentic tool-use path (needs infra + OpenRouter key).

Runs :func:`run_agent` against the real stack — Postgres, Qdrant, and
nex-n2.5-mini via OpenRouter — and asserts the model actually performed
function calls (not a plain completion) and returned cited evidence.

Usage:
    python scripts/agent_smoke.py [--title-substr PROPOSAL] [--question "..."]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.db.models import Document  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.generation.agent import run_agent  # noqa: E402
from sqlalchemy import select  # noqa: E402


def find_document_id(title_substr: str) -> tuple[str, str]:
    """Return (document_id, title) of the newest doc whose title matches."""
    with SessionLocal() as session:
        rows = session.execute(select(Document).order_by(Document.created_at.desc())).scalars().all()
    for doc in rows:
        if title_substr.lower() in (doc.title or "").lower():
            return doc.document_id, doc.title
    raise SystemExit(f"No document with {title_substr!r} in its title.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live agent tool-use smoke test")
    parser.add_argument("--title-substr", default="Proposal")
    parser.add_argument(
        "--question",
        default="What is this document about? Summarize its main topic.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        print("FAIL: OPENROUTER_API_KEY is not set.")
        return 2
    print(f"Model: {settings.openrouter_model}")

    document_id, title = find_document_id(args.title_substr)
    print(f"Document: {title} ({document_id})")
    print(f"Question: {args.question}")
    print("-" * 70)

    result = run_agent(args.question, document_ids=[document_id], top_k=5, max_steps=6)

    for i, step in enumerate(result.steps, 1):
        status = "ok" if step.ok else "ERROR"
        print(f"[step {i}] {step.tool_name} {status} ({step.latency_ms}ms)")
        print(f"  args: {step.arguments}")
        print(f"  output: {step.output_preview[:200]}")
    print("-" * 70)
    print(f"finish_reason: {result.finish_reason}")
    print(f"evidence chunks: {len(result.evidence)}")
    print(f"answer: {result.answer[:800]}")
    print("-" * 70)

    failures = []
    if not result.steps:
        failures.append("model made zero tool calls (plain completion, not tool use)")
    if not any(s.tool_name == "search_documents" and s.ok for s in result.steps):
        failures.append("no successful search_documents call in the trace")
    if not result.evidence:
        failures.append("no evidence chunks retrieved")
    if not result.answer.strip():
        failures.append("empty final answer")

    if failures:
        print("SMOKE RESULT: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"SMOKE RESULT: PASS "
        f"({len(result.steps)} tool calls, "
        f"{len(result.evidence)} evidence chunks, "
        f"{result.latency_ms / 1000:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
