"""Repair golden-dataset chunk references against the *stored* chunks.

Why this exists
---------------
``evals/dataset/golden.jsonl`` references evidence as ``slug:index``. The
indices were authored by hand (or against an older chunking configuration), so
a subset point past the end of the document — e.g. ``oauth-guide:6`` when the
markdown rendering produced only 6 chunks (indices 0-5). Those references can
never be retrieved, which deflates recall for reasons unrelated to retrieval
quality.

``scripts/resolve_relevant_chunks.py`` re-derives references from the *markdown*
rendering only, so it cannot repair references whose slug is stored under a
different format. This script instead asks the database for the chunks that were
actually ingested and rewrites each bad reference to the chunk whose text best
matches the question and expected answer.

Every rewrite is printed with before/after so the change is auditable.

Usage
-----
::

    python scripts/repair_chunk_refs.py            # dry run
    python scripts/repair_chunk_refs.py --write    # rewrite golden.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.models import Chunk, Document  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from sqlalchemy import select  # noqa: E402

STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is", "are",
    "what", "which", "how", "does", "do", "whats", "your", "with", "that",
    "this", "from", "be", "by", "as", "it", "at", "can",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _keywords(text: str) -> set[str]:
    return _tokens(text) - STOPWORDS


def _score(chunk_text: str, keywords: set[str]) -> float:
    """Fraction of keywords present in the chunk (0 when nothing overlaps)."""
    chunk_tokens = _tokens(chunk_text)
    if not keywords or not chunk_tokens:
        return 0.0
    return len(keywords & chunk_tokens) / len(keywords)


def _phrase_bonus(chunk_text: str, expected_answer: str) -> float:
    """Reward contiguous phrase matches from the expected answer.

    Bag-of-words overlap alone is too weak: ``403 permission_denied`` shares
    generic tokens ("token", "error") with every chunk of the error reference,
    so the wrong chunk wins. Counting the longest shared word n-gram instead
    surfaces the chunk that literally contains the answer sentence.
    """
    expected = _words(expected_answer)
    chunk = _words(chunk_text)
    if len(expected) < 3 or not chunk:
        return 0.0
    chunk_ngrams: set[str] = set()
    for n in range(2, 9):
        for i in range(len(chunk) - n + 1):
            chunk_ngrams.add(" ".join(chunk[i : i + n]))
    for n in range(8, 1, -1):
        for i in range(len(expected) - n + 1):
            if " ".join(expected[i : i + n]) in chunk_ngrams:
                return float(n) / len(expected)
    return 0.0


def _match_score(chunk_text: str, keywords: set[str], expected_answer: str) -> float:
    """Combined score: phrase containment first, then keyword overlap."""
    phrase = _phrase_bonus(chunk_text, expected_answer)
    return phrase * 10.0 + _score(chunk_text, keywords)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair out-of-range chunk references.")
    parser.add_argument("--write", action="store_true", help="Rewrite golden.jsonl in place")
    parser.add_argument("--dataset", default="evals/dataset/golden.jsonl")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset

    # Load stored chunks grouped by slug (document title).
    by_slug: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with session_scope() as session:
        rows = session.execute(
            select(Document.title, Chunk.chunk_index, Chunk.text)
            .join(Chunk, Chunk.document_id == Document.document_id)
        ).all()
    for title, index, text in rows:
        by_slug[title].append((int(index), text or ""))
    for slug in by_slug:
        by_slug[slug].sort()

    print(f"Loaded chunks for {len(by_slug)} documents from the database.")

    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    repairs: list[str] = []
    unrepaired: list[str] = []

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        refs: list[str] = list(record.get("relevant_chunk_ids", []))
        if not refs:
            updated.append(json.dumps(record, ensure_ascii=False))
            continue

        new_refs: list[str] = []
        for ref in refs:
            slug, sep, index_str = ref.rpartition(":")
            if not sep or slug not in by_slug:
                new_refs.append(ref)
                continue
            indices = {i for i, _ in by_slug[slug]}
            index = int(index_str)
            if index in indices:
                new_refs.append(ref)
                continue

            # Out of range — pick the best-matching stored chunk instead.
            expected = record.get("expected_answer", "")
            keywords = _keywords(expected) | _keywords(record.get("question", ""))
            best = max(
                by_slug[slug],
                key=lambda pair: (_match_score(pair[1], keywords, expected), -pair[0]),
            )
            replacement = f"{slug}:{best[0]}"
            if replacement in new_refs:
                repairs.append(
                    f"  {record['id']}: {ref} -> dropped (duplicate of {replacement})"
                )
                continue
            new_refs.append(replacement)
            repairs.append(
                f"  {record['id']}: {ref} ({index} out of range, max "
                f"{max(indices)}) -> {replacement} (score "
                f"{_match_score(best[1], keywords, expected):.2f})"
            )

        if not new_refs:
            unrepaired.append(f"  {record['id']}: no replacement found for {refs}")
        if new_refs != refs:
            record["relevant_chunk_ids"] = new_refs
        updated.append(json.dumps(record, ensure_ascii=False))

    print()
    print(f"Repairs: {len(repairs)}")
    for r in repairs:
        print(r)
    if unrepaired:
        print()
        print(f"Unrepaired: {len(unrepaired)}")
        for u in unrepaired:
            print(u)

    if args.write:
        dataset_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print()
        print(f"Wrote {dataset_path}")
    else:
        print()
        print("Dry run — re-run with --write to apply.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())