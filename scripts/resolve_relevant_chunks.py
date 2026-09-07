"""Resolve golden dataset chunk references against the real chunker output.

The golden dataset was authored with *approximate* chunk indices. This script:

1. Loads every corpus markdown document.
2. Runs the real paragraph-aware chunker (``app.ingestion.chunker``).
3. Rebuilds ``relevant_chunk_ids`` per question by matching the expected
   answer's key phrases against the chunk texts (best-match chunk).
4. Rewrites ``evals/dataset/golden.jsonl`` in place.

This keeps the dataset truthful: every referenced chunk actually contains
the evidence that answers the question.

Usage:
    python scripts/resolve_relevant_chunks.py [--write]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.chunker import Chunker, ChunkingStrategy  # noqa: E402

QUESTION_CHUNK_OVERRIDES: dict[str, list[str]] = {}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _ngram_keywords(text: str, n: int = 3) -> list[str]:
    """Extract salient n-gram phrases for matching."""
    tokens = _tokenize(text)
    # Remove very common filler words for matching robustness.
    stop = {
        "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is",
        "are", "what", "which", "how", "does", "do", "whats", "what's", "your",
    }
    tokens = tokens - stop
    return list(tokens)


def _match_score(chunk_text: str, keywords: list[str]) -> float:
    """Jaccard-like overlap between chunk tokens and question keywords."""
    chunk_tokens = _tokenize(chunk_text)
    if not keywords or not chunk_tokens:
        return 0.0
    hit = sum(1 for k in keywords if k in chunk_tokens)
    return hit / len(keywords)


def resolve_for_document(
    doc_id: str,
    chunks: list,
    question: str,
    expected_answer: str,
) -> str | None:
    """Return the chunk id whose text best matches the expected answer."""
    answer_keywords = _ngram_keywords(expected_answer or question)

    candidates = chunks
    # Prefer chunks that contain answer-specific token matches.
    scored = [
        (chunk, _match_score(chunk.text, answer_keywords) + _match_score(chunk.text, _ngram_keywords(question)))
        for chunk in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    best_chunk, best_score = scored[0]
    if best_score <= 0.0:
        return None
    return best_chunk.chunk_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve golden dataset chunk references.")
    parser.add_argument("--write", action="store_true", help="Rewrite golden.jsonl in place")
    args = parser.parse_args()

    corpus_dir = ROOT / "data" / "corpus" / "markdown"
    dataset_path = ROOT / "evals" / "dataset" / "golden.jsonl"

    chunker = Chunker(strategy=ChunkingStrategy.PARAGRAPH, chunk_size=512, chunk_overlap=64)

    # Chunk every markdown document once.
    doc_chunks: dict[str, list] = {}
    for md_path in sorted(corpus_dir.glob("*.md")):
        doc_id = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        chunks = chunker.chunk_text(
            text=text,
            document_id=doc_id,
            document_name=doc_id,
            source=f"markdown/{md_path.name}",
        )
        doc_chunks[doc_id] = chunks

    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    changes = 0
    unresolved: list[str] = []

    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("difficulty") == "unanswerable":
            updated.append(line)
            continue
        new_refs: list[str] = []
        for ref in record.get("relevant_chunk_ids", []):
            doc_id, _, _ = ref.rpartition(":")
            chunks = doc_chunks.get(doc_id)
            if chunks is None:
                unresolved.append(f"{record['id']}: unknown doc '{doc_id}'")
                continue
            matched = resolve_for_document(
                doc_id, chunks, record["question"], record.get("expected_answer", "")
            )
            if matched is None:
                unresolved.append(f"{record['id']}: no match in '{doc_id}'")
                continue
            if matched not in new_refs:
                new_refs.append(matched)
        if new_refs != record.get("relevant_chunk_ids", []):
            changes += 1
            record["relevant_chunk_ids"] = new_refs
        updated.append(json.dumps(record, ensure_ascii=False))

    # Report chunk inventory per document for verification.
    print(f"Processed {len(updated)} records; {changes} changed; {len(unresolved)} unresolved.")
    if unresolved:
        for u in unresolved:
            print(f"  UNRESOLVED: {u}")
        return 1
    if args.write:
        dataset_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print(f"Wrote {dataset_path}")
    else:
        print("Dry-run: no file written (use --write to commit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())