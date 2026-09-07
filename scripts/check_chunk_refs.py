"""Verify that the golden dataset's chunk references match the corpus.

The ingestion pipeline assigns chunk ids of the form ``{doc_id}:{chunk_index}``
in document order using the paragraph-aware chunker. This script runs the
*actual* chunker installed in the repo against every corpus markdown document
and verifies that each referenced chunk index is in range.

Usage:
    python scripts/check_chunk_refs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.chunker import Chunker, ChunkingStrategy  # noqa: E402

from eval_dataset import load_default  # noqa: E402


def main() -> int:
    dataset = load_default()
    corpus_dir = ROOT / "data" / "corpus" / "markdown"

    chunker = Chunker(strategy=ChunkingStrategy.PARAGRAPH, chunk_size=512, chunk_overlap=64)
    doc_counts: dict[str, int] = {}
    for md_path in sorted(corpus_dir.glob("*.md")):
        chunks = chunker.chunk_text(
            text=md_path.read_text(encoding="utf-8"),
            document_id=md_path.stem,
            document_name=md_path.stem,
            source=f"markdown/{md_path.name}",
        )
        doc_counts[md_path.stem] = len(chunks)

    problems: list[str] = []
    total_refs = 0
    for ex in dataset.examples:
        for chunk_ref in ex.relevant_chunk_ids:
            total_refs += 1
            doc_id, _, idx = chunk_ref.rpartition(":")
            count = doc_counts.get(doc_id)
            if count is None:
                problems.append(f"{ex.id}: unknown document '{doc_id}'")
                continue
            if int(idx) >= count:
                problems.append(
                    f"{ex.id}: chunk id '{chunk_ref}' out of range (doc '{doc_id}' has {count} chunks)"
                )

    if problems:
        print("Chunk reference check FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"Chunk reference check PASSED ({total_refs} references in range)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())