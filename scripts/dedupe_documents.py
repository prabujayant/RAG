"""Remove duplicate uploaded documents from the index.

Duplicates were created by re-uploading the same file before uploads became
content-addressed: each upload landed in a different ``data/uploads/<random>``
folder and was indexed under its own ``document_id``, so the same content could
appear twice in retrieval results and unfairly dominate the ranking.

This script groups uploaded documents by ``(file name, extension)`` and keeps
the newest copy of each, deleting the older ones from PostgreSQL, Qdrant and
keyword postings.

Usage::

    python scripts/dedupe_documents.py --dry-run   # show what would be removed
    python scripts/dedupe_documents.py             # perform the cleanup

Note: the corpus intentionally stores the same topic in several formats
(e.g. ``api-errors.md`` and ``api-errors.html``). Those are NOT duplicates and
are left untouched, because the grouping key includes the file extension.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.models import Chunk, Document, IngestionJob  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.retrieval.bm25 import BM25Indexer  # noqa: E402
from app.retrieval.vector import VectorStore  # noqa: E402

UPLOADS_MARKER = "uploads"

# Some older uploads were stored under an 8-hex prefixed name to avoid a
# filename collision. Normalise it away so those are recognised as duplicates
# of the original file rather than distinct documents.
_COLLISION_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_")


def _group_key(doc: Document) -> tuple[str, str]:
    """Group by (normalised file name, lower-cased extension)."""
    name = os.path.basename(doc.source or "")
    name = _COLLISION_PREFIX_RE.sub("", name)
    return name.lower(), os.path.splitext(name)[1].lower()


def find_duplicate_uploads(session) -> list[tuple[Document, list[Document]]]:
    """Return ``(keep, [drop, ...])`` pairs for duplicated uploaded files."""
    docs = session.query(Document).filter(Document.source.like(f"%{UPLOADS_MARKER}%")).all()

    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for doc in docs:
        groups[_group_key(doc)].append(doc)

    plan: list[tuple[Document, list[Document]]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        # Prefer the copy whose stored name has no collision prefix (the
        # original), then the most recently created one.
        group.sort(
            key=lambda d: (
                bool(_COLLISION_PREFIX_RE.match(os.path.basename(d.source or ""))),
                -d.created_at.timestamp(),
            )
        )
        plan.append((group[0], group[1:]))
    return plan


def _delete_document(session, document_id: str) -> int:
    """Delete a document and its chunks from every store. Returns chunks removed."""
    # Order matters: ingestion_jobs and chunks both reference documents, so they
    # must go first or PostgreSQL raises a foreign-key violation.
    session.query(IngestionJob).filter(
        IngestionJob.document_id == document_id
    ).delete(synchronize_session=False)
    removed = (
        session.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .delete(synchronize_session=False)
    )
    session.query(Document).filter(Document.document_id == document_id).delete(
        synchronize_session=False
    )
    session.commit()

    try:
        VectorStore().delete_by_document(document_id)
    except Exception as exc:  # noqa: BLE001
        print(f"    warning: Qdrant delete failed for {document_id[:12]}: {exc}")

    try:
        BM25Indexer().delete_by_document(document_id)
    except Exception as exc:  # noqa: BLE001
        print(f"    warning: keyword delete failed for {document_id[:12]}: {exc}")

    return int(removed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed without deleting anything",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        plan = find_duplicate_uploads(session)
        if not plan:
            print("No duplicate uploads found.")
            return 0

        total = 0
        for keep, drops in plan:
            print(f"{os.path.basename(keep.source or '')} ({len(drops) + 1} copies)")
            print(f"  KEEP  {keep.document_id[:12]}  {keep.created_at}")
            for doc in drops:
                count = session.query(Chunk).filter_by(document_id=doc.document_id).count()
                if args.dry_run:
                    print(f"  DROP  {doc.document_id[:12]}  chunks={count}  {doc.created_at}")
                else:
                    removed = _delete_document(session, doc.document_id)
                    print(f"  DROPPED {doc.document_id[:12]}  chunks={removed}")
                    total += 1
            print()

        if args.dry_run:
            print(f"Dry run: would remove {sum(len(d) for _, d in plan)} document(s).")
        else:
            print(f"Removed {total} duplicate document(s).")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
