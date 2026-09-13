"""Bulk-ingest the local document corpus into AskMyDocs.

Walks ``data/corpus`` and runs the :class:`IngestionPipeline` for every
supported document (markdown, HTML, PDF, DOCX), which parses -> cleans ->
chunks -> persists to PostgreSQL and indexes into Qdrant + keyword postings.

Usage:
    python scripts/ingest_corpus.py [--root data/corpus] [--skip-existing]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.models import Document  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from sqlalchemy import select  # noqa: E402

SUPPORTED_EXTENSIONS = {".md", ".html", ".pdf", ".docx"}


def doc_id_from_path(file_path: Path) -> str:
    """Stable SHA-1 doc ID from the canonical posix path (matches pipeline)."""
    return hashlib.sha1(file_path.resolve().as_posix().encode("utf-8")).hexdigest()[:32]


def _collect_files(root: Path) -> list[Path]:
    """Return supported corpus files, sorted for deterministic ordering."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def register_document(file_path: Path, title: str) -> str:
    """Upsert a Document row (required by the ingestion_jobs FK). Returns doc_id."""
    doc_id = doc_id_from_path(file_path)
    with session_scope() as sess:
        existing = sess.execute(
            select(Document).where(Document.document_id == doc_id)
        ).scalar_one_or_none()
        if existing is not None:
            existing.title = title
            return doc_id
        fmt = file_path.suffix.lstrip(".").lower()
        sess.add(
            Document(
                document_id=doc_id,
                title=title,
                module="corpus",
                source=str(file_path.resolve()),
                format=fmt,
                file_size_bytes=file_path.stat().st_size,
            )
        )
        sess.commit()
    return doc_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-ingest the document corpus.")
    parser.add_argument(
        "--root",
        default=str(ROOT / "data" / "corpus"),
        help="Corpus root directory (default: data/corpus)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Ingest at most N documents (0 = no limit). Files are sorted, so "
            "the selection is deterministic. Useful on slow CPU where the full "
            "corpus would take many minutes."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = _collect_files(root)
    if not files:
        print(f"No supported documents found under {root}")
        return 1

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"Found {len(files)} documents to ingest under {root}")

    pipeline = IngestionPipeline()
    ok = 0
    failed: list[str] = []
    for i, path in enumerate(files, start=1):
        try:
            register_document(path, title=path.stem)
            print(f"[{i}/{len(files)}] Ingesting {path.name} ...", flush=True)
            result = pipeline.ingest(file_path=path, title=path.stem)
            print(
                f"    -> ok doc_id={result.document_id} chunks={result.chunk_count}",
                flush=True,
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"    -> FAILED: {exc}", flush=True)
            failed.append(f"{path}: {exc}")

    print(f"\nDone. Ingested {ok}/{len(files)} documents.")
    if failed:
        print("Failures:")
        for f in failed:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
