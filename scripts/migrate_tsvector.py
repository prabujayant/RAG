"""Migrate keyword search from OpenSearch to Postgres tsvector.

Idempotent: safe to run on fresh DBs (creates tables) and existing DBs
(adds ``keyword_postings`` + trigger, backfills postings from ``chunks``).

Usage:
    python scripts/migrate_tsvector.py

What it does:
1. ``Base.metadata.create_all`` (creates ``keyword_postings`` if missing).
2. Runs ``KEYWORD_DDL`` (tsv column, weighted trigger, GIN index).
3. Backfills postings for chunks that have none (existing corpus keeps
   working without re-ingestion; embeddings/Qdrant untouched).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.models import Base  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.retrieval.bm25 import KEYWORD_DDL  # noqa: E402
from sqlalchemy import text  # noqa: E402


def main() -> int:
    Base.metadata.create_all(bind=engine)
    print("Tables ensured.")

    with SessionLocal() as session:
        for stmt in KEYWORD_DDL:
            session.execute(text(stmt))
        session.commit()
    print("tsvector column, trigger, and GIN index ensured.")

    with SessionLocal() as session:
        res = session.execute(
            text(
                "INSERT INTO keyword_postings"
                " (chunk_id, document_id, text, source, page_number, section)"
                " SELECT c.id, c.document_id, c.text, c.source,"
                " c.page_number, c.section FROM chunks c"
                " WHERE NOT EXISTS (SELECT 1 FROM keyword_postings k"
                " WHERE k.chunk_id = c.id)"
            )
        )
        session.commit()
        print(f"Backfilled {res.rowcount} postings from chunks.")

        total = session.execute(text("SELECT COUNT(*) FROM keyword_postings")).scalar()
        print(f"Total keyword postings: {total}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
