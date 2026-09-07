"""Initialize the AskMyDocs platform.

Creates all PostgreSQL tables, bootstraps the Qdrant collection and the
OpenSearch BM25 index, and ensures the evaluation output directories exist.

Usage:
    python scripts/init_db.py                # full bootstrap (PG + Qdrant + OpenSearch)
    python scripts/init_db.py --no-vectors   # PG only (offline / unit-test setup)

The default .env connects to the services started by
`docker compose up -d`. Each step is independent — a failure in vector or
BM25 bootstrap does not roll back the database schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.db.session import init_db  # noqa: E402


def ensure_eval_dirs() -> None:
    for rel in ("evals/reports", "evals/results", "evals/baselines"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def bootstrap_vectors() -> tuple[bool, bool]:
    """Best-effort bootstrap of Qdrant + OpenSearch.

    Returns ``(qdrant_ok, opensearch_ok)``. Failures are logged but do not
    raise, so this script can run in environments where the vector stores
    are not yet up.
    """
    qdrant_ok = opensearch_ok = False
    try:
        from app.retrieval.vector import VectorStore

        VectorStore().ensure_collection()
        qdrant_ok = True
        print("Qdrant collection ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Qdrant bootstrap skipped: {exc}")

    try:
        from app.retrieval.bm25 import BM25Indexer

        BM25Indexer().ensure_index()
        opensearch_ok = True
        print("OpenSearch BM25 index ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] OpenSearch bootstrap skipped: {exc}")

    return qdrant_ok, opensearch_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize AskMyDocs storage.")
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Only initialize PostgreSQL (skip Qdrant + OpenSearch).",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"Connecting to: {settings.database_url}")
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialize database: {exc}")
        print("Is PostgreSQL running? Try `make docker-up` first.")
        return 1
    print("Database schema initialized.")

    if not args.no_vectors:
        bootstrap_vectors()
    else:
        print("Skipping Qdrant + OpenSearch bootstrap (--no-vectors).")

    ensure_eval_dirs()
    print("Evaluation directories ensured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())