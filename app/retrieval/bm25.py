"""Postgres tsvector keyword search (BM25-side of hybrid retrieval).

Same public interface as the vector store, so no caller changes: hybrid
fusion, the ingestion pipeline, and the evaluation runner all keep working.
Ranking uses ``ts_rank_cd`` over a weighted ``tsv`` (chunk text weight A,
section weight B, english config).

Postings live in their own ``keyword_postings`` table — not as a column on
``chunks`` — so posting lifecycle (index / delete / count) stays independent
of chunk rows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from threading import Lock

from sqlalchemy import text

from app.config import get_settings
from app.config.settings import Settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# DDL for the tsvector column, trigger, and GIN index. Runs idempotently via
# ensure_index(), scripts/migrate_tsvector.py (existing DBs + backfill), and
# scripts/init_db.py. The ORM model (KeywordPosting) intentionally does NOT
# map ``tsv`` — the trigger owns it on every insert/update.
KEYWORD_DDL = [
    "ALTER TABLE keyword_postings ADD COLUMN IF NOT EXISTS tsv tsvector",
    """CREATE OR REPLACE FUNCTION keyword_postings_tsv_trigger() RETURNS trigger AS $$
BEGIN
  NEW.tsv :=
    setweight(to_tsvector('pg_catalog.english', coalesce(NEW.text, '')), 'A') ||
    setweight(to_tsvector('pg_catalog.english', coalesce(NEW.section, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql""",
    """DROP TRIGGER IF EXISTS keyword_postings_tsv_update ON keyword_postings""",
    """CREATE TRIGGER keyword_postings_tsv_update
BEFORE INSERT OR UPDATE OF text, section ON keyword_postings
FOR EACH ROW EXECUTE FUNCTION keyword_postings_tsv_trigger()""",
    "CREATE INDEX IF NOT EXISTS keyword_postings_tsv_gin ON keyword_postings USING GIN (tsv)",
]

_ENSURED_KEY = "keyword_postings"
_ENSURED: set[str] = set()
_ENSURED_LOCK = Lock()


def _session_factory_from_settings() -> object:
    """Return the process SessionLocal (imported lazily to avoid cycles)."""
    from app.db.session import SessionLocal

    return SessionLocal


class BM25Indexer:
    """Postgres-backed keyword indexer (index / delete / search / count).

    Parameters
    ----------
    settings:
        Application settings (currently unused beyond compat; kept so all
        existing constructors ``BM25Indexer(settings)`` keep working).
    session_factory:
        Callable returning a context-managed SQLAlchemy session
        (``with factory() as session: ...``). Defaults to ``SessionLocal``.
        Inject a fake in unit tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        session_factory: object | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._sessions = session_factory or _session_factory_from_settings()

    # ------------------------------------------------------------------ ddl

    def ensure_index(self) -> None:
        """Create the tsv column, trigger, and GIN index if missing.

        Idempotent and memoized per process. Safe to call on every search.
        """
        with _ENSURED_LOCK:
            if _ENSURED_KEY in _ENSURED:
                return
        with self._sessions() as session:  # type: ignore[operator]
            for stmt in KEYWORD_DDL:
                session.execute(text(stmt))
            session.commit()
        with _ENSURED_LOCK:
            _ENSURED.add(_ENSURED_KEY)
        logger.debug("Keyword postings tsvector objects ensured")

    # ----------------------------------------------------------------- write

    def index_documents(self, chunks: list[Chunk], refresh: bool = True) -> None:  # noqa: ARG002
        """Upsert keyword postings for *chunks* (idempotent on chunk_id).

        ``refresh`` is accepted for interface compatibility and ignored:
        Postgres postings are visible on commit — there is no segment-refresh
        step anymore (that per-document refresh was the old bulk-ingest tax).
        """
        if not chunks:
            return
        self.ensure_index()
        from app.db.models import KeywordPosting

        with self._sessions() as session:  # type: ignore[operator]
            for chunk in chunks:
                posting = KeywordPosting(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    section=chunk.section,
                )
                session.merge(posting)
            session.commit()
        logger.info("Keyword-indexed %d chunks (tsvector)", len(chunks))

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        """Alias for :meth:`index_documents` (parity with VectorStore)."""
        chunk_list = list(chunks)
        self.index_documents(chunk_list)
        return len(chunk_list)

    # ------------------------------------------------------------------ read

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_document_ids: list[str] | None = None,
        exclude_document_ids: list[str] | None = None,
        fields: list[str] | None = None,  # noqa: ARG002
    ) -> list[dict]:
        """Search postings with ``ts_rank_cd``.

        ``fields`` is accepted for interface compatibility and ignored: the
        ``tsv`` always covers text (weight A) + section (weight B).
        Returns dicts with ``id``, ``score``, ``payload`` like before.
        """
        if not query or not query.strip():
            return []
        self.ensure_index()
        # OR semantics for natural-language questions: websearch_to_tsquery
        # ANDs unquoted terms ('a & b'), which returns near-empty results.
        # Rewriting '&' to '|' keeps PG stemming while scoring any-term
        # matches. nullif guards stopword-only queries (empty tsquery would
        # be a syntax error; NULL matches nothing instead).
        tsquery = (
            "to_tsquery('english', nullif(replace("
            "websearch_to_tsquery('english', :q)::text, ' & ', ' | '), ''))"
        )
        clauses = [f"tsv @@ {tsquery}"]
        params: dict = {"q": query, "k": top_k}
        if filter_document_ids:
            clauses.append("document_id = ANY(:f)")
            params["f"] = list(filter_document_ids)
        if exclude_document_ids:
            clauses.append("NOT (document_id = ANY(:e))")
            params["e"] = list(exclude_document_ids)
        stmt = text(
            "SELECT chunk_id, document_id, text, source, page_number, section,"
            f" ts_rank_cd(tsv, {tsquery}) AS score"
            " FROM keyword_postings"
            f" WHERE {' AND '.join(clauses)}"
            " ORDER BY score DESC LIMIT :k"
        )
        with self._sessions() as session:  # type: ignore[operator]
            rows = session.execute(stmt, params).mappings().all()
        return [
            {
                "id": r["chunk_id"],
                "score": float(r["score"]),
                "payload": {
                    "document_id": r["document_id"],
                    "text": r["text"],
                    "source": r["source"],
                    "page_number": r["page_number"],
                    "section": r["section"],
                },
            }
            for r in rows
        ]

    def get(self, chunk_id: str) -> dict | None:
        """Return the posting for a single chunk, or None if missing."""
        with self._sessions() as session:  # type: ignore[operator]
            from app.db.models import KeywordPosting

            row = session.get(KeywordPosting, chunk_id)
            if row is None:
                return None
            return {
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "text": row.text,
                "source": row.source,
                "page_number": row.page_number,
                "section": row.section,
            }

    def count(self, document_id: str | None = None) -> int:
        """Count postings, optionally filtered by document_id."""
        with self._sessions() as session:  # type: ignore[operator]
            from sqlalchemy import func, select

            from app.db.models import KeywordPosting

            q = select(func.count()).select_from(KeywordPosting)
            if document_id is not None:
                q = q.where(KeywordPosting.document_id == document_id)
            return int(session.execute(q).scalar() or 0)

    # ---------------------------------------------------------------- delete

    def delete_by_document(self, document_id: str, refresh: bool = True) -> int:  # noqa: ARG002
        """Delete all postings of one document. Returns the number deleted."""
        from app.db.models import KeywordPosting

        with self._sessions() as session:  # type: ignore[operator]
            n = (
                session.query(KeywordPosting)
                .filter(KeywordPosting.document_id == document_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(n)

    def delete_by_ids(self, chunk_ids: Iterable[str], refresh: bool = True) -> int:  # noqa: ARG002
        """Delete specific postings by chunk_id. Returns the number deleted."""
        from app.db.models import KeywordPosting

        ids = list(chunk_ids)
        if not ids:
            return 0
        with self._sessions() as session:  # type: ignore[operator]
            n = (
                session.query(KeywordPosting)
                .filter(KeywordPosting.chunk_id.in_(ids))
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(n)

    def delete_index(self) -> None:
        """Drop all postings (idempotent). The table and trigger stay."""
        from app.db.models import KeywordPosting

        with self._sessions() as session:  # type: ignore[operator]
            session.query(KeywordPosting).delete(synchronize_session=False)
            session.commit()
        with _ENSURED_LOCK:
            _ENSURED.discard(_ENSURED_KEY)
        logger.info("Cleared keyword postings")
