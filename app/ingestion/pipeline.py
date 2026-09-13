"""Ingestion pipeline: parse → clean → chunk → embed → store.

This module wires together the individual ingestion components into a single
callable (:class:`IngestionPipeline`) that a route or background worker can
invoke.

Usage::

    pipeline = IngestionPipeline()
    result = await pipeline.ingest(file_path="data/corpus/markdown/authentication-guide.md")
    print(f"Ingested {result.chunk_count} chunks")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.config.settings import Settings
from app.db.models import Chunk as ChunkModel
from app.db.models import Document, IngestionJob
from app.db.session import session_scope
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import Chunk, Chunker
from app.ingestion.cleaner import clean_text
from app.ingestion.parsers import extract_docx, extract_html, extract_markdown, extract_pdf
from app.observability.metrics import record
from app.observability.timing import time_operation

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------  .
# Format → parser mapping
# --------------------------------------------------------------------------  .

_FORMAT_PARSERS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_markdown,
}


def _parser_for(path: str | Path):
    suffix = Path(path).suffix.lower()
    parser = _FORMAT_PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"No parser for file suffix {suffix!r}. Supported: {list(_FORMAT_PARSERS)}")
    return parser


# --------------------------------------------------------------------------  .
# Document ID generation
# --------------------------------------------------------------------------  .

def _doc_id_from_path(file_path: str | Path) -> str:
    """Stable document ID = SHA-256 of the canonical path, truncated to 32 hex chars."""
    p = Path(file_path).resolve()
    return hashlib.sha1(p.as_posix().encode()).hexdigest()[:32]


# --------------------------------------------------------------------------  .
# Result type
# --------------------------------------------------------------------------  .


@dataclass
class IngestResult:
    """Outcome of a single document ingestion."""

    document_id: str
    title: str
    chunk_count: int
    job_id: str
    stage_ms: dict[str, float] | None = None


# --------------------------------------------------------------------------  .
# Pipeline
# --------------------------------------------------------------------------  .

class IngestionPipeline:
    """End-to-end document ingestion.

    Parameters
    ----------
    settings:
        Application settings (injected via FastAPI DI).
    embedder:
        Embedder instance for vector storage.
    chunker:
        Chunker instance. Defaults to the configured strategy from settings.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(settings)
        self.chunker = Chunker(
            strategy=self.settings.chunk_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def ingest(self, file_path: str | Path, title: str | None = None) -> IngestResult:
        """Run the full ingestion pipeline synchronously.

        Steps
        -----
        1. Parse the file (format detected from extension).
        2. Clean the extracted text.
        3. Chunk the cleaned text.
        4. Store chunk metadata in PostgreSQL.
        5. Compute embeddings and upsert to Qdrant + keyword postings.

        Returns
        -------
        IngestResult
        """

        file_path = Path(file_path)
        doc_id = _doc_id_from_path(file_path)
        logger.info("Starting ingestion for %s (doc_id=%s)", file_path, doc_id)

        # Create a running IngestionJob record.
        job = self._create_ingestion_job(doc_id)
        stage_ms: dict[str, float] = {}

        try:
            # 1. Parse
            with time_operation("ingest.parse", log_on_exit=False) as t:
                parser = _parser_for(file_path)
                parsed = parser(file_path)
            stage_ms["parse"] = t.duration_ms
            logger.info("Parsed %s: %d chars raw", file_path, len(parsed.text))

            # 2. Clean
            with time_operation("ingest.clean", log_on_exit=False) as t:
                cleaned = clean_text(parsed.text)
            stage_ms["clean"] = t.duration_ms
            logger.info("Cleaned %s: %d chars after cleaning", file_path, len(cleaned))

            # 3. Chunk
            with time_operation("ingest.chunk", log_on_exit=False) as t:
                chunks: list[Chunk] = self.chunker.chunk(
                    text=cleaned,
                    document_id=doc_id,
                    document_name=file_path.stem,
                    source=str(file_path),
                )
            stage_ms["chunk"] = t.duration_ms
            logger.info("Chunked %s: %d chunks produced", file_path, len(chunks))

            # 4 & 5. Persist to PostgreSQL + index in Qdrant / keyword postings
            persist_stages = self._persist_chunks(chunks, doc_id, title or file_path.stem)
            stage_ms.update(persist_stages)

            self._mark_job_success(job.id, len(chunks))
            total = sum(stage_ms.values())
            for stage, ms in stage_ms.items():
                record(f"ingest.{stage}", ms)
            logger.info(
                "Ingestion complete for %s: %d chunks in %.1fs (stages ms: %s)",
                file_path,
                len(chunks),
                total / 1000,
                {k: round(v, 1) for k, v in stage_ms.items()},
            )
            return IngestResult(
                document_id=doc_id,
                title=title or file_path.stem,
                chunk_count=len(chunks),
                job_id=job.id,
                stage_ms={k: round(v, 1) for k, v in stage_ms.items()},
            )
        except Exception as exc:
            self._mark_job_failed(job.id, str(exc))
            raise

    def _create_ingestion_job(self, document_id: str) -> IngestionJob:
        """Create an IngestionJob record and mark it as running."""
        with session_scope() as sess:
            job = IngestionJob(
                document_id=document_id,
                status="running",
                started_at=datetime.now(UTC),
            )
            sess.add(job)
            sess.commit()
            sess.refresh(job)
            return job

    def _mark_job_success(self, job_id: str, chunk_count: int) -> None:
        """Mark an IngestionJob as success with its final chunk count."""
        with session_scope() as sess:
            job = sess.get(IngestionJob, job_id)
            if not job:
                return
            job.status = "success"
            job.chunk_count = chunk_count
            job.finished_at = datetime.now(UTC)
            sess.commit()

    def _mark_job_failed(self, job_id: str, error: str) -> None:
        """Mark an IngestionJob as failed with the error message."""
        with session_scope() as sess:
            job = sess.get(IngestionJob, job_id)
            if not job:
                return
            job.status = "failed"
            job.error = error
            job.finished_at = datetime.now(UTC)
            sess.commit()

    def _persist_chunks(self, chunks: list[Chunk], doc_id: str, title: str) -> dict[str, float]:
        """Write chunks to PostgreSQL, then embed + index them in Qdrant + keyword postings.

        Returns per-stage timings (postgres/embed/qdrant/keyword ms) for logging
        and GET /metrics attribution.
        """
        from app.retrieval.bm25 import BM25Indexer
        from app.retrieval.vector import VectorStore

        stages: dict[str, float] = {}
        with time_operation("ingest.postgres", log_on_exit=False) as t, session_scope() as sess:
            # Upsert document record.
            doc_record = sess.query(Document).filter_by(document_id=doc_id).first()
            if not doc_record:
                doc_record = Document(
                    id=doc_id,
                    document_id=doc_id,
                    title=title,
                    module=title.split("-")[0] if "-" in title else title,
                    source=title,
                    format=Path(title).suffix or "unknown",
                )
                sess.add(doc_record)
                sess.flush()

            # Re-ingesting a document must replace its previous chunks rather
            # than insert duplicates: chunk ids are stable per document, so a
            # plain insert would violate the primary key on re-upload.
            sess.query(ChunkModel).filter(ChunkModel.document_id == doc_id).delete(
                synchronize_session=False
            )

            # Chunk records.
            chunk_records = []
            for chunk in chunks:
                chunk_record = ChunkModel(
                    id=chunk.chunk_id,
                    document_id=doc_id,
                    chunk_index=chunk.index,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    text=chunk.text,
                    source=chunk.source,
                    content_hash=chunk.content_hash,
                )
                chunk_records.append(chunk_record)
            sess.add_all(chunk_records)
            sess.commit()
        stages["postgres"] = t.duration_ms

        vector_store = VectorStore(self.settings)
        bm25_indexer = BM25Indexer(self.settings)

        if not chunks:
            # Nothing to index, but a previous version must not stay searchable.
            vector_store.delete_by_document(doc_id)
            bm25_indexer.delete_by_document(doc_id)
            return stages

        # Index chunks after the DB commit (don't hold DB locks during network I/O).
        texts = [c.text for c in chunks]
        with time_operation("ingest.embed", log_on_exit=False) as t:
            vectors = self.embedder.embed_documents(texts)
        stages["embed"] = t.duration_ms

        # Drop any vectors/postings left by a previous ingest of this document,
        # so a re-upload that produces fewer chunks cannot leave orphans behind.
        vector_store.delete_by_document(doc_id)
        bm25_indexer.delete_by_document(doc_id)

        with time_operation("ingest.qdrant", log_on_exit=False) as t:
            vector_store.upsert(chunks, vectors=vectors)
        stages["qdrant"] = t.duration_ms
        with time_operation("ingest.keyword", log_on_exit=False) as t:
            bm25_indexer.index_documents(chunks)
        stages["keyword"] = t.duration_ms
        logger.info(
            "Indexed %d chunks: pg=%d, qdrant=%d, bm25=%d",
            len(chunks),
            len(chunk_records),
            len(vectors),
            len(chunks),
        )
        return stages
