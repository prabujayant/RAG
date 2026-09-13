"""
Celery tasks for asynchronous document ingestion.

These tasks run the full ingestion pipeline (parse → clean → chunk → embed → index)
in a Celery worker, keeping API requests responsive.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.celery_app import celery_app
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".html", ".pdf", ".docx"}


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_corpus(self, corpus_path: str | None = None) -> dict:
    """
    Ingest all documents from ``corpus_path`` (defaults to ``data/corpus``).

    Args:
        corpus_path: Optional path to corpus root. If None, uses the configured default.

    Returns:
        Summary dict with keys: files_processed, chunks_created, duration_seconds, errors.
    """
    root = Path(corpus_path) if corpus_path else Path("data/corpus")
    start = time.monotonic()

    # Collect corpus files
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    pipeline = IngestionPipeline()
    total_chunks = 0
    errors: list[str] = []

    for file_path in files:
        try:
            result = pipeline.ingest(file_path)
            total_chunks += result.chunk_count
        except Exception as exc:
            logger.warning("Failed to ingest %s: %s", file_path, exc)
            errors.append(f"{file_path}: {exc}")

    duration = time.monotonic() - start
    logger.info(
        "ingest_corpus completed",
        extra={
            "files": len(files),
            "chunks": total_chunks,
            "duration": duration,
            "errors": len(errors),
        },
    )
    return {
        "status": "success",
        "files_processed": len(files),
        "chunks_created": total_chunks,
        "duration_seconds": round(duration, 2),
        "errors": errors,
    }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_document_job(
    self, document_id: str, source_path: str, title: str, job_id: str
) -> dict:
    """Ingest one registered document and update its IngestionJob row.

    Used by POST /documents/{id}/ingest?background=true so the HTTP
    request returns 202 immediately while the worker does the blocking
    parse → chunk → embed → index work (~20-60s cold).
    """
    from datetime import datetime

    from sqlalchemy import select

    from app.api.schemas.documents import JobStatus
    from app.db.models import IngestionJob
    from app.db.session import SessionLocal

    start = time.monotonic()
    try:
        pipeline = IngestionPipeline()
        result = pipeline.ingest(file_path=source_path, title=title)
        session = SessionLocal()
        try:
            job = session.execute(
                select(IngestionJob).where(
                    IngestionJob.id == job_id,
                    IngestionJob.document_id == document_id,
                )
            ).scalar_one_or_none()
            if job is not None:
                job.status = JobStatus.SUCCESS.value
                job.finished_at = datetime.utcnow()
                job.chunk_count = result.chunk_count
                session.commit()
        finally:
            session.close()
        return {
            "status": "success",
            "document_id": document_id,
            "job_id": job_id,
            "chunks_created": result.chunk_count,
            "duration_seconds": round(time.monotonic() - start, 2),
            "error": None,
        }
    except Exception as exc:
        logger.exception("ingest_document_job failed for %s", source_path)
        session = SessionLocal()
        try:
            from sqlalchemy import select as _select

            job = session.execute(
                _select(IngestionJob).where(
                    IngestionJob.id == job_id,
                    IngestionJob.document_id == document_id,
                )
            ).scalar_one_or_none()
            if job is not None:
                job.status = "failed"
                job.finished_at = datetime.utcnow()
                job.error = str(exc)
                session.commit()
        finally:
            session.close()
        return {
            "status": "error",
            "document_id": document_id,
            "job_id": job_id,
            "chunks_created": 0,
            "duration_seconds": round(time.monotonic() - start, 2),
            "error": str(exc),
        }
