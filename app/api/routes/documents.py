"""Document management routes: upload, list, get, ingest, and job status."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.schemas.common import ErrorResponse
from app.api.schemas.documents import (
    MAX_FILE_SIZE_BYTES,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
    IngestionJobResponse,
    JobListResponse,
    JobStatus,
    validate_file_extension,
)
from app.config import get_settings
from app.db.models import Document, IngestionJob
from app.db.session import SessionLocal
from app.ingestion.pipeline import IngestionPipeline, _doc_id_from_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# How long a successful worker detection stays valid. Avoids re-probing (and
# risking a blocking ping) on every upload while still noticing a stopped
# worker within the TTL.
_WORKER_PROBE_TTL = 30.0
_worker_probe_cache: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persist_upload(upload_file: UploadFile) -> tuple[str, str]:
    """Persist an upload under a content-addressed directory.

    The file is streamed to a temporary name while a SHA-1 of its bytes is
    computed, then moved to ``<upload_dir>/<content_hash>/<filename>``. Only the
    basename of the client-supplied name is used, so a crafted name cannot
    escape the upload directory.

    Addressing by content hash makes re-uploading the same file resolve to the
    same document ID, so the pipeline upserts the existing chunks instead of
    indexing a duplicate copy.

    Returns ``(absolute_path, content_hash)``.
    """
    base_dir = _upload_destination()
    safe_name = os.path.basename(upload_file.filename or "upload")

    incoming_dir = os.path.join(base_dir, ".incoming")
    os.makedirs(incoming_dir, exist_ok=True)
    tmp_path = os.path.join(incoming_dir, f"{uuid.uuid4().hex}_{safe_name}")

    digest = hashlib.sha1()
    with open(tmp_path, "wb") as f:
        while True:
            block = upload_file.file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            f.write(block)

    content_hash = digest.hexdigest()[:12]
    dest_dir = os.path.join(base_dir, content_hash)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.abspath(os.path.join(dest_dir, safe_name))

    # The directory name *is* the content hash, so if the destination already
    # exists it holds byte-identical content. Reuse it and discard our copy:
    # writing under a suffixed name here would create a second document for the
    # same file and make the content appear twice in retrieval results.
    if os.path.exists(dest_path):
        os.remove(tmp_path)
        return dest_path, content_hash

    try:
        os.replace(tmp_path, dest_path)
    except PermissionError:
        # Windows can still refuse if the destination appeared and is held open
        # between the check above and this move. Same reasoning applies: the
        # existing file has identical content, so keep it and discard ours.
        os.remove(tmp_path)
    return dest_path, content_hash


def _upload_destination() -> str:
    """Return the directory that uploads are persisted into."""
    return get_settings().upload_dir


def _validate_upload(file: UploadFile) -> None:
    """Validate a single uploaded file; raise HTTPException on failure."""
    if file.size is not None and file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // 1024 // 1024
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File {file.filename!r} exceeds the {max_mb} MB limit",
                "request_id": None,
            },
        )
    try:
        validate_file_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": str(exc),
                "request_id": None,
            },
        ) from exc


def _create_document_record(
    session,  # SQLAlchemy Session
    document_id: str,
    title: str,
    file_path: str,
    file_size_bytes: int | None,
) -> Document:
    """Create or update (upsert) a Document record."""
    # Check if document already exists
    existing = session.execute(
        select(Document).where(Document.document_id == document_id)
    ).scalar_one_or_none()

    if existing:
        existing.title = title
        existing.updated_at = datetime.utcnow()
        return existing

    _, ext = os.path.splitext(file_path)
    format_ = ext.lstrip(".").lower()
    module = "unknown"  # could be derived from path structure

    doc = Document(
        document_id=document_id,
        title=title,
        module=module,
        source=file_path,
        format=format_,
        file_size_bytes=file_size_bytes,
    )
    session.add(doc)
    return doc


def _celery_worker_available(timeout: float = 0.5) -> bool:
    """Return True when at least one Celery worker answers a control ping.

    Two hazards shape this probe:

    * ``control.ping`` travels over the broker. A single-threaded
      (``--pool=solo``) worker that is *busy* executing a task does not process
      the control message until it finishes, so a naive ping blocks the caller
      for the whole task duration — which stalled uploads for ~25s while a
      previous ingest was running.
    * A reachable broker is not a running worker. Skipping the check entirely
      strands tasks in Redis, leaving documents stuck in ``processing``.

    The probe therefore runs on a daemon thread with a hard deadline and a
    positive verdict is cached briefly. If the deadline expires we fail *open*
    (assume a worker exists): a busy worker is still a worker, and it will drain
    the queue. A quick, definitively empty reply means nothing is listening, so
    callers fall back to synchronous ingestion.
    """
    now = time.monotonic()
    cached = _worker_probe_cache.get("available_until", 0.0)
    if cached > now:
        return True

    try:
        from app.celery_app import celery_app
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery app unavailable: %s", exc)
        return False

    outcome: list[bool | None] = [None]

    def _probe() -> None:
        try:
            outcome[0] = bool(celery_app.control.ping(timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — broker unreachable, etc.
            logger.warning("Celery worker ping failed: %s", exc)
            outcome[0] = False

    thread = threading.Thread(target=_probe, daemon=True, name="celery-ping")
    thread.start()
    # Allow the ping timeout plus a small scheduling margin.
    thread.join(timeout + 0.5)

    if outcome[0] is None:
        # Still waiting on a busy worker — don't block the request.
        logger.info("Celery ping timed out (worker busy?); enqueuing anyway.")
        return True

    if outcome[0]:
        _worker_probe_cache["available_until"] = time.monotonic() + _WORKER_PROBE_TTL
    return outcome[0]


def _try_enqueue_ingest(
    document_id: str, source_path: str, title: str, job_id: str
) -> bool:
    """Enqueue ingestion on a *live* Celery worker, else return False.

    Returns False when no worker is consuming the ``ingestion`` queue (or the
    broker is unreachable) so callers fall back to synchronous ingestion. That
    keeps uploads working with just ``docker compose up`` and no worker
    running, and stops jobs from silently stalling in Redis forever.
    """
    if not _celery_worker_available():
        logger.info(
            "No Celery worker is consuming the ingestion queue; "
            "ingesting synchronously instead."
        )
        return False
    try:
        from app.tasks.ingestion import ingest_document_job

        ingest_document_job.delay(document_id, source_path, title, job_id)
        return True
    except Exception as exc:  # noqa: BLE001 — broker down, old pickle, etc.
        logger.warning("Celery enqueue failed, falling back to sync: %s", exc)
        return False


def _create_job_record(
    session,
    document_id: str,
    job_id: str,
) -> IngestionJob:
    """Create a new IngestionJob record in RUNNING state."""
    job = IngestionJob(
        id=job_id,
        document_id=document_id,
        status=JobStatus.RUNNING.value,
        started_at=datetime.utcnow(),
    )
    session.add(job)
    return job


# ---------------------------------------------------------------------------
# POST /documents — upload
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
    },
)
def upload_document(
    response: Response,
    file: UploadFile = File(..., description="Document file to ingest"),  # noqa: B008
) -> DocumentUploadResponse:
    """Upload and register a single document for later ingestion.

    The document is saved to a temporary location and a metadata record is
    created. Call POST /documents/{document_id}/ingest to actually process it.
    """
    _validate_upload(file)

    # Persist the upload so a later /ingest call can still read the file.
    file_path, _content_hash = _persist_upload(file)
    document_id = _doc_id_from_path(file_path)

    # Create DB record
    session = SessionLocal()
    try:
        doc = _create_document_record(
            session, document_id, file.filename or "unknown", file_path, file.size
        )
        title = doc.title
        session.commit()
    finally:
        session.close()

    response.headers["X-Request-ID"] = document_id

    return DocumentUploadResponse(
        document_id=document_id,
        title=title,
        status=DocumentStatus.PENDING,
        message="Document registered. Call POST /documents/{document_id}/ingest to start processing.",
    )


# ---------------------------------------------------------------------------
# POST /documents/upload — upload + ingest in one call
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "Ingestion failed"},
    },
)
# Sync endpoint: ingestion is blocking (parsing + CPU-bound embedding +
# synchronous calls to Qdrant/Postgres). Running it on the event loop would
# freeze the server for the whole ingest (~30s on a cold embedding model). A
# sync endpoint runs in FastAPI's worker threadpool instead.
def upload_and_ingest(
    file: UploadFile = File(..., description="Document file to ingest"),  # noqa: B008
    background: bool = Query(default=False),
) -> DocumentUploadResponse:
    """Upload a document and ingest it.

    Default is synchronous (blocking, returns when indexed). Pass
    ``?background=true`` to enqueue on the Celery/Redis worker and return
    201 immediately with status=processing — follow
    ``GET /documents/{id}/jobs/{job_id}/stream`` for progress.
    """
    _validate_upload(file)

    file_path, _content_hash = _persist_upload(file)
    document_id = _doc_id_from_path(file_path)
    title = file.filename or Path(file_path).stem

    session = SessionLocal()
    try:
        _create_document_record(session, document_id, title, file_path, file.size)
        job = _create_job_record(session, document_id, str(uuid.uuid4()))
        job_id = job.id
        session.commit()
    finally:
        session.close()

    if background and _try_enqueue_ingest(document_id, file_path, title, job_id):
        return DocumentUploadResponse(
            document_id=document_id,
            title=title,
            status=DocumentStatus.PROCESSING,
            chunk_count=None,
            message=(
                f"Ingestion queued as job {job_id}. Stream progress at "
                f"/documents/{document_id}/jobs/{job_id}/stream."
            ),
        )

    try:
        result = IngestionPipeline().ingest(file_path=file_path, title=title)
    except Exception as exc:
        _finish_job(document_id, job_id, status_=JobStatus.FAILED, error=str(exc))
        logger.exception("Ingestion failed for %s", file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INGESTION_FAILED",
                "message": f"Ingestion failed: {exc}",
                "request_id": document_id,
            },
        ) from exc

    return DocumentUploadResponse(
        document_id=result.document_id,
        title=result.title,
        status=DocumentStatus.READY,
        chunk_count=result.chunk_count,
        message=(
            f"Ingested {result.chunk_count} chunks. Query it with "
            f'document_ids=["{result.document_id}"].'
        ),
    )


# ---------------------------------------------------------------------------
# GET /documents — list
# ---------------------------------------------------------------------------

@router.get("", response_model=DocumentListResponse)
# Sync endpoint: performs blocking SQLAlchemy queries.
def list_documents() -> DocumentListResponse:
    """Return all registered documents and their processing status."""
    session = SessionLocal()
    try:
        docs = session.execute(select(Document)).scalars().all()
        chunk_counts = {}
        for doc_id in [d.document_id for d in docs]:
            chunk_counts[doc_id] = _get_chunk_count(session, doc_id)

        metadata = [
            DocumentMetadata(
                document_id=d.document_id,
                title=d.title,
                status=_infer_status(d),
                chunk_count=chunk_counts.get(d.document_id, 0),
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in docs
        ]
        return DocumentListResponse(documents=metadata, total=len(metadata))
    finally:
        session.close()


def _infer_status(doc: Document) -> DocumentStatus:
    """Infer document processing status from its job history."""
    session = SessionLocal()
    try:
        latest_job = (
            session.execute(
                select(IngestionJob)
                .where(IngestionJob.document_id == doc.document_id)
                .order_by(IngestionJob.started_at.desc())
            )
            .scalars()
            .first()
        )
        if latest_job is None:
            return DocumentStatus.PENDING
        if latest_job.status == JobStatus.RUNNING.value:
            return DocumentStatus.PROCESSING
        if latest_job.status == JobStatus.SUCCESS.value:
            return DocumentStatus.READY
        return DocumentStatus.FAILED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /documents/{document_id} — get metadata
# ---------------------------------------------------------------------------

@router.get("/{document_id}", response_model=DocumentMetadata)
# Sync endpoint: performs blocking SQLAlchemy queries.
def get_document(document_id: str) -> DocumentMetadata:
    """Return metadata for a specific document."""
    session = SessionLocal()
    try:
        doc = session.execute(
            select(Document).where(Document.document_id == document_id)
        ).scalar_one_or_none()
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "DOCUMENT_NOT_FOUND",
                    "message": f"No document found with ID {document_id!r}",
                    "request_id": None,
                },
            )
        return DocumentMetadata(
            document_id=doc.document_id,
            title=doc.title,
            status=_infer_status(doc),
            chunk_count=_get_chunk_count(session, document_id),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
    finally:
        session.close()


def _get_chunk_count(session, document_id: str) -> int:
    """Return the number of chunks for a document."""
    from sqlalchemy import func

    from app.db.models import Chunk

    result = session.execute(
        select(func.count()).select_from(Chunk).where(
            Chunk.document_id == document_id
        )
    ).scalar()
    return int(result) if result else 0


# ---------------------------------------------------------------------------
# POST /documents/{document_id}/ingest — trigger ingestion
# ---------------------------------------------------------------------------

@router.post(
    "/{document_id}/ingest",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
# Sync endpoint: runs the same blocking ingestion path as /upload.
# Pass ?background=true to enqueue on the Celery worker (Redis) and return
# 202 immediately — the UI can then follow progress via the SSE stream below.
def ingest_document(
    document_id: str, background: bool = Query(default=False)
) -> IngestionJobResponse:
    """Trigger ingestion of a registered document.

    Default is synchronous (blocking, returns when complete). Pass
    ``?background=true`` to enqueue on the Celery/Redis worker and return
    202 immediately with status=running — poll
    ``GET /documents/{id}/jobs`` or stream
    ``GET /documents/{id}/jobs/{job_id}/stream`` for progress.
    """
    session = SessionLocal()
    try:
        doc = session.execute(
            select(Document).where(Document.document_id == document_id)
        ).scalar_one_or_none()
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "DOCUMENT_NOT_FOUND",
                    "message": f"No document found with ID {document_id!r}",
                    "request_id": None,
                },
            )

        # Capture values before the session closes: `commit` expires the
        # instance, so its attributes become unreadable once detached.
        source_path = doc.source
        doc_title = doc.title

        # Create a job record
        job = _create_job_record(session, document_id, str(uuid.uuid4()))
        job_id = job.id
        started_at = job.started_at or datetime.utcnow()
        session.commit()
    finally:
        session.close()

    if background and _try_enqueue_ingest(document_id, source_path, doc_title, job_id):
        # Async path: worker owns the job now; return immediately.
        return IngestionJobResponse(
            job_id=job_id,
            document_id=document_id,
            status=JobStatus.RUNNING,
            chunk_count=None,
            error=None,
            started_at=started_at,
            finished_at=None,
        )

    # Run ingestion synchronously in the endpoint (blocking)
    pipeline = IngestionPipeline()
    try:
        result = pipeline.ingest(file_path=source_path, title=doc_title)
        # Update job as success
        _finish_job(document_id, job_id, status_=JobStatus.SUCCESS, chunk_count=result.chunk_count)
        return IngestionJobResponse(
            job_id=job_id,
            document_id=document_id,
            status=JobStatus.SUCCESS,
            chunk_count=result.chunk_count,
            error=None,
            started_at=started_at,
            finished_at=datetime.utcnow(),
        )
    except Exception as exc:
        _finish_job(
            document_id, job_id, status_=JobStatus.FAILED, error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INGESTION_FAILED",
                "message": f"Ingestion failed: {exc}",
                "request_id": None,
            },
        ) from exc


def _finish_job(
    document_id: str,
    job_id: str,
    status_: JobStatus,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    """Update a job record to a terminal state."""
    session = SessionLocal()
    try:
        job = session.execute(
            select(IngestionJob).where(
                IngestionJob.id == job_id,
                IngestionJob.document_id == document_id,
            )
        ).scalar_one_or_none()
        if job is None:
            return
        job.status = status_.value
        job.finished_at = datetime.utcnow()
        if chunk_count is not None:
            job.chunk_count = chunk_count
        if error is not None:
            job.error = error
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/jobs — list jobs
# ---------------------------------------------------------------------------

# Sync endpoint: performs blocking SQLAlchemy queries.
@router.get("/{document_id}/jobs", response_model=JobListResponse)
def list_ingestion_jobs(document_id: str) -> JobListResponse:
    """Return all ingestion jobs for a document, newest first."""
    session = SessionLocal()
    try:
        jobs = (
            session.execute(
                select(IngestionJob)
                .where(IngestionJob.document_id == document_id)
                .order_by(IngestionJob.started_at.desc())
            )
            .scalars()
            .all()
        )
        return JobListResponse(
            jobs=[
                IngestionJobResponse(
                    job_id=j.id,
                    document_id=j.document_id,
                    status=JobStatus(j.status),
                    chunk_count=j.chunk_count,
                    error=j.error,
                    started_at=j.started_at or datetime.utcnow(),
                    finished_at=j.finished_at,
                )
                for j in jobs
            ]
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/jobs/{job_id}/stream — SSE job progress
# ---------------------------------------------------------------------------


@router.get("/{document_id}/jobs/{job_id}/stream")
def stream_ingestion_job(document_id: str, job_id: str) -> StreamingResponse:
    """Stream ingestion progress as Server-Sent Events.

    Emits ``event: progress`` every second with the job row, then
    ``event: done`` once it reaches success/failed (or after ~120s timeout).
    The browser stops spinning immediately and follows real progress instead
    of blocking on the 20-60s cold embed.
    """
    import asyncio
    import json

    async def _events():
        for _ in range(120):
            session = SessionLocal()
            try:
                job = session.execute(
                    select(IngestionJob).where(
                        IngestionJob.id == job_id,
                        IngestionJob.document_id == document_id,
                    )
                ).scalar_one_or_none()
                if job is None:
                    yield "event: error\ndata: "
                    yield json.dumps({"code": "JOB_NOT_FOUND"})
                    yield "\n\n"
                    return
                payload = json.dumps(
                    {
                        "job_id": job.id,
                        "document_id": job.document_id,
                        "status": job.status,
                        "chunk_count": job.chunk_count,
                        "error": job.error,
                    }
                )
                if job.status in (
                    JobStatus.SUCCESS.value,
                    JobStatus.FAILED.value,
                    "success",
                    "failed",
                ):
                    yield f"event: done\ndata: {payload}\n\n"
                    return
                yield f"event: progress\ndata: {payload}\n\n"
            finally:
                session.close()
            await asyncio.sleep(1)
        yield 'event: timeout\ndata: {"code": "STREAM_TIMEOUT"}\n\n'

    return StreamingResponse(_events(), media_type="text/event-stream")
