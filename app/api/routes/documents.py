"""Document management routes: upload, list, get, ingest, and job status."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
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
from app.db.models import Document, IngestionJob
from app.db.session import SessionLocal
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_id_from_path(file_path: str) -> str:
    """Stable SHA-1 document ID from an absolute file path."""
    return hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:32]


def _save_upload_file(upload_file: UploadFile, dest_dir: str) -> str:
    """Save an UploadFile to *dest_dir* and return the absolute file path."""
    os.makedirs(dest_dir, exist_ok=True)
    file_path = os.path.join(dest_dir, upload_file.filename or "upload")
    with open(file_path, "wb") as f:
        shutil_copyfileobj(upload_file.file, f)
    return file_path


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
async def upload_document(
    response: Response,
    file: UploadFile = File(..., description="Document file to ingest"),  # noqa: B008
) -> DocumentUploadResponse:
    """Upload and register a single document for later ingestion.

    The document is saved to a temporary location and a metadata record is
    created. Call POST /documents/{document_id}/ingest to actually process it.
    """
    _validate_upload(file)

    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = _save_upload_file(file, tmp_dir)
        document_id = _doc_id_from_path(file_path)

    # Get file size
    file_size = file.size

    # Create DB record
    session = SessionLocal()
    try:
        doc = _create_document_record(
            session, document_id, file.filename or "unknown", file_path, file_size
        )
        session.commit()
    finally:
        session.close()

    response.headers["X-Request-ID"] = document_id

    return DocumentUploadResponse(
        document_id=document_id,
        title=doc.title,
        status=DocumentStatus.PENDING,
        message="Document registered. Call POST /documents/{document_id}/ingest to start processing.",
    )


# ---------------------------------------------------------------------------
# GET /documents — list
# ---------------------------------------------------------------------------

@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
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
async def get_document(document_id: str) -> DocumentMetadata:
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
async def ingest_document(document_id: str) -> IngestionJobResponse:
    """Trigger synchronous ingestion of a registered document.

    The document is parsed, chunked, and indexed into the vector store (Qdrant)
    and BM25 index (OpenSearch). This is a blocking call — it returns when
    ingestion is complete or has failed.
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

        # Create a job record
        import uuid
        job_id = str(uuid.uuid4())
        job = _create_job_record(session, document_id, job_id)
        session.commit()
    finally:
        session.close()

    # Run ingestion synchronously in the endpoint (blocking)
    pipeline = IngestionPipeline()
    try:
        result = pipeline.ingest(file_path=doc.source, title=doc.title)
        # Update job as success
        _finish_job(document_id, job_id, status_=JobStatus.SUCCESS, chunk_count=result.chunk_count)
        return IngestionJobResponse(
            job_id=job_id,
            document_id=document_id,
            status=JobStatus.SUCCESS,
            chunk_count=result.chunk_count,
            error=None,
            started_at=job.started_at or datetime.utcnow(),
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

@router.get("/{document_id}/jobs", response_model=JobListResponse)
async def list_ingestion_jobs(document_id: str) -> JobListResponse:
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
# shim for shutil.copyfileobj (not available in all environments)
# ---------------------------------------------------------------------------

def shutil_copyfileobj(src, dst, length: int = 16384) -> None:
    """Copy *src* file object to *dst* in chunks."""
    while True:
        chunk = src.read(length)
        if not chunk:
            break
        dst.write(chunk)
