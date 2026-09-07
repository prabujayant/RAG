"""API schemas for document operations."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.api.schemas.common import ErrorDetail


class DocumentStatus(StrEnum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Ingestion job status."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""

    document_id: str = Field(description="Stable document identifier (SHA-1 of path)")
    title: str = Field(description="Document title or file name")
    status: DocumentStatus = Field(description="Processing status")
    chunk_count: int = Field(description="Number of chunks indexed for this document")
    created_at: datetime = Field(description="When the document was first ingested")
    updated_at: datetime = Field(description="When the document metadata was last updated")


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""

    documents: list[DocumentMetadata] = Field(
        default_factory=list,
        description="List of known documents",
    )
    total: int = Field(description="Total number of documents")


class DocumentUploadResponse(BaseModel):
    """Response after a successful POST /documents."""

    document_id: str = Field(description="Stable document identifier")
    title: str = Field(description="Document title or file name")
    status: DocumentStatus = Field(description="Initial processing status")
    message: str = Field(description="Human-readable status message")


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------

class IngestionJobResponse(BaseModel):
    """Response for a single ingestion job."""

    job_id: str = Field(description="Unique job identifier")
    document_id: str = Field(description="Document this job belongs to")
    status: JobStatus = Field(description="Job execution status")
    chunk_count: int | None = Field(
        default=None,
        description="Number of chunks produced (null if job not yet finished)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the job failed",
    )
    started_at: datetime = Field(description="When the job started")
    finished_at: datetime | None = Field(
        default=None,
        description="When the job finished (null if still running)",
    )


class JobListResponse(BaseModel):
    """Response for GET /documents/{document_id}/jobs."""

    jobs: list[IngestionJobResponse] = Field(
        default_factory=list,
        description="Ingestion jobs for this document",
    )


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: frozenset[str] = frozenset([".pdf", ".docx", ".html", ".htm", ".md"])
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB


class UploadValidationError(BaseModel):
    """Validation error for a single file in a multipart upload."""

    filename: str = Field(description="Name of the file that failed validation")
    error: str = Field(description="Human-readable reason for the failure")


class UploadErrorResponse(BaseModel):
    """Response when one or more files fail validation."""

    error: ErrorDetail
    failed_files: list[UploadValidationError] = Field(
        default_factory=list,
        description="Per-file validation failures",
    )


def validate_file_extension(filename: str) -> str:
    """Validate *filename* has a supported extension; raise ValueError if not."""
    import os
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext!r}. "
            f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return filename
