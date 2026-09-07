"""Unit tests for API schemas."""

from __future__ import annotations

import pytest
from app.api.schemas.common import (
    ComponentStatus,
    DependencyStatus,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    ReadinessResponse,
)
from app.api.schemas.documents import (
    ALLOWED_EXTENSIONS,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
    IngestionJobResponse,
    JobStatus,
    validate_file_extension,
)
from app.api.schemas.query import QueryRequest, QueryResponse
from app.generation.schemas import GroundingStatus
from pydantic import ValidationError


class TestHealthSchemas:
    def test_health_response_valid(self) -> None:
        resp = HealthResponse(status=HealthStatus.HEALTHY, version="1.0.0")
        assert resp.status == HealthStatus.HEALTHY
        assert resp.version == "1.0.0"

    def test_health_status_degraded(self) -> None:
        resp = HealthResponse(status=HealthStatus.DEGRADED, version="1.0.0")
        assert resp.status == HealthStatus.DEGRADED

    def test_dependency_status_healthy(self) -> None:
        dep = DependencyStatus(
            name="postgresql",
            status=ComponentStatus.HEALTHY,
            latency_ms=1.5,
        )
        assert dep.status == ComponentStatus.HEALTHY
        assert dep.latency_ms == 1.5
        assert dep.error is None

    def test_dependency_status_unhealthy_no_error_leak(self) -> None:
        dep = DependencyStatus(
            name="postgresql",
            status=ComponentStatus.UNHEALTHY,
            error="connection refused",
        )
        assert dep.error == "connection refused"

    def test_readiness_response(self) -> None:
        deps = [
            DependencyStatus(name="postgresql", status=ComponentStatus.HEALTHY),
            DependencyStatus(name="qdrant", status=ComponentStatus.UNHEALTHY, error="timeout"),
        ]
        resp = ReadinessResponse(status=HealthStatus.UNHEALTHY, dependencies=deps)
        assert resp.status == HealthStatus.UNHEALTHY
        assert len(resp.dependencies) == 2

class TestErrorSchemas:
    def test_error_response_structure(self) -> None:
        err = ErrorResponse(
            error=ErrorDetail(
                code="DOCUMENT_NOT_FOUND",
                message="No document found",
                request_id="req-123",
            )
        )
        assert err.error.code == "DOCUMENT_NOT_FOUND"
        assert err.error.request_id == "req-123"

    def test_error_detail_without_request_id(self) -> None:
        err = ErrorResponse(
            error=ErrorDetail(code="BAD_REQUEST", message="invalid input")
        )
        assert err.error.request_id is None

class TestDocumentSchemas:
    def test_document_metadata(self) -> None:
        from datetime import datetime
        meta = DocumentMetadata(
            document_id="abc123",
            title="Test Doc",
            status=DocumentStatus.READY,
            chunk_count=42,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert meta.document_id == "abc123"
        assert meta.chunk_count == 42

    def test_job_status_enum_values(self) -> None:
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.SUCCESS.value == "success"
        assert JobStatus.FAILED.value == "failed"

    def test_document_upload_response(self) -> None:
        resp = DocumentUploadResponse(
            document_id="doc-456",
            title="my-file.pdf",
            status=DocumentStatus.PENDING,
            message="Document registered",
        )
        assert resp.status == DocumentStatus.PENDING

    def test_ingestion_job_response(self) -> None:
        from datetime import datetime
        job = IngestionJobResponse(
            job_id="job-789",
            document_id="doc-456",
            status=JobStatus.SUCCESS,
            chunk_count=10,
            error=None,
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
        )
        assert job.chunk_count == 10
        assert job.error is None

class TestQuerySchemas:
    def test_query_request_valid(self) -> None:
        req = QueryRequest(question="How do I rotate API keys?")
        assert req.question == "How do I rotate API keys?"
        assert req.top_k is None
        assert req.temperature is None

    def test_query_request_with_options(self) -> None:
        req = QueryRequest(question="What is the rate limit?", top_k=5, temperature=0.7)
        assert req.top_k == 5
        assert req.temperature == 0.7

    def test_query_request_too_short(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")  # min_length=5

    def test_query_request_temperature_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="What is the rate limit?", temperature=3.0)  # max 2.0

    def test_query_response_structure(self) -> None:
        resp = QueryResponse(
            answer="Access tokens expire after 60 minutes.",
            citations=[],
            claims=[],
            grounded=True,
            grounding_status=GroundingStatus.GROUNDED,
            confidence=0.95,
            refused=False,
            latency_ms=450.0,
            model="gpt-4o",
        )
        assert resp.grounded is True
        assert resp.grounding_status == GroundingStatus.GROUNDED

class TestFileValidation:
    def test_validate_file_extension_pdf(self) -> None:
        assert validate_file_extension("document.pdf") == "document.pdf"

    def test_validate_file_extension_docx(self) -> None:
        assert validate_file_extension("report.docx") == "report.docx"

    def test_validate_file_extension_case_insensitive(self) -> None:
        assert validate_file_extension("doc.PDF") == "doc.PDF"

    def test_validate_file_extension_unsupported(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_file_extension("malware.exe")
        assert "Unsupported file type" in str(exc_info.value)

    def test_allowed_extensions_defined(self) -> None:
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".docx" in ALLOWED_EXTENSIONS
        assert ".html" in ALLOWED_EXTENSIONS
        assert ".md" in ALLOWED_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS
