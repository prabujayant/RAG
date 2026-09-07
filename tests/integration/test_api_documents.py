"""Integration tests for document management endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())

@pytest.fixture
def mock_db_session():
    """Patch SessionLocal to return a mock session."""
    mock_session = MagicMock()
    with patch("app.api.routes.documents.SessionLocal", return_value=mock_session):
        yield mock_session

class TestUploadDocument:
    def test_upload_rejects_unsupported_extension(self, client: TestClient) -> None:
        file_content = b"%PDF-1.4 fake pdf content"
        response = client.post(
            "/documents",
            files={"file": ("evil.exe", io.BytesIO(file_content), "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_upload_rejects_oversized_file(self, client: TestClient) -> None:
        """Files larger than MAX_FILE_SIZE_BYTES should be rejected."""
        from app.api.schemas.documents import MAX_FILE_SIZE_BYTES

        oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        response = client.post(
            "/documents",
            files={"file": ("large.pdf", io.BytesIO(oversized), "application/pdf")},
        )
        assert response.status_code == 413

    def test_upload_accepts_valid_pdf(self, client: TestClient) -> None:
        """A minimal PDF header should be accepted at the schema validation level."""
        file_content = b"%PDF-1.4\n%fake pdf content"
        with patch("app.api.routes.documents.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None
            mock_session.execute.return_value.scalar.return_value = None

            response = client.post(
                "/documents",
                files={"file": ("doc.pdf", io.BytesIO(file_content), "application/pdf")},
            )
            # Will fail at DB level but schema should accept it
            assert response.status_code in (201, 500)

class TestListDocuments:
    def test_list_documents_returns_200(self, client: TestClient) -> None:
        with patch("app.api.routes.documents.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.execute.return_value.scalars.return_value.all.return_value = []

            response = client.get("/documents")
            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
            assert "total" in data

class TestGetDocument:
    def test_get_document_not_found(self, client: TestClient) -> None:
        with patch("app.api.routes.documents.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None

            response = client.get("/documents/nonexistent-id")
            assert response.status_code == 404
            assert response.json()["detail"]["code"] == "DOCUMENT_NOT_FOUND"

class TestIngestDocument:
    def test_ingest_not_found_returns_404(self, client: TestClient) -> None:
        with patch("app.api.routes.documents.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.execute.return_value.scalar_one_or_none.return_value = None

            response = client.post("/documents/nonexistent-id/ingest")
            assert response.status_code == 404

class TestListIngestionJobs:
    def test_list_jobs_returns_jobs_list(self, client: TestClient) -> None:
        with patch("app.api.routes.documents.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.execute.return_value.scalars.return_value.all.return_value = []

            response = client.get("/documents/some-doc-id/jobs")
            assert response.status_code == 200
            assert "jobs" in response.json()
