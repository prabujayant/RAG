"""Integration tests for the query endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.generation.schemas import AnswerResponse, Citation, GroundingStatus
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())

@pytest.fixture
def mock_retriever():
    """Return a mock HybridRetriever that returns a single retrieval result."""
    from app.retrieval.models import RetrievalResult, RetrieverType

    mock = MagicMock()
    mock.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Access tokens expire after 60 minutes.",
            score=0.95,
            source="auth-guide.md",
            page_number=1,
            section="Authentication",
            retriever=RetrieverType.HYBRID,
            rank=0,
            metadata={},
        )
    ]
    return mock

@pytest.fixture
def mock_generation_service():
    """Return a MockLLM pre-loaded with a grounded answer."""
    mock = MagicMock()
    mock.generate.return_value = AnswerResponse(
        answer="Access tokens expire after 60 minutes [C1].",
        citations=[
            Citation(
                citation_id="[C1]",
                chunk_id="chunk-1",
                text="Access tokens expire after 60 minutes.",
                page_number=1,
                section="Authentication",
            )
        ],
        claims=[],
        grounded=True,
        grounding_status=GroundingStatus.GROUNDED,
        confidence=0.95,
        refused=False,
        refused_reason=None,
        total_latency_ms=150.0,
        model="mock-llm",
    )
    return mock

class TestQueryEndpoint:
    def test_query_returns_200_with_valid_request(
        self,
        client: TestClient,
        mock_retriever,
        mock_generation_service,
    ) -> None:
        with patch(
            "app.api.routes.query.HybridRetriever", return_value=mock_retriever
        ), patch(
            "app.api.routes.query.GenerationService", return_value=mock_generation_service
        ):
            response = client.post(
                "/query",
                json={"question": "How long do access tokens last?"},
            )
            assert response.status_code == 200

    def test_query_returns_grounded_answer(
        self,
        client: TestClient,
        mock_retriever,
        mock_generation_service,
    ) -> None:
        with patch(
            "app.api.routes.query.HybridRetriever", return_value=mock_retriever
        ), patch(
            "app.api.routes.query.GenerationService", return_value=mock_generation_service
        ):
            response = client.post(
                "/query",
                json={"question": "How long do access tokens last?"},
            )
            data = response.json()
            assert data["grounded"] is True
            assert data["grounding_status"] == GroundingStatus.GROUNDED.value

    def test_query_returns_citations(
        self,
        client: TestClient,
        mock_retriever,
        mock_generation_service,
    ) -> None:
        with patch(
            "app.api.routes.query.HybridRetriever", return_value=mock_retriever
        ), patch(
            "app.api.routes.query.GenerationService", return_value=mock_generation_service
        ):
            response = client.post(
                "/query",
                json={"question": "How long do access tokens last?"},
            )
            data = response.json()
            assert len(data["citations"]) == 1
            assert data["citations"][0]["citation_id"] == "[C1]"

    def test_query_question_too_short(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"question": "Hi"},
        )
        assert response.status_code == 422  # Validation error

    def test_query_missing_question(self, client: TestClient) -> None:
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_no_retrieval_results_returns_empty_answer(
        self, client: TestClient
    ) -> None:
        empty_retriever = MagicMock()
        empty_retriever.retrieve.return_value = []

        with patch(
            "app.api.routes.query.HybridRetriever", return_value=empty_retriever
        ):
            response = client.post(
                "/query",
                json={"question": "What is the secret key?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "No relevant documents were found for your question."
            assert data["grounded"] is False

class TestQueryRequestID:
    def test_query_request_id_returned_in_header(
        self,
        client: TestClient,
        mock_retriever,
        mock_generation_service,
    ) -> None:
        with patch(
            "app.api.routes.query.HybridRetriever", return_value=mock_retriever
        ), patch(
            "app.api.routes.query.GenerationService", return_value=mock_generation_service
        ):
            response = client.post(
                "/query",
                json={"question": "How long do access tokens last?"},
            )
            assert "X-Request-ID" in response.headers

    def test_query_custom_request_id_propagated(
        self,
        client: TestClient,
        mock_retriever,
        mock_generation_service,
    ) -> None:
        custom_id = "my-custom-query-id-abc123"
        with patch(
            "app.api.routes.query.HybridRetriever", return_value=mock_retriever
        ), patch(
            "app.api.routes.query.GenerationService", return_value=mock_generation_service
        ):
            response = client.post(
                "/query",
                json={"question": "How long do access tokens last?"},
                headers={"X-Request-ID": custom_id},
            )
            assert response.headers["X-Request-ID"] == custom_id
