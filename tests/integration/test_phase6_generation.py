"""Integration tests for the full generation pipeline."""

from __future__ import annotations

import json

import pytest
from app.generation.client import MockLLM
from app.generation.schemas import GroundingStatus
from app.generation.service import GenerationService
from app.retrieval.models import RetrievalResult, RetrieverType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_candidates() -> list[RetrievalResult]:
    """Realistic evidence list for integration tests."""
    return [
        RetrievalResult(
            chunk_id="auth-guide:0",
            document_id="auth-guide",
            text="Access tokens expire after 60 minutes and must be refreshed.",
            score=0.95,
            source="authentication-guide.md",
            retriever=RetrieverType.HYBRID,
            rank=0,
            section="Token Lifecycle",
            page_number=3,
        ),
        RetrievalResult(
            chunk_id="auth-guide:1",
            document_id="auth-guide",
            text="Refresh tokens are valid for 30 days and can be used once.",
            score=0.90,
            source="authentication-guide.md",
            retriever=RetrieverType.HYBRID,
            rank=1,
            section="Token Lifecycle",
            page_number=3,
        ),
        RetrievalResult(
            chunk_id="rate-limits:0",
            document_id="rate-limits",
            text="The API allows 1000 requests per minute per client ID.",
            score=0.85,
            source="rate-limits.md",
            retriever=RetrieverType.VECTOR,
            rank=2,
            section="Overview",
            page_number=1,
        ),
    ]

_DEFAULT_MOCK_RESPONSE = (
    '{"answer": "default", "citations": [], '
    '"confidence": 0.5, "refused": false, "refused_reason": null}'
)

@pytest.fixture
def mock_llm() -> MockLLM:
    """Fresh MockLLM for each test."""
    return MockLLM(response_text=_DEFAULT_MOCK_RESPONSE)

@pytest.fixture
def service(mock_llm: MockLLM) -> GenerationService:
    """GenerationService wired to a MockLLM."""
    return GenerationService(client=mock_llm)

# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestGenerationPipeline:
    """Full pipeline: evidence → service → structured answer."""

    def test_answer_with_citations(
        self,
        service: GenerationService,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Service correctly parses a well-formed answer with citations."""
        service._client.configure(response_text=json.dumps({
            "answer": "Access tokens expire after 60 minutes. [C1]",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "auth-guide:0",
                    "text": "Access tokens expire after 60 minutes.",
                },
            ],
            "confidence": 0.97,
            "refused": False,
            "refused_reason": None,
        }))
        resp = service.generate("How long are access tokens valid?", sample_candidates)

        assert "expire after 60 minutes" in resp.answer
        assert len(resp.citations) == 1
        assert resp.citations[0].chunk_id == "auth-guide:0"
        assert resp.citations[0].citation_id == "[C1]"
        assert resp.confidence == 0.97
        assert resp.refused is False
        assert resp.grounding_status == GroundingStatus.GROUNDED
        assert resp.model == "mock/test"

    def test_multiple_chunks_cited(
        self,
        service: GenerationService,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Service handles multiple citations across chunks."""
        service._client.configure(response_text=json.dumps({
            "answer": "Access tokens last 60 minutes [C1] and refresh tokens last 30 days [C2].",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "auth-guide:0",
                    "text": "Access tokens expire after 60 minutes.",
                },
                {
                    "citation_id": "[C2]",
                    "chunk_id": "auth-guide:1",
                    "text": "Refresh tokens are valid for 30 days.",
                },
            ],
            "confidence": 0.95,
            "refused": False,
            "refused_reason": None,
        }))
        resp = service.generate("token lifetimes?", sample_candidates)

        assert len(resp.citations) == 2
        citation_ids = {c.citation_id for c in resp.citations}
        assert "[C1]" in citation_ids
        assert "[C2]" in citation_ids

    def test_unanswerable_question_refused(
        self,
        service: GenerationService,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Service correctly refuses when evidence doesn't support the question."""
        service._client.configure(response_text=json.dumps({
            "answer": "I don't have enough information to answer this question.",
            "citations": [],
            "confidence": 0.05,
            "refused": True,
            "refused_reason": "No supporting evidence found for the asked topic.",
        }))
        resp = service.generate("What is the capital of France?", sample_candidates)

        assert resp.refused is True
        assert resp.grounding_status == GroundingStatus.REFUSED
        assert resp.confidence == 0.05
        assert len(resp.citations) == 0

    def test_malformed_llm_output_yields_refusal(
        self,
        service: GenerationService,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Service gracefully handles non-JSON LLM output without crashing."""
        service._client.configure(response_text="the model output was not JSON formatted properly!!!")
        resp = service.generate("any question?", sample_candidates)

        assert resp.refused is True
        assert resp.grounding_status == GroundingStatus.REFUSED
        assert resp.refused_reason is not None

    def test_missing_evidence_handled(self, service: GenerationService) -> None:
        """Service handles an empty candidates list gracefully."""
        service._client.configure(response_text=json.dumps({
            "answer": "I don't have enough information.",
            "citations": [],
            "confidence": 0.0,
            "refused": True,
            "refused_reason": "No evidence provided.",
        }))
        resp = service.generate("What is the secret key?", [])

        assert resp.refused is True
        assert resp.grounding_status == GroundingStatus.REFUSED

    def test_prompts_receive_correct_evidence_format(
        self,
        service: GenerationService,
        mock_llm: MockLLM,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Prompts contain correctly formatted evidence with [C1], [C2], ... markers."""
        service._client.configure(response_text=json.dumps({
            "answer": "OK",
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service.generate("token lifetime?", sample_candidates)

        call = mock_llm.calls[0]
        user_prompt = call["user_prompt"]
        assert "[C1]" in user_prompt
        assert "[C2]" in user_prompt
        assert "[C3]" in user_prompt
        assert "Access tokens expire after 60 minutes" in user_prompt
        assert "Refresh tokens are valid for 30 days" in user_prompt
        assert "1000 requests per minute" in user_prompt
        assert "## Evidence" in user_prompt
        assert "## Question" in user_prompt

    def test_temperature_passed_to_client(
        self,
        service: GenerationService,
        mock_llm: MockLLM,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Custom temperature is forwarded to the LLM client."""
        service._client.configure(response_text=json.dumps({
            "answer": "OK",
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service.generate("q?", sample_candidates, temperature=0.1)
        assert mock_llm.calls[0]["temperature"] == 0.1

    def test_llm_error_returns_safe_response(
        self,
        service: GenerationService,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """LLM client exceptions are caught and converted to refusal responses."""
        service._client.configure(raise_on_call=True)
        resp = service.generate("q?", sample_candidates)

        assert resp.refused is True
        assert resp.grounding_status == GroundingStatus.REFUSED
        assert "Generation failed" in (resp.refused_reason or "")

    def test_unknown_citation_ids_are_dropped(
        self,
        service: GenerationService,
        mock_llm: MockLLM,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Citations referencing non-existent IDs are dropped with a debug log."""
        service._client.configure(response_text=json.dumps({
            "answer": "Based on [C1] and [C99].",
            "citations": [
                {"citation_id": "[C1]", "chunk_id": "auth-guide:0", "text": "valid"},
                {"citation_id": "[C99]", "chunk_id": "auth-guide:0", "text": "invalid"},
            ],
            "confidence": 0.8,
            "refused": False,
            "refused_reason": None,
        }))
        resp = service.generate("q?", sample_candidates)

        assert len(resp.citations) == 1
        assert resp.citations[0].citation_id == "[C1]"

    def test_unknown_chunk_ids_are_dropped(
        self,
        service: GenerationService,
        mock_llm: MockLLM,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Citations referencing unknown chunk_ids are silently dropped."""
        service._client.configure(response_text=json.dumps({
            "answer": "Based on [C1] and [C2].",
            "citations": [
                {"citation_id": "[C1]", "chunk_id": "auth-guide:0", "text": "valid"},
                {"citation_id": "[C2]", "chunk_id": "nonexistent:99", "text": "invalid"},
            ],
            "confidence": 0.8,
            "refused": False,
            "refused_reason": None,
        }))
        resp = service.generate("q?", sample_candidates)

        assert len(resp.citations) == 1
        assert resp.citations[0].chunk_id == "auth-guide:0"

    def test_latency_recorded(
        self,
        service: GenerationService,
        sample_candidates: list[RetrievalResult],
    ) -> None:
        """Response includes end-to-end latency in milliseconds."""
        service._client.configure(response_text=json.dumps({
            "answer": "OK",
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        resp = service.generate("q?", sample_candidates)

        assert resp.total_latency_ms is not None
        assert resp.total_latency_ms >= 0

class TestGenerationServiceLazyClient:
    """Tests for lazy client initialization."""

    def test_lazy_client_resolved_on_first_use(self) -> None:
        """Service builds the real client only when generate() is called."""
        service = GenerationService()  # no client passed
        # Client property should create OpenRouterClient lazily
        assert service._client is None  # not created yet
        # Accessing .client property should create it
        _ = service.client
        assert service._client is not None
