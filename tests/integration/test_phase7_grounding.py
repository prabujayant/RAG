"""Integration tests for the full grounding pipeline."""

from __future__ import annotations

import pytest
from app.generation.client import MockLLM
from app.generation.schemas import (
    AnswerResponse,
    Citation,
    CitationStatus,
    GroundingStatus,
)
from app.grounding.grounding_validator import GroundingConfig, GroundingValidator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_citations() -> list[Citation]:
    return [
        Citation(
            citation_id="[C1]",
            chunk_id="auth-guide:0",
            text="Access tokens expire after 60 minutes and must be refreshed.",
            page_number=3,
            section="Token Lifecycle",
        ),
        Citation(
            citation_id="[C2]",
            chunk_id="auth-guide:1",
            text="Refresh tokens are valid for 30 days and can be used once.",
            page_number=3,
            section="Token Lifecycle",
        ),
        Citation(
            citation_id="[C3]",
            chunk_id="rate-limits:0",
            text="The API allows 1000 requests per minute per client ID.",
            page_number=1,
            section="Overview",
        ),
    ]

@pytest.fixture
def fully_supported_response(sample_citations: list[Citation]) -> AnswerResponse:
    return AnswerResponse(
        answer="Access tokens expire after 60 minutes [C1]. "
        "Refresh tokens last 30 days [C2].",
        citations=sample_citations[:2],
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=0.95,
        refused=False,
        total_latency_ms=150.0,
        model="openai/gpt-4o-mini",
    )

@pytest.fixture
def partially_supported_response(sample_citations: list[Citation]) -> AnswerResponse:
    return AnswerResponse(
        answer="Access tokens expire after 60 minutes [C1]. "
        "Wrong fact here [C2].",
        citations=sample_citations,
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=0.8,
        refused=False,
        total_latency_ms=150.0,
        model="openai/gpt-4o-mini",
    )

@pytest.fixture
def unsupported_response(sample_citations: list[Citation]) -> AnswerResponse:
    return AnswerResponse(
        answer="Tokens expire in 30 minutes [C1].",
        citations=[sample_citations[0]],
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=0.9,
        refused=False,
        total_latency_ms=100.0,
        model="openai/gpt-4o-mini",
    )

@pytest.fixture
def unanswerable_response() -> AnswerResponse:
    return AnswerResponse(
        answer="I don't have enough evidence to answer this.",
        citations=[],
        grounded=False,
        grounding_status=GroundingStatus.REFUSED,
        confidence=0.0,
        refused=True,
        refused_reason="No supporting evidence.",
        total_latency_ms=50.0,
        model="openai/gpt-4o-mini",
    )

@pytest.fixture
def invalid_citation_response(sample_citations: list[Citation]) -> AnswerResponse:
    return AnswerResponse(
        answer="Unauthorized access is never permitted [C99]. "
        "Access tokens expire after 60 minutes [C1].",
        citations=[sample_citations[0]],  # Only C1 exists
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=0.7,
        refused=False,
        total_latency_ms=100.0,
        model="openai/gpt-4o-mini",
    )

@pytest.fixture
def ambiguous_response() -> AnswerResponse:
    """Answer whose deterministic validation is ambiguous (PARTIALLY_SUPPORTED)."""
    return AnswerResponse(
        answer="Access tokens expire [C1].",
        citations=[
            Citation(
                citation_id="[C1]",
                chunk_id="auth-guide:0",
                text="Tokens used for authentication expire after some time.",
                page_number=3,
                section="Token Lifecycle",
            )
        ],
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=0.8,
        refused=False,
        total_latency_ms=100.0,
        model="openai/gpt-4o-mini",
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGroundingPipelineFullySupported:
    def test_fully_supported_answer_is_grounded(
        self,
        fully_supported_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(fully_supported_response)

        assert result.grounding_status == GroundingStatus.GROUNDED
        assert result.grounded is True
        assert len(result.claims) == 2
        assert all(c.status == CitationStatus.SUPPORTED for c in result.claims)
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_answer_preserved(
        self,
        fully_supported_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(fully_supported_response)

        assert result.answer == fully_supported_response.answer
        assert result.model == fully_supported_response.model
        assert result.total_latency_ms == fully_supported_response.total_latency_ms

class TestGroundingPipelinePartiallySupported:
    def test_partially_supported_answer(
        self,
        partially_supported_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(partially_supported_response)

        assert result.grounding_status == GroundingStatus.PARTIALLY_GROUNDED
        assert result.grounded is False
        assert any(c.status == CitationStatus.SUPPORTED for c in result.claims)
        assert any(c.status == CitationStatus.UNSUPPORTED for c in result.claims)

class TestGroundingPipelineUnsupported:
    def test_unsupported_answer(
        self,
        unsupported_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(unsupported_response)

        assert result.grounding_status == GroundingStatus.UNGROUNDED
        assert result.grounded is False
        assert all(c.status == CitationStatus.UNSUPPORTED for c in result.claims)

class TestGroundingPipelineRefused:
    def test_refused_answer_stays_refused(
        self,
        unanswerable_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(unanswerable_response)

        assert result.grounding_status == GroundingStatus.REFUSED
        assert result.grounded is False
        assert result.confidence == 0.0
        assert result.refused is True

class TestGroundingPipelineInvalidCitation:
    def test_unknown_citation_id(
        self,
        invalid_citation_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(invalid_citation_response)

        # Unknown C99 should be marked unsupported
        assert result.grounding_status == GroundingStatus.PARTIALLY_GROUNDED
        assert result.grounded is False

class TestGroundingPipelineLLMJudge:
    def test_llm_judge_validates_ambiguous_case(
        self,
        ambiguous_response: AnswerResponse,
    ) -> None:
        mock_llm = MockLLM(
            response_text='{"status": "supported", "score": 0.92, '
            '"reason": "Directly supported."}',
        )
        config = GroundingConfig(use_llm_judge=True)
        validator = GroundingValidator(config=config, llm_client=mock_llm)
        result = validator.validate(ambiguous_response)

        # LLM was called for the ambiguous claim
        assert len(mock_llm.calls) >= 1
        # LLM confirmed support -> grounded
        assert result.grounding_status == GroundingStatus.GROUNDED
        assert result.grounded is True

    def test_llm_judge_failure_falls_back_to_deterministic(
        self,
        ambiguous_response: AnswerResponse,
    ) -> None:
        mock_llm = MockLLM(response_text="not json at all!!!")
        config = GroundingConfig(use_llm_judge=True)
        validator = GroundingValidator(config=config, llm_client=mock_llm)

        # Should not raise, falls back to deterministic PARTIALLY_SUPPORTED,
        # which (with no fully SUPPORTED claim) aggregates to UNGROUNDED
        result = validator.validate(ambiguous_response)
        assert result.grounding_status == GroundingStatus.UNGROUNDED
        assert result.grounded is False

    def test_llm_judge_not_called_when_disabled(
        self,
        fully_supported_response: AnswerResponse,
    ) -> None:
        mock_llm = MockLLM(response_text='{"status": "supported", "score": 0.9}')
        config = GroundingConfig(use_llm_judge=False)
        validator = GroundingValidator(config=config, llm_client=mock_llm)
        validator.validate(fully_supported_response)

        assert len(mock_llm.calls) == 0

class TestGroundingPipelineConfidence:
    def test_confidence_in_range(self, fully_supported_response: AnswerResponse) -> None:
        validator = GroundingValidator()
        result = validator.validate(fully_supported_response)

        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_none_for_refused(self, unanswerable_response: AnswerResponse) -> None:
        validator = GroundingValidator()
        result = validator.validate(unanswerable_response)

        assert result.confidence == 0.0

class TestGroundingPipelineEndToEnd:
    def test_full_pipeline_produces_valid_response(
        self,
        fully_supported_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(fully_supported_response)

        # All required fields present
        assert result.answer is not None
        assert result.grounded is not None
        assert result.grounding_status is not None
        assert result.confidence is not None
        # citations and claims lists
        assert isinstance(result.citations, list)
        assert isinstance(result.claims, list)

    def test_no_supported_answer_can_be_marked_grounded(
        self,
        unsupported_response: AnswerResponse,
    ) -> None:
        validator = GroundingValidator()
        result = validator.validate(unsupported_response)

        # Important: an unsupported answer must NOT be grounded
        assert result.grounded is False
        assert result.grounding_status != GroundingStatus.GROUNDED
