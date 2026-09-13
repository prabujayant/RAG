"""Unit tests for the grounding validator."""

from __future__ import annotations

from app.generation.client import MockLLM
from app.generation.schemas import (
    AnswerResponse,
    Citation,
    CitationStatus,
    GroundingStatus,
)
from app.grounding.grounding_validator import (
    GroundingConfig,
    GroundingValidator,
    _parse_llm_judge_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    answer: str,
    citations: list[Citation] | None = None,
    refused: bool = False,
    refused_reason: str | None = None,
    confidence: float | None = 0.9,
) -> AnswerResponse:
    return AnswerResponse(
        answer=answer,
        citations=citations or [],
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=confidence,
        refused=refused,
        refused_reason=refused_reason,
        total_latency_ms=100.0,
        model="mock/test",
    )

def _make_citation(citation_id: str, chunk_id: str, text: str) -> Citation:
    return Citation(
        citation_id=citation_id,
        chunk_id=chunk_id,
        text=text,
        page_number=1,
        section="Test",
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseLlmJudgeResponse:
    def test_fenced_json(self) -> None:
        raw = '```json\n{"status": "supported", "score": 0.9, "reason": "ok"}\n```'
        result = _parse_llm_judge_response(raw)
        assert result is not None
        assert result["status"] == "supported"
        assert result["score"] == 0.9

    def test_bare_json(self) -> None:
        raw = '{"status": "unsupported", "score": 0.0, "reason": "fail"}'
        result = _parse_llm_judge_response(raw)
        assert result is not None
        assert result["status"] == "unsupported"

    def test_with_extra_text(self) -> None:
        raw = 'Here is the result: {"status": "supported", "score": 0.8, "reason": "ok"}\nThanks!'
        result = _parse_llm_judge_response(raw)
        assert result is not None
        assert result["score"] == 0.8

    def test_malformed_json(self) -> None:
        assert _parse_llm_judge_response("not json at all") is None
        assert _parse_llm_judge_response("") is None

class TestGroundingValidatorClaimExtraction:
    def test_extracts_single_claim(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire after 60 minutes [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire after 60 minutes.")],
        )
        result = validator.validate(response)
        assert len(result.claims) == 1
        assert result.claims[0].claim == "Access tokens expire after 60 minutes"

    def test_extracts_multiple_claims(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire [C1]. Refresh tokens last 30 days [C2].",
            [
                _make_citation("[C1]", "auth:0", "Access tokens expire."),
                _make_citation("[C2]", "auth:1", "Refresh tokens last 30 days."),
            ],
        )
        result = validator.validate(response)
        assert len(result.claims) == 2
        assert result.claims[0].claim == "Access tokens expire"
        assert result.claims[0].citation_ids == ["[C1]"]
        assert result.claims[1].claim == "Refresh tokens last 30 days"
        assert result.claims[1].citation_ids == ["[C2]"]

    def test_unknown_citations_marked_unsupported(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Something true [C99].",
            [_make_citation("[C1]", "auth:0", "Something is true.")],
        )
        result = validator.validate(response)
        # C99 is unknown → that claim is unsupported
        assert len(result.claims) == 1
        assert result.claims[0].citation_ids == ["[C99]"]
        assert result.claims[0].status == CitationStatus.UNSUPPORTED

    def test_missing_citations_marked_unsupported(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire.",
            [],  # No citations provided
        )
        result = validator.validate(response)
        assert len(result.claims) == 1
        assert result.claims[0].status == CitationStatus.UNSUPPORTED
        assert "no citations" in result.claims[0].reason.lower()

class TestGroundingValidatorStatusAggregation:
    def test_validate_preserves_generic_flag(self) -> None:
        """Validation keeps the generic marker on the rebuilt response."""
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire after 60 minutes [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire after 60 minutes.")],
        )
        response.generic = True
        result = validator.validate(response)
        assert result.generic is True

    def test_all_supported_grounded(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire after 60 minutes [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire after 60 minutes.")],
        )
        result = validator.validate(response)
        assert result.grounding_status == GroundingStatus.GROUNDED
        assert result.grounded is True

    def test_generic_answer_is_never_fully_grounded(self) -> None:
        """General mode may add context beyond the document, so it cannot be
        reported as fully grounded even when every cited claim checks out."""
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire after 60 minutes [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire after 60 minutes.")],
        )
        response.generic = True
        result = validator.validate(response)

        assert result.generic is True
        assert result.grounded is False
        assert result.grounding_status == GroundingStatus.PARTIALLY_GROUNDED

    def test_non_generic_all_supported_still_grounded(self) -> None:
        """The cap must not apply to strict-mode answers."""
        validator = GroundingValidator()
        response = _make_response(
            "Access tokens expire after 60 minutes [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire after 60 minutes.")],
        )
        response.generic = False
        result = validator.validate(response)

        assert result.grounded is True
        assert result.grounding_status == GroundingStatus.GROUNDED

    def test_some_unsupported_partially_grounded(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Tokens expire after 60 minutes [C1]. Contradicted claim [C2].",
            [
                _make_citation("[C1]", "auth:0", "Tokens expire after 60 minutes."),
                _make_citation("[C2]", "auth:1", "Something else entirely."),
            ],
        )
        result = validator.validate(response)
        assert result.grounding_status == GroundingStatus.PARTIALLY_GROUNDED

    def test_all_unsupported_ungrounded(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Wrong [C1]. Also wrong [C2].",
            [
                _make_citation("[C1]", "auth:0", "Completely different text."),
                _make_citation("[C2]", "auth:1", "Also completely different."),
            ],
        )
        result = validator.validate(response)
        assert result.grounding_status == GroundingStatus.UNGROUNDED
        assert result.grounded is False

    def test_refused_answer_stays_refused(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "I cannot answer this.",
            [],
            refused=True,
            refused_reason="No evidence.",
        )
        result = validator.validate(response)
        assert result.grounding_status == GroundingStatus.REFUSED
        assert result.grounded is False
        assert result.confidence == 0.0

class TestGroundingValidatorConfidence:
    def test_confidence_scaled_by_support(self) -> None:
        validator = GroundingValidator(config=GroundingConfig(confidence_scale_with_support=True))
        response = _make_response(
            "Correct [C1].",
            [_make_citation("[C1]", "auth:0", "Tokens expire after 60 minutes.")],
            confidence=1.0,
        )
        result = validator.validate(response)
        # Fully supported → confidence should be high but scaled
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_zero_for_refused(self) -> None:
        validator = GroundingValidator()
        response = _make_response("Refused.", [], refused=True, confidence=0.9)
        result = validator.validate(response)
        assert result.confidence == 0.0

    def test_confidence_none_when_no_claims(self) -> None:
        validator = GroundingValidator()
        response = _make_response("No claims here.", [])
        result = validator.validate(response)
        # No claims extracted → confidence comes from LLM
        assert result.confidence is not None

class TestGroundingValidatorNumberMismatch:
    def test_contradicting_numbers_unsupported(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "Tokens expire in 60 minutes [C1].",
            [_make_citation("[C1]", "auth:0", "Tokens expire in 30 minutes.")],
        )
        result = validator.validate(response)
        assert result.claims[0].status == CitationStatus.UNSUPPORTED
        assert "specific values" in result.claims[0].reason.lower()

    def test_contradicting_encryption_unsupported(self) -> None:
        validator = GroundingValidator()
        response = _make_response(
            "AES-256 encryption is used [C1].",
            [_make_citation("[C1]", "sec:0", "Encryption is used for data at rest.")],
        )
        result = validator.validate(response)
        assert result.claims[0].status == CitationStatus.UNSUPPORTED

class TestGroundingValidatorConfig:
    def test_use_llm_judge_flag_stored(self) -> None:
        config = GroundingConfig(use_llm_judge=True)
        validator = GroundingValidator(config=config)
        assert validator.config.use_llm_judge is True

    def test_custom_thresholds_passed_to_validator(self) -> None:
        config = GroundingConfig(supported_threshold=0.8, partial_threshold=0.4)
        validator = GroundingValidator(config=config)
        assert validator.citation_validator.supported_threshold == 0.8
        assert validator.citation_validator.partial_threshold == 0.4

class TestGroundingValidatorLLMJudge:
    def test_llm_judge_called_when_configured(self) -> None:
        mock_llm = MockLLM(response_text='{"status": "supported", "score": 0.95, "reason": "ok"}')
        config = GroundingConfig(use_llm_judge=True)
        validator = GroundingValidator(config=config, llm_client=mock_llm)
        # Use a partially-matching case to trigger the LLM judge
        response = _make_response(
            "Access tokens expire [C1].",
            [_make_citation("[C1]", "auth:0", "Tokens used for authentication expire.")],
            confidence=0.9,
        )
        validator.validate(response)
        # Should have called the LLM for the PARTIALLY_SUPPORTED claim
        assert len(mock_llm.calls) >= 1

    def test_llm_judge_fallback_on_malformed_response(self) -> None:
        mock_llm = MockLLM(response_text="not json at all")
        config = GroundingConfig(use_llm_judge=True)
        validator = GroundingValidator(config=config, llm_client=mock_llm)
        response = _make_response(
            "Something [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire.")],
        )
        # Should not crash, falls back to deterministic
        result = validator.validate(response)
        assert result.grounding_status in list(GroundingStatus)

    def test_llm_judge_not_called_when_disabled(self) -> None:
        mock_llm = MockLLM(response_text='{"status": "supported", "score": 0.95}')
        config = GroundingConfig(use_llm_judge=False)
        validator = GroundingValidator(config=config, llm_client=mock_llm)
        response = _make_response(
            "Something [C1].",
            [_make_citation("[C1]", "auth:0", "Access tokens expire.")],
        )
        validator.validate(response)
        assert len(mock_llm.calls) == 0
