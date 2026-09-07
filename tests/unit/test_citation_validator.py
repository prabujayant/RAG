"""Unit tests for the deterministic citation validator."""

from __future__ import annotations

import pytest
from app.grounding.citation_validator import (
    CitationValidator,
    _check_negation_mismatch,
    _check_number_mismatch,
    _compute_overlap,
    _extract_numbers_and_codes,
    _has_negation,
    _key_phrase_overlap,
    _tokenize,
)


class TestTokenize:
    def test_lowercases(self) -> None:
        tokens = _tokenize("Access TOKENS Expire")
        assert "access" in tokens
        assert "tokens" in tokens
        assert "expire" in tokens

    def test_removes_stopwords(self) -> None:
        tokens = _tokenize("the a and or")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "and" not in tokens

    def test_removes_single_char(self) -> None:
        tokens = _tokenize("a b c")
        assert "a" not in tokens
        assert "b" not in tokens

    def test_keeps_numbers(self) -> None:
        tokens = _tokenize("60 minutes 30 days")
        assert "60" in tokens
        assert "minutes" in tokens
        assert "30" in tokens
        assert "days" in tokens

class TestExtractNumbersAndCodes:
    def test_extracts_integers(self) -> None:
        nums = _extract_numbers_and_codes("60 minutes and 30 days")
        assert "60" in nums
        assert "30" in nums

    def test_extracts_decimals(self) -> None:
        nums = _extract_numbers_and_codes("version 2.5 and 1.0")
        assert "2.5" in nums
        assert "1.0" in nums

    def test_extracts_alphanumeric_codes(self) -> None:
        codes = _extract_numbers_and_codes("AES-256 and TLS-1.2")
        assert "AES-256" in codes
        assert "TLS-1.2" in codes

    def test_empty_string(self) -> None:
        assert _extract_numbers_and_codes("") == set()

class TestHasNegation:
    def test_no_negation(self) -> None:
        assert _has_negation("Access tokens expire after 60 minutes.") is False

    def test_negation_not(self) -> None:
        assert _has_negation("Access tokens do not expire.") is True

    def test_negation_no(self) -> None:
        assert _has_negation("No refresh tokens are valid.") is True

    def test_negation_never(self) -> None:
        assert _has_negation("This never expires.") is True

    def test_contraction_isnt(self) -> None:
        assert _has_negation("It isn't valid.") is True

    def test_contraction_doesnt(self) -> None:
        assert _has_negation("It doesn't expire.") is True

class TestComputeOverlap:
    def test_full_overlap(self) -> None:
        tokens1 = {"access", "tokens", "expire", "60", "minutes"}
        tokens2 = {"access", "tokens", "expire", "60", "minutes"}
        assert _compute_overlap(tokens1, tokens2) == 1.0

    def test_partial_overlap(self) -> None:
        tokens1 = {"access", "tokens", "expire", "60", "minutes"}
        tokens2 = {"access", "tokens", "expire"}
        assert _compute_overlap(tokens1, tokens2) == 3 / 5

    def test_overlap_single_shared_token(self) -> None:
        tokens1 = {"access", "tokens"}
        tokens2 = {"refresh", "tokens"}
        # 1 shared token out of 2 claim tokens = 0.5
        assert _compute_overlap(tokens1, tokens2) == 0.5

    def test_empty_claim(self) -> None:
        assert _compute_overlap(set(), {"a", "b"}) == 0.0

class TestKeyPhraseOverlap:
    def test_exact_phrase(self) -> None:
        score = _key_phrase_overlap(
            "Access tokens expire",
            "Access tokens expire after 60 minutes.",
        )
        assert score == 1.0

    def test_partial_phrase_match(self) -> None:
        score = _key_phrase_overlap(
            "Access tokens expire after 60 minutes",
            "Tokens expire in 60 minutes",
        )
        # Should have some overlap
        assert score > 0.0

    def test_no_phrase_match(self) -> None:
        score = _key_phrase_overlap(
            "Access tokens expire",
            "Refresh tokens are valid",
        )
        assert score == 0.0

class TestCheckNumberMismatch:
    def test_no_mismatch(self) -> None:
        assert _check_number_mismatch(
            "Expires in 60 minutes",
            "Token expires in 60 minutes.",
        ) is False

    def test_mismatch_different_numbers(self) -> None:
        assert _check_number_mismatch(
            "Expires in 60 minutes",
            "Token expires in 30 minutes.",
        ) is True

    def test_claim_has_numbers_evidence_doesnt(self) -> None:
        assert _check_number_mismatch(
            "AES-256 encryption is used",
            "Encryption is used.",
        ) is True

    def test_no_numbers(self) -> None:
        assert _check_number_mismatch(
            "Tokens expire",
            "Tokens expire.",
        ) is False

class TestNegationMismatch:
    def test_both_affirmative(self) -> None:
        assert _check_negation_mismatch(
            "Tokens expire after 60 minutes",
            "Tokens expire after 60 minutes.",
        ) is False

    def test_both_negative(self) -> None:
        assert _check_negation_mismatch(
            "Tokens do not expire",
            "Tokens do not expire.",
        ) is False

    def test_claim_negative_evidence_affirmative(self) -> None:
        assert _check_negation_mismatch(
            "Tokens do not expire",
            "Tokens expire.",
        ) is True

    def test_claim_affirmative_evidence_negative(self) -> None:
        assert _check_negation_mismatch(
            "Tokens expire",
            "Tokens do not expire.",
        ) is True

class TestCitationValidatorValidate:
    def test_supported_claim(self) -> None:
        validator = CitationValidator()
        result = validator.validate(
            "Access tokens expire after 60 minutes.",
            "Access tokens expire after 60 minutes and must be refreshed.",
        )
        assert result.status.value == "supported"
        assert result.score >= 0.65

    def test_unsupported_claim_wrong_numbers(self) -> None:
        validator = CitationValidator()
        result = validator.validate(
            "Access tokens expire after 60 minutes.",
            "Access tokens expire after 30 minutes.",
        )
        assert result.status.value == "unsupported"
        assert result.score == 0.0

    def test_unsupported_claim_no_overlap(self) -> None:
        validator = CitationValidator()
        result = validator.validate(
            "Access tokens expire after 60 minutes.",
            "The weather is nice today.",
        )
        assert result.status.value == "unsupported"
        assert result.score == 0.0

    def test_partially_supported(self) -> None:
        validator = CitationValidator()
        result = validator.validate(
            "Access tokens expire after 60 minutes.",
            "Access tokens expire.",
        )
        # Partial overlap but not enough for full support
        assert result.status.value in ("partially_supported", "unsupported")

    def test_negation_mismatch(self) -> None:
        validator = CitationValidator()
        result = validator.validate(
            "Tokens expire after 60 minutes.",
            "Tokens do not expire.",
        )
        assert result.status.value == "unsupported"
        assert "Negation mismatch" in result.reason

    def test_empty_claim(self) -> None:
        validator = CitationValidator()
        result = validator.validate("", "Some evidence.")
        assert result.status.value == "unsupported"
        assert result.score == 0.0

    def test_empty_evidence(self) -> None:
        validator = CitationValidator()
        result = validator.validate("Some claim.", "")
        assert result.status.value == "unsupported"
        assert result.score == 0.0

    def test_short_claim(self) -> None:
        validator = CitationValidator()
        result = validator.validate("Expires.", "Something expires.")
        assert result.status.value == "partially_supported"
        assert "too short" in result.reason

    def test_configurable_thresholds(self) -> None:
        validator = CitationValidator(supported_threshold=0.9, partial_threshold=0.5)
        result = validator.validate(
            "Access tokens expire after 60 minutes.",
            "Access tokens expire after 60 minutes and are refreshed.",
        )
        # With high threshold, might be partially supported instead of supported
        assert result.status.value in ("supported", "partially_supported")

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError, match="supported_threshold must be >= partial_threshold"):
            CitationValidator(supported_threshold=0.3, partial_threshold=0.6)
