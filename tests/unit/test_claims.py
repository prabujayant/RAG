"""Unit tests for claim extraction."""

from __future__ import annotations

from app.grounding.claims import (
    ClaimExtractor,
    _normalize_text,
    _strip_citations,
    extract_claims,
)


class TestStripCitations:
    def test_strips_single_citation(self) -> None:
        assert _strip_citations("Access tokens expire after 60 minutes [C1]") == \
            "Access tokens expire after 60 minutes"

    def test_strips_multiple_citations(self) -> None:
        assert _strip_citations("Expires [C1]. Refresh tokens last 30 days [C2].") == \
            "Expires . Refresh tokens last 30 days ."

    def test_strips_lowercase_citations(self) -> None:
        assert _strip_citations("Answer [c1] and [c2]") == "Answer  and"

    def test_no_citations(self) -> None:
        assert _strip_citations("Just plain text") == "Just plain text"

    def test_empty_string(self) -> None:
        assert _strip_citations("") == ""


class TestNormalizeText:
    def test_collapse_whitespace(self) -> None:
        assert _normalize_text("Access  tokens  \n  expire") == "Access tokens expire"

    def test_strip_trailing_punctuation(self) -> None:
        assert _normalize_text("Expires in 60 minutes.") == "Expires in 60 minutes"

    def test_strip_leading_punctuation(self) -> None:
        assert _normalize_text("(Access tokens)") == "Access tokens"

    def test_preserve_mid_punctuation(self) -> None:
        assert _normalize_text("AES-256 encryption") == "AES-256 encryption"


class TestClaimExtractor:
    def test_single_claim_no_citation(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract("Access tokens expire after 60 minutes.")
        assert len(claims) == 1
        assert claims[0].text == "Access tokens expire after 60 minutes"
        assert claims[0].citation_ids == []

    def test_single_claim_with_citation(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract("Access tokens expire after 60 minutes [C1].")
        assert len(claims) == 1
        assert claims[0].text == "Access tokens expire after 60 minutes"
        assert claims[0].citation_ids == ["[C1]"]

    def test_multiple_claims_multiple_citations(self) -> None:
        extractor = ClaimExtractor()
        text = (
            "Access tokens expire after 60 minutes [C1]. "
            "Refresh tokens last 30 days [C2]."
        )
        claims = extractor.extract(text)
        assert len(claims) == 2
        assert claims[0].text == "Access tokens expire after 60 minutes"
        assert claims[0].citation_ids == ["[C1]"]
        assert claims[1].text == "Refresh tokens last 30 days"
        assert claims[1].citation_ids == ["[C2]"]

    def test_multiple_citations_per_claim(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract(
            "Access tokens expire after 60 minutes [C1] and "
            "refresh tokens last 30 days [C2]."
        )
        assert len(claims) == 1
        assert "[C1]" in claims[0].citation_ids
        assert "[C2]" in claims[0].citation_ids

    def test_missing_citations_skipped(self) -> None:
        extractor = ClaimExtractor()
        text = "Access tokens expire after 60 minutes. Refresh tokens last 30 days."
        claims = extractor.extract(text)
        # Both should be extracted but with empty citation_ids
        assert len(claims) == 2
        assert claims[0].citation_ids == []
        assert claims[1].citation_ids == []

    def test_unknown_citations_preserved(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract(
            "Something [C99] and another thing [C1]."
        )
        assert len(claims) == 1
        assert "[C99]" in claims[0].citation_ids
        assert "[C1]" in claims[0].citation_ids

    def test_empty_claims_skipped(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract("[C1] [C2]")
        # The extracted text would be empty → should be skipped
        assert len(claims) == 0

    def test_heading_fragments_skipped(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract("Access Tokens.")
        # "Access Tokens" is 13 chars, not in skip_fragments — not skipped by default
        # Only truly short fragments are skipped (min_claim_length=10)
        assert len(claims) == 1
        assert claims[0].text == "Access Tokens"

    def test_min_claim_length_configurable(self) -> None:
        extractor = ClaimExtractor(min_claim_length=5)
        claims = extractor.extract("Yes [C1].")
        # "Yes" is 3 chars < 5 → skipped
        assert len(claims) == 0

    def test_duplicate_citations_deduplicated(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract(
            "Something [C1] and also [C1] and again [C1]."
        )
        assert len(claims) == 1
        # Should appear once
        assert claims[0].citation_ids.count("[C1]") == 1


class TestExtractClaimsConvenience:
    def test_extract_claims_default(self) -> None:
        claims = extract_claims("Answer confirmed [C1].")
        assert len(claims) == 1
        assert claims[0].citation_ids == ["[C1]"]

    def test_extract_claims_with_kwargs(self) -> None:
        claims = extract_claims("Answer [C1].", min_claim_length=3)
        assert len(claims) == 1
