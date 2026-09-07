"""Unit tests for the generation domain schemas (Phase 2)."""

from __future__ import annotations

from app.generation.schemas import (
    AnswerClaim,
    AnswerResponse,
    Citation,
    CitationStatus,
    GroundingStatus,
)


def test_citation_status_values() -> None:
    assert CitationStatus.SUPPORTED.value == "supported"
    assert CitationStatus.PARTIALLY_SUPPORTED.value == "partially_supported"
    assert CitationStatus.UNSUPPORTED.value == "unsupported"

def test_grounding_status_values() -> None:
    assert GroundingStatus.GROUNDED.value == "grounded"
    assert GroundingStatus.PARTIALLY_GROUNDED.value == "partially_grounded"
    assert GroundingStatus.UNGROUNDED.value == "ungrounded"
    assert GroundingStatus.REFUSED.value == "refused"

def test_citation_defaults() -> None:
    c = Citation(citation_id="[C1]", chunk_id="doc-1:0", text="Access tokens expire.")
    assert c.page_number is None
    assert c.section is None

def test_citation_round_trip() -> None:
    c = Citation(
        citation_id="[C2]",
        chunk_id="doc-1:5",
        text="OAuth flows vary.",
        page_number=12,
        section="OAuth",
    )
    assert Citation.from_dict(c.to_dict()) == c

def test_answer_claim_defaults() -> None:
    claim = AnswerClaim(claim="Tokens expire after 60 minutes.")
    assert claim.citation_ids == []
    assert claim.status is CitationStatus.UNSUPPORTED
    assert claim.reason is None

def test_answer_claim_round_trip() -> None:
    claim = AnswerClaim(
        claim="Tokens expire after 60 minutes.",
        citation_ids=["[C1]"],
        status=CitationStatus.SUPPORTED,
        reason=None,
    )
    d = claim.to_dict()
    assert d["status"] == "supported"
    restored = AnswerClaim.from_dict(d)
    assert restored == claim

def test_answer_response_defaults() -> None:
    resp = AnswerResponse(answer="No evidence available.")
    assert resp.grounded is False
    assert resp.grounding_status is GroundingStatus.UNGROUNDED
    assert resp.confidence is None
    assert resp.refused is False
    assert resp.citations == []
    assert resp.claims == []

def test_answer_response_round_trip() -> None:
    resp = AnswerResponse(
        answer="Rate limits reset hourly [C1].",
        citations=[
            Citation(
                citation_id="[C1]",
                chunk_id="doc-1:0",
                text="Rate limits reset hourly.",
            )
        ],
        claims=[
            AnswerClaim(
                claim="Rate limits reset hourly.",
                citation_ids=["[C1]"],
                status=CitationStatus.SUPPORTED,
            )
        ],
        grounded=True,
        grounding_status=GroundingStatus.GROUNDED,
        confidence=0.95,
        refused=False,
        refused_reason=None,
        total_latency_ms=123.4,
        model="openai/gpt-4o-mini",
    )
    restored = AnswerResponse.from_dict(resp.to_dict())
    assert restored == resp
    assert restored.grounded is True
    assert restored.model == "openai/gpt-4o-mini"

def test_answer_response_refused() -> None:
    resp = AnswerResponse(
        answer="I cannot answer from the given context.",
        grounded=False,
        grounding_status=GroundingStatus.REFUSED,
        refused=True,
        refused_reason="Insufficient evidence.",
    )
    d = resp.to_dict()
    assert d["refused"] is True
    assert d["grounding_status"] == "refused"
    restored = AnswerResponse.from_dict(d)
    assert restored.refused is True
    assert restored.grounding_status is GroundingStatus.REFUSED
