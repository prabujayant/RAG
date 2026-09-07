"""Unit tests for app.evaluation.citation_metrics."""

from __future__ import annotations

from app.evaluation.citation_metrics import (
    citation_completeness,
    citation_correctness,
    compute_citation_metrics,
    unsupported_claim_rate,
)


class TestCitationCorrectness:
    def test_empty_answer(self) -> None:
        from app.generation.schemas import AnswerResponse
        resp = AnswerResponse(answer="", citations=[], claims=[], grounding_status=None)
        assert citation_correctness(resp, set()) == 0.0

    def test_all_cited_are_relevant(self) -> None:
        from app.generation.schemas import AnswerResponse, Citation
        citations = [
            Citation(citation_id="C1", chunk_id="c1", text="text"),
            Citation(citation_id="C2", chunk_id="c2", text="text"),
        ]
        resp = AnswerResponse(answer="Answer", citations=citations, claims=[], grounding_status=None)
        relevant = {"c1", "c2", "c3"}
        assert citation_correctness(resp, relevant) == 1.0

    def test_none_cited_are_relevant(self) -> None:
        from app.generation.schemas import AnswerResponse, Citation
        citations = [
            Citation(citation_id="C1", chunk_id="c5", text="text"),
            Citation(citation_id="C2", chunk_id="c6", text="text"),
        ]
        resp = AnswerResponse(answer="Answer", citations=citations, claims=[], grounding_status=None)
        relevant = {"c1", "c2"}
        assert citation_correctness(resp, relevant) == 0.0

    def test_partial_citation_correctness(self) -> None:
        from app.generation.schemas import AnswerResponse, Citation
        citations = [
            Citation(citation_id="C1", chunk_id="c1", text="text"),
            Citation(citation_id="C2", chunk_id="c5", text="text"),
        ]
        resp = AnswerResponse(answer="Answer", citations=citations, claims=[], grounding_status=None)
        relevant = {"c1", "c2"}
        # 1 of 2 cited are relevant
        assert citation_correctness(resp, relevant) == 0.5

class TestCitationCompleteness:
    def test_all_relevant_cited(self) -> None:
        from app.generation.schemas import AnswerResponse, Citation
        citations = [
            Citation(citation_id="C1", chunk_id="c1", text="text"),
            Citation(citation_id="C2", chunk_id="c2", text="text"),
        ]
        resp = AnswerResponse(answer="Answer", citations=citations, claims=[], grounding_status=None)
        relevant = {"c1", "c2"}
        assert citation_completeness(resp, relevant) == 1.0

    def test_none_of_relevant_cited(self) -> None:
        from app.generation.schemas import AnswerResponse, Citation
        citations = [
            Citation(citation_id="C1", chunk_id="c5", text="text"),
        ]
        resp = AnswerResponse(answer="Answer", citations=citations, claims=[], grounding_status=None)
        relevant = {"c1", "c2"}
        assert citation_completeness(resp, relevant) == 0.0

class TestComputeCitationMetrics:
    def test_answerable_with_correct_citations(self) -> None:
        citations = [
            {"chunk_id": "c1"}, {"chunk_id": "c2"},
        ]
        result = compute_citation_metrics(
            question="What is Python?",
            answer="It is a programming language.",
            citations=citations,
            expected_chunk_ids=["c1", "c2", "c3"],
            answerable=True,
        )
        assert result["citation_correctness"] == 1.0
        assert result["citation_completeness"] == 2.0 / 3.0
        assert result["citation_precision"] == 1.0

    def test_unanswerable_correct_refusal(self) -> None:
        # When both expected_chunk_ids and citations are empty, the early-return
        # dict is returned; refusal_correctness is 0.0 in that case.
        result = compute_citation_metrics(
            question="What is the secret to eternal life?",
            answer="",
            citations=[],
            expected_chunk_ids=[],
            answerable=False,
        )
        assert result["citation_correctness"] == 0.0
        assert result["refusal_correctness"] == 0.0

    def test_empty_cited_and_relevant(self) -> None:
        result = compute_citation_metrics(
            question="What is Python?",
            answer="A programming language.",
            citations=[],
            expected_chunk_ids=[],
            answerable=True,
        )
        # With no citations and no expected chunks, all metrics are 0
        assert result["citation_correctness"] == 0.0
        assert result["grounded_answer_rate"] == 0.0

class TestUnsupportedClaimRate:
    def test_no_claims(self) -> None:
        from app.generation.schemas import AnswerResponse
        resp = AnswerResponse(answer="No claims here.", citations=[], claims=[], grounding_status=None)
        assert unsupported_claim_rate(resp) == 0.0

    def test_all_supported(self) -> None:
        from app.generation.schemas import AnswerClaim, AnswerResponse, CitationStatus
        claims = [
            AnswerClaim(claim="Claim 1", status=CitationStatus.SUPPORTED),
            AnswerClaim(claim="Claim 2", status=CitationStatus.SUPPORTED),
        ]
        resp = AnswerResponse(answer="Answer", citations=[], claims=claims, grounding_status=None)
        assert unsupported_claim_rate(resp) == 0.0

    def test_some_unsupported(self) -> None:
        from app.generation.schemas import AnswerClaim, AnswerResponse, CitationStatus
        claims = [
            AnswerClaim(claim="Claim 1", status=CitationStatus.SUPPORTED),
            AnswerClaim(claim="Claim 2", status=CitationStatus.UNSUPPORTED),
            AnswerClaim(claim="Claim 3", status=CitationStatus.UNSUPPORTED),
        ]
        resp = AnswerResponse(answer="Answer", citations=[], claims=claims, grounding_status=None)
        assert unsupported_claim_rate(resp) == 2.0 / 3.0
