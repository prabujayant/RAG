"""Unit tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalResult, RetrieverType


def _result(chunk_id: str, score: float, retriever: RetrieverType, rank: int) -> RetrievalResult:
    """Helper: create a RetrievalResult with minimal required fields."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        text=f"text for {chunk_id}",
        score=score,
        source=f"{chunk_id.split(':')[0]}.md",
        retriever=retriever,
        rank=rank,
    )

class TestReciprocalRankFusion:
    def test_single_retriever(self) -> None:
        """Fusion with one retriever returns it with updated metadata."""
        results = {
            RetrieverType.VECTOR: [
                _result("a:0", 0.9, RetrieverType.VECTOR, 0),
                _result("a:1", 0.8, RetrieverType.VECTOR, 1),
            ],
        }
        fused = reciprocal_rank_fusion(results, rrf_k=60)
        assert len(fused) == 2
        assert fused[0].chunk_id == "a:0"
        assert fused[0].retriever == RetrieverType.HYBRID
        assert fused[1].chunk_id == "a:1"

    def test_duplicate_chunks_from_different_retrievers(self) -> None:
        """Same chunk appearing in both retrievers appears only once with combined score."""
        vec_results = [
            _result("a:0", 0.9, RetrieverType.VECTOR, 0),
            _result("b:0", 0.8, RetrieverType.VECTOR, 1),
        ]
        bm25_results = [
            _result("a:0", 5.0, RetrieverType.BM25, 0),  # duplicate of a:0
            _result("c:0", 4.0, RetrieverType.BM25, 1),
        ]
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=60,
        )
        assert len(fused) == 3  # deduplicated
        chunk_ids = [r.chunk_id for r in fused]
        assert "a:0" in chunk_ids
        assert "b:0" in chunk_ids
        assert "c:0" in chunk_ids

    def test_different_rankings(self) -> None:
        """Result ranked first in BM25 but second in vector gets correct RRF score."""
        vec_results = [
            _result("a:0", 0.9, RetrieverType.VECTOR, 0),
            _result("b:0", 0.8, RetrieverType.VECTOR, 1),
        ]
        bm25_results = [
            _result("b:0", 5.0, RetrieverType.BM25, 0),  # b:0 is first in BM25
            _result("a:0", 4.0, RetrieverType.BM25, 1),  # a:0 is second in BM25
        ]
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=60,
        )
        # b:0 gets 1/(60+0) + 1/(60+1) = 2 RRF contributions
        # a:0 gets 1/(60+1) + 1/(60+0) = 2 RRF contributions
        # Same RRF score at rrf_k=60; a:0 wins (lower chunk_id alphabetically)
        assert fused[0].chunk_id == "a:0"
        assert fused[1].chunk_id == "b:0"

    def test_empty_results_from_one_retriever(self) -> None:
        """Empty results from one retriever are handled gracefully."""
        vec_results = [
            _result("a:0", 0.9, RetrieverType.VECTOR, 0),
        ]
        bm25_results: list[RetrievalResult] = []
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=60,
        )
        assert len(fused) == 1
        assert fused[0].chunk_id == "a:0"

    def test_all_empty(self) -> None:
        """Empty results from all retrievers returns empty list."""
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: [], RetrieverType.BM25: []},
            rrf_k=60,
        )
        assert fused == []

    def test_deterministic_ordering_equal_scores(self) -> None:
        """When two chunks have the same RRF score, chunk_id determines order."""
        # Both chunks get same RRF: a:0 gets rank0 from vec, b:0 gets rank0 from bm25
        # Both get 1/60 from one retriever, 0 from the other → same score
        vec_results = [
            _result("a:0", 0.9, RetrieverType.VECTOR, 0),
            _result("b:0", 0.8, RetrieverType.VECTOR, 1),
        ]
        bm25_results = [
            _result("b:0", 5.0, RetrieverType.BM25, 0),
            _result("a:0", 4.0, RetrieverType.BM25, 1),
        ]
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=60,
        )
        # Same RRF score, a:0 < b:0 alphabetically → a:0 first
        assert fused[0].chunk_id == "a:0"
        assert fused[1].chunk_id == "b:0"

    def test_rrf_k_affects_ranking(self) -> None:
        """Higher rrf_k reduces rank differences between retrievers."""
        vec_results = [
            _result("a:0", 0.9, RetrieverType.VECTOR, 0),
            _result("b:0", 0.8, RetrieverType.VECTOR, 1),
        ]
        bm25_results = [
            _result("b:0", 5.0, RetrieverType.BM25, 0),
            _result("a:0", 4.0, RetrieverType.BM25, 1),
        ]
        # With small rrf_k, rank differences are amplified — a:0 still leads
        # due to consistent rank advantage in both retrievers.
        fused_small_k = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=1,
        )
        # With large rrf_k, both retrievers contribute nearly equally.
        # a:0 wins on lower chunk_id tiebreaker.
        fused_large_k = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=1000,
        )
        assert fused_small_k[0].chunk_id == "a:0"
        assert fused_large_k[0].chunk_id == "a:0"

    def test_negative_rrf_k_raises(self) -> None:
        """Negative rrf_k is rejected."""
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            reciprocal_rank_fusion({RetrieverType.VECTOR: []}, rrf_k=-1)

    def test_zero_rrf_k_raises(self) -> None:
        """Zero rrf_k is rejected (division by zero in formula)."""
        with pytest.raises(ValueError, match="rrf_k must be positive"):
            reciprocal_rank_fusion({RetrieverType.VECTOR: []}, rrf_k=0)

    def test_metadata_preserved(self) -> None:
        """Result metadata (page_number, section, source) is preserved after fusion."""
        result = RetrievalResult(
            chunk_id="a:0",
            document_id="a",
            text="text for a:0",
            score=0.9,
            source="guide.md",
            page_number=3,
            section="Overview",
            retriever=RetrieverType.VECTOR,
            rank=0,
        )
        fused = reciprocal_rank_fusion({RetrieverType.VECTOR: [result]}, rrf_k=60)
        assert fused[0].source == "guide.md"
        assert fused[0].page_number == 3
        assert fused[0].section == "Overview"

    def test_contributing_retrievers_tracked(self) -> None:
        """The 'contributing_retrievers' metadata tracks which retrievers found a chunk."""
        vec_result = _result("a:0", 0.9, RetrieverType.VECTOR, 0)
        bm25_result = _result("a:0", 5.0, RetrieverType.BM25, 0)  # same chunk
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: [vec_result], RetrieverType.BM25: [bm25_result]},
            rrf_k=60,
        )
        assert "vector" in fused[0].metadata["contributing_retrievers"]
        assert "bm25" in fused[0].metadata["contributing_retrievers"]
