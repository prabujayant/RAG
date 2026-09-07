"""Unit tests for app.evaluation.retrieval_metrics."""

from __future__ import annotations

from app.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1", "c2", "c3"}
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1", "c2", "c3", "c4"}
        # 3 of 4 relevant are retrieved → 3/4
        assert recall_at_k(retrieved, relevant, k=5) == 0.75

    def test_recall_at_k_cutoff(self) -> None:
        retrieved = ["c1", "c4", "c2", "c3"]
        relevant = {"c1", "c2", "c3"}
        # c2 and c3 are beyond k=2
        assert recall_at_k(retrieved, relevant, k=2) == 1.0 / 3.0

    def test_no_relevant(self) -> None:
        assert recall_at_k(["c1", "c2"], set(), k=5) == 0.0

    def test_empty_retrieved(self) -> None:
        assert recall_at_k([], {"c1"}, k=5) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k(["c1"], {"c1"}, k=0) == 0.0

class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        retrieved = ["c1", "c2"]
        relevant = {"c1", "c2", "c3"}
        # Both c1 and c3 are in the relevant set, so precision is 2/2 = 1.0
        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_partial_precision(self) -> None:
        retrieved = ["c1", "c3"]
        relevant = {"c1", "c2", "c3"}
        # Both c1 and c3 are in the relevant set, so precision is 2/2 = 1.0
        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_no_relevant_in_retrieved(self) -> None:
        assert precision_at_k(["c4", "c5"], {"c1", "c2"}, k=2) == 0.0

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"c1"}, k=2) == 0.0

class TestHitRate:
    def test_hit(self) -> None:
        assert hit_rate(["c1", "c2", "c3"], {"c2"}) == 1.0

    def test_miss(self) -> None:
        assert hit_rate(["c1", "c2"], {"c3"}) == 0.0

    def test_hit_at_k_cutoff(self) -> None:
        assert hit_rate(["c1", "c3", "c2"], {"c3"}, k=2) == 1.0

    def test_miss_at_k_cutoff(self) -> None:
        assert hit_rate(["c1", "c2", "c3"], {"c3"}, k=2) == 0.0

class TestReciprocalRank:
    def test_first_hit(self) -> None:
        assert reciprocal_rank(["c1", "c2"], {"c1"}) == 1.0

    def test_second_hit(self) -> None:
        assert reciprocal_rank(["c1", "c2"], {"c2"}) == 0.5

    def test_no_hit(self) -> None:
        assert reciprocal_rank(["c1", "c2"], {"c3"}) == 0.0

    def test_no_relevant(self) -> None:
        assert reciprocal_rank(["c1"], set()) == 0.0

class TestMRR:
    def test_single_query_hit(self) -> None:
        queries = [(["c1", "c2"], {"c2"})]
        assert mrr(queries) == 0.5

    def test_multiple_queries(self) -> None:
        queries = [
            (["c1", "c2"], {"c1"}),   # rank 1 → 1.0
            (["c1", "c2"], {"c2"}),   # rank 2 → 0.5
        ]
        assert mrr(queries) == 0.75

    def test_empty_queries(self) -> None:
        assert mrr([]) == 0.0

class TestNDCGAtK:
    def test_perfect_ndcg(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1", "c2", "c3"}
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_ideal_order_matters(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1", "c2", "c3"}
        # All relevant at top: DCG = 1/log3 + 1/log4 + 1/log5
        # IDCG = same since they're already in ideal order
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_worst_order(self) -> None:
        """When all relevant items are at the bottom of the list."""
        retrieved = ["c4", "c5", "c1", "c2", "c3"]
        relevant = {"c1", "c2", "c3"}
        ndcg = ndcg_at_k(retrieved, relevant, k=5)
        # Only 3 relevant items in positions 3,4,5
        # DCG = 1/log4 + 1/log5 + 1/log6
        # IDCG = 1/log2 + 1/log3 + 1/log4
        assert 0.0 < ndcg < 1.0

    def test_partial_retrieval(self) -> None:
        retrieved = ["c1"]
        relevant = {"c1", "c2", "c3"}
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        # DCG = 1/log2
        # IDCG = 1/log2 + 1/log3 + 1/log4
        assert 0.0 < ndcg <= 1.0

    def test_k_zero(self) -> None:
        assert ndcg_at_k(["c1"], {"c1"}, k=0) == 0.0

class TestComputeRetrievalMetrics:
    def test_all_metrics_returned(self) -> None:
        result = compute_retrieval_metrics(
            retrieved_ids=["c1", "c2", "c3", "c4", "c5"],
            expected_ids=["c1", "c3", "c5"],
            k=5,
        )
        assert set(result.keys()) == {"recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_rate"}

    def test_no_retrieved(self) -> None:
        result = compute_retrieval_metrics([], ["c1"], k=5)
        assert all(v == 0.0 for v in result.values())

    def test_no_expected(self) -> None:
        result = compute_retrieval_metrics(["c1"], [], k=5)
        assert result["recall_at_k"] == 0.0
