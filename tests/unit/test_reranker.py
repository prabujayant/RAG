"""Unit tests for the cross-encoder reranker."""

from __future__ import annotations

from app.retrieval.models import RetrievalResult, RetrieverType
from app.retrieval.reranker import Reranker


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        text=f"text for {chunk_id}",
        score=score,
        source=f"{chunk_id.split(':')[0]}.md",
        retriever=RetrieverType.HYBRID,
        rank=0,
    )

class _FakeCrossEncoder:
    """Fake cross-encoder that returns scores reversed from the input order."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs, show_progress_bar=False):  # noqa: ARG002
        return self._scores[: len(pairs)]

class TestReranker:
    def test_rerank_updates_scores_and_order(self) -> None:
        """Reranking re-orders candidates by cross-encoder scores."""
        candidates = [_result("a:0", 0.9), _result("b:0", 0.8), _result("c:0", 0.7)]
        fake_model = _FakeCrossEncoder(scores=[0.1, 0.9, 0.5])  # b > c > a
        reranker = Reranker(settings=None, model=fake_model)  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": True}
        )
        result = reranker.rerank("query", candidates)
        assert result[0].chunk_id == "b:0"
        assert result[1].chunk_id == "c:0"
        assert result[2].chunk_id == "a:0"

    def test_rerank_preserves_top_k(self) -> None:
        """Only top_k results are returned."""
        candidates = [
            _result("a:0", 0.9),
            _result("b:0", 0.8),
            _result("c:0", 0.7),
            _result("d:0", 0.6),
        ]
        fake_model = _FakeCrossEncoder(scores=[0.4, 0.8, 0.2, 0.1])  # b > a > c > d
        reranker = Reranker(settings=None, model=fake_model)  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": True, "rerank_top_k": 2}
        )
        result = reranker.rerank("query", candidates)
        assert len(result) == 2
        assert result[0].chunk_id == "b:0"
        assert result[1].chunk_id == "a:0"

    def test_disabled_reranker_returns_original_order(self) -> None:
        """When enable_reranker=False, original ordering is returned."""
        candidates = [_result("a:0", 0.9), _result("b:0", 0.8), _result("c:0", 0.7)]
        reranker = Reranker(settings=None)  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": False}
        )
        result = reranker.rerank("query", candidates)
        assert [r.chunk_id for r in result] == ["a:0", "b:0", "c:0"]

    def test_empty_candidates_returns_empty(self) -> None:
        """Empty candidate list returns an empty list."""
        reranker = Reranker(settings=None)  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": True}
        )
        assert reranker.rerank("query", []) == []

    def test_model_load_failure_returns_original(self) -> None:
        """If the model fails to load, original ordering is returned."""
        candidates = [_result("a:0", 0.9), _result("b:0", 0.8)]

        class _BrokenModel:
            def predict(self, pairs, show_progress_bar=False):  # noqa: ARG002
                raise RuntimeError("model unavailable")

        reranker = Reranker(settings=None, model=_BrokenModel())  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": True}
        )
        result = reranker.rerank("query", candidates)
        assert [r.chunk_id for r in result] == ["a:0", "b:0"]

    def test_rerank_updates_rank(self) -> None:
        """After reranking, ranks reflect the new ordering."""
        candidates = [_result("a:0", 0.9), _result("b:0", 0.8)]
        fake_model = _FakeCrossEncoder(scores=[0.2, 0.9])  # b > a
        reranker = Reranker(settings=None, model=fake_model)  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": True}
        )
        result = reranker.rerank("query", candidates)
        assert result[0].rank == 0
        assert result[1].rank == 1

    def test_rerank_attaches_rerank_score(self) -> None:
        """Original score is preserved and new rerank_score is added to metadata."""
        candidates = [_result("a:0", 0.9)]
        fake_model = _FakeCrossEncoder(scores=[0.77])
        reranker = Reranker(settings=None, model=fake_model)  # type: ignore[arg-type]
        reranker._settings = reranker._settings.model_copy(
            update={"enable_reranker": True}
        )
        result = reranker.rerank("query", candidates)
        assert result[0].metadata["rerank_score"] == 0.77
        assert result[0].score == 0.77
