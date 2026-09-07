"""Unit tests for the hybrid retriever."""

from __future__ import annotations

from typing import Any

from app.retrieval.hybrid import HybridRetriever, _bm25_hit_to_result, _vector_hit_to_result
from app.retrieval.models import RetrievalResult, RetrieverType


def _result(chunk_id: str, score: float, retriever: RetrieverType) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        text=f"text for {chunk_id}",
        score=score,
        source=f"{chunk_id.split(':')[0]}.md",
        retriever=retriever,
        rank=0,
    )

class _FakeEmbedder:
    """Fake embedder that returns a fixed vector."""

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        # Return a deterministic 4-element vector
        return [[0.1, 0.2, 0.3, 0.4] for _ in queries]

    def embed_documents(self, texts: list[str]) -> list[list[list[float]]]:
        return [[[0.1, 0.2, 0.3, 0.4]] for _ in texts]

class _FakeVectorStore:
    """Fake vector store returning configurable hits."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filter_document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        self.search_calls.append(
            {
                "query_vector": query_vector,
                "top_k": top_k,
                "filter_document_ids": filter_document_ids,
                "score_threshold": score_threshold,
            }
        )
        return self._hits

class _FakeBM25Indexer:
    """Fake BM25 indexer returning configurable hits."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filter_document_ids: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        self.search_calls.append(
            {"query": query, "top_k": top_k, "filter_document_ids": filter_document_ids}
        )
        return self._hits

class TestHybridRetriever:
    def test_retrieve_fuses_vector_and_bm25(self) -> None:
        """retrieve() combines vector and BM25 results via RRF."""
        vec_hits = [
            {
                "id": "a:0",
                "score": 0.9,
                "payload": {
                    "text": "text a",
                    "document_id": "a",
                    "source": "a.md",
                    "page_number": 1,
                    "section": "Intro",
                },
            },
            {
                "id": "b:0",
                "score": 0.8,
                "payload": {
                    "text": "text b",
                    "document_id": "b",
                    "source": "b.md",
                    "page_number": 1,
                    "section": "Intro",
                },
            },
        ]
        bm25_hits = [
            {
                "id": "b:0",
                "score": 5.0,
                "payload": {
                    "text": "text b",
                    "document_id": "b",
                    "source": "b.md",
                    "page_number": 1,
                    "section": "Intro",
                },
            },
            {
                "id": "c:0",
                "score": 4.0,
                "payload": {
                    "text": "text c",
                    "document_id": "c",
                    "source": "c.md",
                    "page_number": 1,
                    "section": "Intro",
                },
            },
        ]
        hybrid = HybridRetriever(
            settings=None,
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore(vec_hits),
            bm25_indexer=_FakeBM25Indexer(bm25_hits),
        )
        # Override settings
        hybrid._settings = hybrid._settings.model_copy(
            update={
                "vector_top_k": 10,
                "bm25_top_k": 10,
                "hybrid_top_k": 5,
                "rrf_k": 60,
            }
        )

        results = hybrid.retrieve("test query")

        assert len(results) <= 5
        # b:0 appears in both retrievers, so it should be first (highest RRF score)
        chunk_ids = [r.chunk_id for r in results]
        assert "b:0" in chunk_ids
        # All results should be HYBRID retriever type
        for r in results:
            assert r.retriever == RetrieverType.HYBRID

    def test_retrieve_empty_when_both_retrievers_return_nothing(self) -> None:
        """Both retrievers returning empty yields empty list."""
        hybrid = HybridRetriever(
            settings=None,
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([]),
            bm25_indexer=_FakeBM25Indexer([]),
        )
        hybrid._settings = hybrid._settings.model_copy(
            update={"vector_top_k": 10, "bm25_top_k": 10, "hybrid_top_k": 5}
        )
        assert hybrid.retrieve("query") == []

    def test_retrieve_with_document_filter(self) -> None:
        """filter_document_ids is passed to both retrievers."""
        hybrid = HybridRetriever(
            settings=None,
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([]),
            bm25_indexer=_FakeBM25Indexer([]),
        )
        hybrid._settings = hybrid._settings.model_copy(
            update={"vector_top_k": 10, "bm25_top_k": 10, "hybrid_top_k": 5}
        )
        hybrid.retrieve("query", filter_document_ids=["doc1", "doc2"])

        # Check vector store received the filter
        assert len(hybrid._vector.search_calls) == 1
        assert hybrid._vector.search_calls[0]["filter_document_ids"] == ["doc1", "doc2"]
        # Check BM25 received the filter
        assert len(hybrid._bm25.search_calls) == 1
        assert hybrid._bm25.search_calls[0]["filter_document_ids"] == ["doc1", "doc2"]

    def test_retrieve_respects_top_k_override(self) -> None:
        """Passing top_k to retrieve() overrides hybrid_top_k."""
        hybrid = HybridRetriever(
            settings=None,
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([
                {"id": f"a:{i}", "score": 1.0 - i * 0.01,
                 "payload": {"text": f"text {i}", "document_id": "a", "source": "a.md"}}
                for i in range(20)
            ]),
            bm25_indexer=_FakeBM25Indexer([]),
        )
        hybrid._settings = hybrid._settings.model_copy(
            update={"vector_top_k": 20, "bm25_top_k": 10, "hybrid_top_k": 20}
        )
        results = hybrid.retrieve("query", top_k=3)
        assert len(results) == 3

    def test_retrieve_handles_one_empty_retriever(self) -> None:
        """One retriever returning empty still yields results from the other."""
        hybrid = HybridRetriever(
            settings=None,
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([
                {"id": "a:0", "score": 0.9,
                 "payload": {"text": "text a", "document_id": "a", "source": "a.md"}}
            ]),
            bm25_indexer=_FakeBM25Indexer([]),
        )
        hybrid._settings = hybrid._settings.model_copy(
            update={"vector_top_k": 10, "bm25_top_k": 10, "hybrid_top_k": 5}
        )
        results = hybrid.retrieve("query")
        assert len(results) == 1
        assert results[0].chunk_id == "a:0"

class TestResultConversionHelpers:
    def test_bm25_hit_to_result(self) -> None:
        hit = {
            "id": "doc:5",
            "score": 3.5,
            "payload": {
                "text": "some text",
                "document_id": "doc",
                "source": "guide.md",
                "page_number": 3,
                "section": "Setup",
            },
        }
        result = _bm25_hit_to_result(hit, rank=1)
        assert result.chunk_id == "doc:5"
        assert result.score == 3.5
        assert result.text == "some text"
        assert result.retriever == RetrieverType.BM25
        assert result.rank == 1
        assert "bm25" in result.metadata["contributing_retrievers"]

    def test_vector_hit_to_result(self) -> None:
        hit = {
            "id": "doc:5",
            "score": 0.85,
            "payload": {
                "text": "some text",
                "document_id": "doc",
                "source": "guide.md",
                "page_number": 3,
                "section": "Setup",
            },
        }
        result = _vector_hit_to_result(hit, rank=0)
        assert result.chunk_id == "doc:5"
        assert result.score == 0.85
        assert result.text == "some text"
        assert result.retriever == RetrieverType.VECTOR
        assert result.rank == 0
        assert "vector" in result.metadata["contributing_retrievers"]
