"""Integration tests for Phase 5 retrieval pipeline.

Tests the full flow: query → embed → vector search → BM25 search → RRF fusion
→ cross-encoder reranking → evidence selection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.config.settings import Settings
from app.retrieval.evidence import EvidenceSelector
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.models import RetrievalResult, RetrieverType
from app.retrieval.reranker import Reranker


class _FakeEmbedder:
    """Minimal embedder that returns a fixed embedding vector."""

    dim = 4

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        # Return a deterministic fake embedding for each query
        return [[0.1, 0.2, 0.3, 0.4] for _ in queries]

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Return a deterministic fake embedding
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

class _FakeVectorStore:
    """Fake vector store that returns pre-configured hits."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.search_calls: list[tuple[str, int]] = []

    def search(
        self, query_vector: list[float], top_k: int, filter_document_ids: list[str] | None = None
    ) -> list[dict]:
        self.search_calls.append((str(query_vector), top_k))
        hits = self._hits[:top_k]
        if filter_document_ids:
            hits = [h for h in hits if h["payload"]["document_id"] in filter_document_ids]
        return hits

class _FakeBM25Indexer:
    """Fake BM25 indexer that returns pre-configured hits."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.search_calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filter_document_ids: list[str] | None = None,
    ) -> list[dict]:
        self.search_calls.append((query, top_k))
        hits = self._hits[:top_k]
        if filter_document_ids:
            hits = [h for h in hits if h["payload"]["document_id"] in filter_document_ids]
        return hits

def _result(
    chunk_id: str,
    score: float,
    retriever: RetrieverType,
    rank: int,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":")[0],
        text=f"Text for {chunk_id}",
        score=score,
        source=f"{chunk_id.split(':')[0]}.md",
        retriever=retriever,
        rank=rank,
        metadata={"contributing_retrievers": {retriever.value}},
    )

class TestReciprocalRankFusionIntegration:
    """Test RRF fusion in a realistic multi-retriever scenario."""

    def test_fusion_combines_vector_and_bm25(self) -> None:
        """Vector and BM25 ranks are merged correctly via RRF."""
        vec_results = [
            _result("a:0", 0.9, RetrieverType.VECTOR, 0),
            _result("b:0", 0.8, RetrieverType.VECTOR, 1),
            _result("c:0", 0.7, RetrieverType.VECTOR, 2),
        ]
        bm25_results = [
            _result("b:0", 5.0, RetrieverType.BM25, 0),
            _result("c:0", 4.0, RetrieverType.BM25, 1),
            _result("d:0", 3.0, RetrieverType.BM25, 2),
        ]
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=60,
        )
        chunk_ids = [r.chunk_id for r in fused]
        # All four chunks must appear (deduplicated)
        assert set(chunk_ids) == {"a:0", "b:0", "c:0", "d:0"}
        # b:0 and c:0 appear in both → higher RRF score → first positions
        assert chunk_ids.index("b:0") < chunk_ids.index("a:0")
        assert chunk_ids.index("c:0") < chunk_ids.index("a:0")

    def test_fusion_assigns_correct_contributing_retrievers(self) -> None:
        """Each fused result knows which retrievers found it."""
        vec_results = [_result("a:0", 0.9, RetrieverType.VECTOR, 0)]
        bm25_results = [_result("a:0", 5.0, RetrieverType.BM25, 0)]
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=60,
        )
        assert fused[0].retriever == RetrieverType.HYBRID
        assert "vector" in fused[0].metadata["contributing_retrievers"]
        assert "bm25" in fused[0].metadata["contributing_retrievers"]

class TestHybridRetrieverIntegration:
    """Test the full HybridRetriever pipeline."""

    def test_retrieve_produces_fused_results(self) -> None:
        """retrieve() runs vector + BM25 in parallel and fuses results."""
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
            settings=Settings(),
            embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=_FakeVectorStore(vec_hits),  # type: ignore[arg-type]
            bm25_indexer=_FakeBM25Indexer(bm25_hits),  # type: ignore[arg-type]
        )

        results = hybrid.retrieve(query="test query", top_k=5)

        # All chunks from both retrievers are present (deduplicated)
        chunk_ids = {r.chunk_id for r in results}
        assert chunk_ids == {"a:0", "b:0", "c:0"}
        # All results are HYBRID retriever type
        for r in results:
            assert r.retriever == RetrieverType.HYBRID

    def test_retrieve_respects_filter_document_ids(self) -> None:
        """filter_document_ids restricts results to specified documents."""
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
                "id": "a:1",
                "score": 5.0,
                "payload": {
                    "text": "text a1",
                    "document_id": "a",
                    "source": "a.md",
                    "page_number": 2,
                    "section": "Details",
                },
            },
            {
                "id": "b:1",
                "score": 4.0,
                "payload": {
                    "text": "text b1",
                    "document_id": "b",
                    "source": "b.md",
                    "page_number": 2,
                    "section": "Details",
                },
            },
        ]
        hybrid = HybridRetriever(
            settings=Settings(),
            embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=_FakeVectorStore(vec_hits),  # type: ignore[arg-type]
            bm25_indexer=_FakeBM25Indexer(bm25_hits),  # type: ignore[arg-type]
        )

        # Filter to only document "a"
        results = hybrid.retrieve(query="test", top_k=5, filter_document_ids=["a"])

        # Only "a" chunks should appear
        assert all(r.document_id == "a" for r in results)

    def test_retrieve_with_empty_results(self) -> None:
        """Empty results from both retrievers return empty list."""
        hybrid = HybridRetriever(
            settings=Settings(),
            embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=_FakeVectorStore([]),  # type: ignore[arg-type]
            bm25_indexer=_FakeBM25Indexer([]),  # type: ignore[arg-type]
        )
        results = hybrid.retrieve(query="test", top_k=5)
        assert results == []

class TestRerankerIntegration:
    """Test the cross-encoder reranker."""

    def test_reranker_reorders_candidates(self) -> None:
        """Reranker reorders candidates by cross-encoder relevance score."""
        candidates = [
            RetrievalResult(
                chunk_id="a:0",
                document_id="a",
                text="Authentication is required for all API calls.",
                score=0.9,
                source="auth.md",
                retriever=RetrieverType.HYBRID,
                rank=0,
            ),
            RetrievalResult(
                chunk_id="b:0",
                document_id="b",
                text="The deployment guide covers Kubernetes setup.",
                score=0.8,
                source="deploy.md",
                retriever=RetrieverType.HYBRID,
                rank=1,
            ),
        ]

        # Fake reranker model that reverses the score order
        fake_model = MagicMock()
        fake_model.predict = MagicMock(return_value=[0.1, 0.9])  # b:0 is more relevant

        reranker = Reranker(settings=Settings(), model=fake_model)
        reranked = reranker.rerank(
            query="How to deploy with Kubernetes?",
            candidates=candidates,
            top_k=2,
        )

        # b:0 should now be first (higher cross-encoder score)
        assert reranked[0].chunk_id == "b:0"
        assert reranked[1].chunk_id == "a:0"

    def test_reranker_respects_top_k(self) -> None:
        """Reranker returns at most top_k results."""
        candidates = [
            RetrievalResult(
                chunk_id=f"doc:{i}",
                document_id="doc",
                text=f"Text for doc {i}",
                score=0.9 - i * 0.1,
                source="doc.md",
                retriever=RetrieverType.HYBRID,
                rank=i,
            )
            for i in range(10)
        ]

        fake_model = MagicMock()
        # Assign descending scores so order reverses
        fake_model.predict = MagicMock(return_value=list(range(10)))

        reranker = Reranker(settings=Settings(), model=fake_model)
        result = reranker.rerank(query="test query", candidates=candidates, top_k=3)

        assert len(result) == 3
        assert result[0].chunk_id == "doc:9"
        assert result[1].chunk_id == "doc:8"
        assert result[2].chunk_id == "doc:7"

    def test_reranker_graceful_fallback(self) -> None:
        """Reranker returns original order if model is unavailable."""
        candidates = [
            RetrievalResult(
                chunk_id="a:0",
                document_id="a",
                text="text a",
                score=0.9,
                source="a.md",
                retriever=RetrieverType.HYBRID,
                rank=0,
            ),
            RetrievalResult(
                chunk_id="b:0",
                document_id="b",
                text="text b",
                score=0.8,
                source="b.md",
                retriever=RetrieverType.HYBRID,
                rank=1,
            ),
        ]
        # Reranker with reranking disabled so model is never accessed
        settings = Settings(enable_reranker=False)
        reranker = Reranker(settings=settings, model=None)
        result = reranker.rerank(query="test", candidates=candidates, top_k=2)

        # Should return original order
        assert result[0].chunk_id == "a:0"
        assert result[1].chunk_id == "b:0"

class TestEvidenceSelectorIntegration:
    """Test evidence selection within token budget."""

    def test_select_respects_token_budget(self) -> None:
        """EvidenceSelector keeps selection within max_context_tokens."""
        candidates = [
            RetrievalResult(
                chunk_id=f"doc:{i}",
                document_id="doc",
                text=f"Text for document {i} " * 50,  # ~350 chars ≈ 87 tokens
                score=1.0 - i * 0.1,
                source="doc.md",
                retriever=RetrieverType.HYBRID,
                rank=i,
            )
            for i in range(10)
        ]

        settings = Settings(max_context_tokens=200)
        selector = EvidenceSelector(settings=settings)
        selected = selector.select(candidates, top_k=5)

        # Should fit within budget (chars/4 ≈ tokens)
        total_chars = sum(len(r.text) for r in selected)
        assert total_chars <= 200 * 4

    def test_select_deduplicates_by_chunk_id(self) -> None:
        """Duplicate chunk_ids are removed from selection."""
        candidates = [
            RetrievalResult(
                chunk_id="a:0",
                document_id="a",
                text="text a",
                score=0.9,
                source="a.md",
                retriever=RetrieverType.HYBRID,
                rank=0,
            ),
            RetrievalResult(
                chunk_id="a:0",  # duplicate
                document_id="a",
                text="text a",
                score=0.8,
                source="a.md",
                retriever=RetrieverType.HYBRID,
                rank=1,
            ),
        ]
        selector = EvidenceSelector(settings=Settings(max_context_tokens=1000))
        selected = selector.select(candidates, top_k=5)
        assert len(selected) == 1
        assert selected[0].chunk_id == "a:0"

    def test_select_respects_top_k(self) -> None:
        """EvidenceSelector returns at most top_k chunks."""
        candidates = [
            RetrievalResult(
                chunk_id=f"doc:{i}",
                document_id="doc",
                text=f"Text {i}",
                score=1.0 - i * 0.01,
                source="doc.md",
                retriever=RetrieverType.HYBRID,
                rank=i,
            )
            for i in range(20)
        ]
        selector = EvidenceSelector(settings=Settings(max_context_tokens=10000))
        selected = selector.select(candidates, top_k=3)
        assert len(selected) <= 3

class TestPipelineIntegration:
    """End-to-end pipeline tests combining all Phase 5 components."""

    def test_full_pipeline_produces_answerable_context(self) -> None:
        """Query → retrieve → rerank → select produces relevant evidence."""
        # 1. Retrieve candidates via hybrid retriever
        vec_hits = [
            {
                "id": "auth:0",
                "score": 0.9,
                "payload": {
                    "text": "To authenticate, include your API key in the Authorization header.",
                    "document_id": "auth",
                    "source": "auth.md",
                    "page_number": 1,
                    "section": "Authentication",
                },
            },
            {
                "id": "auth:1",
                "score": 0.8,
                "payload": {
                    "text": "JWT tokens expire after 3600 seconds.",
                    "document_id": "auth",
                    "source": "auth.md",
                    "page_number": 2,
                    "section": "Tokens",
                },
            },
        ]
        bm25_hits = [
            {
                "id": "deploy:0",
                "score": 5.0,
                "payload": {
                    "text": "Deploy to Kubernetes using the provided helm chart.",
                    "document_id": "deploy",
                    "source": "deploy.md",
                    "page_number": 1,
                    "section": "Deployment",
                },
            },
            {
                "id": "auth:0",  # duplicate
                "score": 4.0,
                "payload": {
                    "text": "To authenticate, include your API key in the Authorization header.",
                    "document_id": "auth",
                    "source": "auth.md",
                    "page_number": 1,
                    "section": "Authentication",
                },
            },
        ]

        hybrid = HybridRetriever(
            settings=Settings(),
            embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            vector_store=_FakeVectorStore(vec_hits),  # type: ignore[arg-type]
            bm25_indexer=_FakeBM25Indexer(bm25_hits),  # type: ignore[arg-type]
        )

        candidates = hybrid.retrieve(query="authentication", top_k=5)
        assert len(candidates) == 3  # auth:0, auth:1, deploy:0 (deduped)
        assert all(r.retriever == RetrieverType.HYBRID for r in candidates)

        # 2. Rerank (simulate — reranker not enabled by default)
        reranker = Reranker(settings=Settings(enable_reranker=False), model=None)
        reranked = reranker.rerank(query="How do I authenticate?", candidates=candidates, top_k=3)

        # 3. Select evidence within token budget
        selector = EvidenceSelector(settings=Settings(max_context_tokens=500))
        evidence = selector.select(reranked, top_k=3)

        # Evidence should be non-empty, within budget, deduplicated
        assert len(evidence) >= 1
        assert len(evidence) <= 3
        chunk_ids = [r.chunk_id for r in evidence]
        assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates
