"""Hybrid retriever: combines vector and BM25 search via Reciprocal Rank Fusion.

The retrieval pipeline is:

1. Embed the user query using the configured embedder.
2. Execute vector search (Qdrant) and BM25 search (OpenSearch) in parallel.
3. Convert raw search results to :class:`RetrievalResult` objects.
4. Fuse both result lists using Reciprocal Rank Fusion.
5. Return the top ``hybrid_top_k`` results.

The fused results carry a ``contributing_retrievers`` field in their metadata
indicating which underlying retrievers found each chunk.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from app.config import get_settings
from app.config.settings import Settings
from app.retrieval.bm25 import BM25Indexer
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalResult, RetrieverType
from app.retrieval.vector import VectorStore

if TYPE_CHECKING:
    from app.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)


def _bm25_hit_to_result(hit: dict, rank: int) -> RetrievalResult:
    """Convert a BM25 search hit dict to a RetrievalResult."""
    payload = hit["payload"]
    return RetrievalResult(
        chunk_id=hit["id"],
        document_id=payload["document_id"],
        text=payload["text"],
        score=hit["score"],
        source=payload.get("source", ""),
        page_number=payload.get("page_number"),
        section=payload.get("section"),
        retriever=RetrieverType.BM25,
        rank=rank,
        metadata={"contributing_retrievers": {"bm25"}},
    )


def _vector_hit_to_result(hit: dict, rank: int) -> RetrievalResult:
    """Convert a Qdrant vector search hit dict to a RetrievalResult."""
    payload = hit["payload"]
    return RetrievalResult(
        chunk_id=str(hit["id"]),
        document_id=payload["document_id"],
        text=payload["text"],
        score=hit["score"],
        source=payload.get("source", ""),
        page_number=payload.get("page_number"),
        section=payload.get("section"),
        retriever=RetrieverType.VECTOR,
        rank=rank,
        metadata={"contributing_retrievers": {"vector"}},
    )


class HybridRetriever:
    """Combined vector + BM25 retriever using Reciprocal Rank Fusion.

    Parameters
    ----------
    settings:
        Application settings containing BM25_TOP_K, VECTOR_TOP_K, HYBRID_TOP_K,
        RRF_K, and reranker configuration.
    embedder:
        Embedder instance for query encoding.
    vector_store:
        Qdrant vector store instance.
    bm25_indexer:
        OpenSearch BM25 indexer instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_indexer: BM25Indexer | None = None,
    ) -> None:
        from app.embeddings.embedder import Embedder  # lazy to avoid circular import

        self._settings = settings or get_settings()
        self._embedder = embedder or Embedder()
        self._vector = vector_store or VectorStore(settings=self._settings)
        self._bm25 = bm25_indexer or BM25Indexer(settings=self._settings)

    @property
    def embedder(self) -> Embedder:
        """The embedder used for query encoding (lazy-loaded)."""
        return self._embedder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the best chunks for ``query`` using hybrid search.

        Parameters
        ----------
        query:
            Free-text query string.
        top_k:
            Override ``hybrid_top_k`` from settings. Use this when you need a
            different batch size than the default (e.g. for reranking).
        filter_document_ids:
            Optional document_id filter passed to both retrievers.

        Returns
        -------
        list[RetrievalResult]
            Ranked results from the fused vector + BM25 result set.
        """
        effective_top_k = top_k or self._settings.hybrid_top_k
        vector_top_k = self._settings.vector_top_k
        bm25_top_k = self._settings.bm25_top_k
        rrf_k = self._settings.rrf_k

        # Step 1: embed the query
        query_embedding = self._embedder.embed_queries([query])[0]

        # Step 2: run vector and BM25 searches
        with ThreadPoolExecutor(max_workers=2) as executor:
            vec_future = executor.submit(
                self._vector.search,
                query_vector=query_embedding,
                top_k=vector_top_k,
                filter_document_ids=filter_document_ids,
            )
            bm25_future = executor.submit(
                self._bm25.search,
                query=query,
                top_k=bm25_top_k,
                filter_document_ids=filter_document_ids,
            )
            vec_hits = vec_future.result()
            bm25_hits = bm25_future.result()

        # Step 3: convert to RetrievalResult with rank metadata
        vec_results = [
            _vector_hit_to_result(hit, rank) for rank, hit in enumerate(vec_hits)
        ]
        bm25_results = [
            _bm25_hit_to_result(hit, rank) for rank, hit in enumerate(bm25_hits)
        ]

        # Step 4: RRF fusion
        fused = reciprocal_rank_fusion(
            {RetrieverType.VECTOR: vec_results, RetrieverType.BM25: bm25_results},
            rrf_k=rrf_k,
        )

        # Step 5: return top-k
        return fused[:effective_top_k]
