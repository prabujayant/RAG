"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval results.

RRF merges results from multiple retrievers (e.g. vector + BM25) into a single
ranked list without requiring score normalization. The formula is:

    RRF_score = Σ 1 / (RRF_K + rank)

where rank is the 0-based position in each retriever's result list.
RRF_K controls how much a retriever's rank matters — higher values reduce the
impact of rank differences between retrievers.

Reference:
    "Reciprocal Rank Fusion Beats Conditionally Independent Models"
    (Cormack et al., 2009)
"""

from __future__ import annotations

from dataclasses import replace

from app.retrieval.models import RetrievalResult, RetrieverType


def reciprocal_rank_fusion(
    results_by_retriever: dict[RetrieverType, list[RetrievalResult]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse results from multiple retrievers using Reciprocal Rank Fusion.

    Parameters
    ----------
    results_by_retriever:
        Mapping from retriever type to its sorted (descending-score) result list.
        Empty lists are allowed; they contribute nothing to the fusion.
    rrf_k:
        RRF constant (default 60). Higher values reduce the influence of
        rank differences between retrievers.

    Returns
    -------
    list[RetrievalResult]
        Fused and deduplicated results sorted by RRF score (descending).
        Ties are broken by chunk_id ascending for determinism.

    Examples
    --------
    >>> vec_result = RetrievalResult(chunk_id="a:0", document_id="a", text="...", score=0.9,
    ...                              source="a.md", retriever=RetrieverType.VECTOR, rank=0)
    >>> bm25_result = RetrievalResult(chunk_id="b:0", document_id="b", text="...", score=5.0,
    ...                               source="b.md", retriever=RetrieverType.BM25, rank=0)
    >>> fused = reciprocal_rank_fusion({RetrieverType.VECTOR: [vec_result],
    ...                                RetrieverType.BM25: [bm25_result]})
    >>> fused[0].chunk_id
    'a:0'
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")

    # Compute RRF score for every (retriever, rank) pair.
    # Use a dict to accumulate scores per chunk_id, then deduplicate.
    chunk_scores: dict[str, float] = {}
    chunk_results: dict[str, RetrievalResult] = {}
    # Track which retrievers contributed to each chunk (before results are deduplicated)
    chunk_contributors: dict[str, set[str]] = {}

    for retriever, results in results_by_retriever.items():
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (rrf_k + rank)
            chunk_scores[result.chunk_id] = (
                chunk_scores.get(result.chunk_id, 0.0) + rrf_score
            )
            # Accumulate contributing retriever for this chunk
            if result.chunk_id not in chunk_contributors:
                chunk_contributors[result.chunk_id] = set()
            chunk_contributors[result.chunk_id].add(retriever.value)
            # Keep the first-seen result for each chunk_id (preserves richer metadata)
            if result.chunk_id not in chunk_results:
                chunk_results[result.chunk_id] = result

    # Sort by RRF score descending; ties broken by chunk_id ascending for determinism.
    # Two-phase sort: first by chunk_id (ascending, stable), then by score (descending).
    fused = sorted(
        sorted(chunk_results.values(), key=lambda r: r.chunk_id),
        key=lambda r: chunk_scores[r.chunk_id],
        reverse=True,
    )

    # Re-assign ranks and mark as HYBRID retriever.
    # Use replace() so caller-owned RetrievalResult objects are never mutated.
    fused_copies: list[RetrievalResult] = []
    for idx, result in enumerate(fused):
        metadata = dict(result.metadata)
        metadata["contributing_retrievers"] = chunk_contributors[result.chunk_id]
        fused_copies.append(
            replace(
                result,
                rank=idx,
                retriever=RetrieverType.HYBRID,
                score=chunk_scores[result.chunk_id],
                metadata=metadata,
            )
        )

    return fused_copies
