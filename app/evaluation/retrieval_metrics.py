"""Pure retrieval metric functions.

All functions in this module operate on plain Python sets/lists — no
databases, LLMs, or models. They are designed to be:

- deterministic
- side-effect free
- easy to unit-test
- safe for empty / degenerate inputs

Metrics implemented
-------------------
- :func:`recall_at_k`       — fraction of relevant items retrieved in top-k
- :func:`precision_at_k`    — fraction of top-k items that are relevant
- :func:`mrr`               — Mean Reciprocal Rank (across multiple queries)
- :func:`ndcg_at_k`         — Normalized Discounted Cumulative Gain at k
- :func:`hit_rate`          — 1.0 if any relevant retrieved, else 0.0
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Single-query metrics
# ---------------------------------------------------------------------------


def _normalize_ids(ids: Iterable[object]) -> list[str]:
    """Coerce any iterable of ids to a list of strings, preserving order."""
    return [str(x) for x in ids]


def recall_at_k(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 5,
) -> float:
    """Compute Recall@K: ``|relevant ∩ retrieved[:k]| / |relevant|``.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk ids (in retrieval order).
    relevant_ids:
        Set-like collection of relevant chunk ids (no duplicates required).
    k:
        Cut-off. If ``k <= 0`` returns 0.0.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``. Returns 0.0 if there are no relevant ids.
    """
    if k <= 0:
        return 0.0
    rel_set = set(_normalize_ids(relevant_ids))
    if not rel_set:
        return 0.0
    retrieved = _normalize_ids(retrieved_ids)[:k]
    hits = sum(1 for cid in retrieved if cid in rel_set)
    return hits / len(rel_set)


def precision_at_k(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 5,
) -> float:
    """Compute Precision@K: ``|relevant ∩ retrieved[:k]| / k``.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk ids (in retrieval order).
    relevant_ids:
        Set-like collection of relevant chunk ids.
    k:
        Cut-off. If ``k <= 0`` returns 0.0.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``. Returns 0.0 if there are no retrieved ids.
    """
    if k <= 0:
        return 0.0
    rel_set = set(_normalize_ids(relevant_ids))
    retrieved = _normalize_ids(retrieved_ids)[:k]
    if not retrieved:
        return 0.0
    hits = sum(1 for cid in retrieved if cid in rel_set)
    return hits / k


def hit_rate(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int | None = None,
) -> float:
    """Return 1.0 if at least one relevant id appears in the top-``k`` (else 0.0).

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk ids.
    relevant_ids:
        Collection of relevant chunk ids.
    k:
        Optional cut-off. ``None`` checks the full list.
    """
    if k is not None and k <= 0:
        return 0.0
    rel_set = set(_normalize_ids(relevant_ids))
    if not rel_set:
        return 0.0
    retrieved = _normalize_ids(retrieved_ids)
    if k is not None:
        retrieved = retrieved[:k]
    return 1.0 if any(cid in rel_set for cid in retrieved) else 0.0


def reciprocal_rank(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
) -> float:
    """Return the reciprocal rank of the first relevant retrieved id.

    Returns 0.0 if no relevant id is found.
    """
    rel_set = set(_normalize_ids(relevant_ids))
    if not rel_set:
        return 0.0
    for idx, cid in enumerate(_normalize_ids(retrieved_ids), start=1):
        if cid in rel_set:
            return 1.0 / idx
    return 0.0


def mrr(
    queries: Iterable[tuple[Iterable[object], Iterable[object]]],
) -> float:
    """Compute Mean Reciprocal Rank over a collection of (retrieved, relevant) pairs.

    Each input element is a 2-tuple ``(retrieved_ids, relevant_ids)``.

    Returns 0.0 if *queries* is empty.
    """
    total = 0.0
    count = 0
    for retrieved, relevant in queries:
        total += reciprocal_rank(retrieved, relevant)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def ndcg_at_k(
    retrieved_ids: Iterable[object],
    relevant_ids: Iterable[object],
    k: int = 5,
) -> float:
    """Compute NDCG@K with binary relevance (1 = relevant, 0 = not).

    Implementation uses the standard DCG formula with log2(rank+1) discount
    and an IDCG that is the optimum ordering (all relevant items first).
    """
    if k <= 0:
        return 0.0
    rel_set = set(_normalize_ids(relevant_ids))
    if not rel_set:
        return 0.0
    retrieved = _normalize_ids(retrieved_ids)[:k]

    # DCG
    dcg = 0.0
    for idx, cid in enumerate(retrieved, start=1):
        if cid in rel_set:
            # +2 because ranks are 1-based
            dcg += 1.0 / math.log2(idx + 1)

    # IDCG: best possible DCG with min(|relevant|, k) relevant items
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregate metric set
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetricSet:
    """Aggregate retrieval metrics for a single evaluation slice.

    The ``per_difficulty`` field carries the same metrics broken down by
    :class:`~app.evaluation.datasets.Difficulty` for stratified reporting.
    """

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    hit_rate_at_5: float = 0.0
    count: int = 0
    per_difficulty: dict[str, RetrievalMetricSet] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "mrr": self.mrr,
            "ndcg_at_10": self.ndcg_at_10,
            "hit_rate_at_5": self.hit_rate_at_5,
            "count": self.count,
            "per_difficulty": {
                k: v.to_dict() for k, v in self.per_difficulty.items()
            },
        }


def aggregate_retrieval_metrics(
    rows: list[tuple[list[str], list[str], str]],
) -> RetrievalMetricSet:
    """Compute aggregate retrieval metrics from per-example rows.

    Parameters
    ----------
    rows:
        List of ``(retrieved_ids, relevant_ids, difficulty_str)`` tuples.
        The order of ``retrieved_ids`` matters for ranking metrics.

    Returns
    -------
    RetrievalMetricSet
        Aggregate metrics overall and per-difficulty bucket.
    """
    if not rows:
        return RetrievalMetricSet()

    # Per-difficulty buckets
    buckets: dict[str, list[tuple[list[str], list[str]]]] = {}
    all_pairs: list[tuple[list[str], list[str]]] = []

    for retrieved, relevant, difficulty in rows:
        pair = (list(retrieved), list(relevant))
        all_pairs.append(pair)
        buckets.setdefault(str(difficulty), []).append(pair)

    overall = _metrics_from_pairs(all_pairs)
    overall.per_difficulty = {
        d: _metrics_from_pairs(pairs) for d, pairs in sorted(buckets.items())
    }
    return overall


def _metrics_from_pairs(pairs: list[tuple[list[str], list[str]]]) -> RetrievalMetricSet:
    if not pairs:
        return RetrievalMetricSet()

    recall5 = _mean(pairs, lambda r, rel: recall_at_k(r, rel, k=5))
    recall10 = _mean(pairs, lambda r, rel: recall_at_k(r, rel, k=10))
    prec5 = _mean(pairs, lambda r, rel: precision_at_k(r, rel, k=5))
    prec10 = _mean(pairs, lambda r, rel: precision_at_k(r, rel, k=10))
    mrr_score = _mean(pairs, reciprocal_rank)
    ndcg10 = _mean(pairs, lambda r, rel: ndcg_at_k(r, rel, k=10))
    hr5 = _mean(pairs, lambda r, rel: hit_rate(r, rel, k=5))

    return RetrievalMetricSet(
        recall_at_5=recall5,
        recall_at_10=recall10,
        precision_at_5=prec5,
        precision_at_10=prec10,
        mrr=mrr_score,
        ndcg_at_10=ndcg10,
        hit_rate_at_5=hr5,
        count=len(pairs),
    )


def _mean(pairs: list[tuple[list[str], list[str]]], fn) -> float:
    if not pairs:
        return 0.0
    total = sum(fn(r, rel) for r, rel in pairs)
    return total / len(pairs)


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query in one call.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk ids.
    expected_ids:
        Relevant chunk ids from the golden example.
    k:
        Cut-off rank for @K metrics (default 10).

    Returns
    -------
    dict with keys: recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate
    """
    return {
        "recall_at_k": recall_at_k(retrieved_ids, expected_ids, k=k),
        "precision_at_k": precision_at_k(retrieved_ids, expected_ids, k=k),
        "mrr": reciprocal_rank(retrieved_ids, expected_ids),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, expected_ids, k=k),
        "hit_rate": hit_rate(retrieved_ids, expected_ids, k=k),
    }


__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "reciprocal_rank",
    "RetrievalMetricSet",
    "aggregate_retrieval_metrics",
]
