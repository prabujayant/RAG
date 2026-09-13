"""Evidence selection: pick the best K chunks within a token budget.

Evidence selection takes the re-ranked candidate list and chooses a subset that:

1. Fits within the ``max_context_tokens`` budget.
2. Prioritises the highest-scoring chunks.
3. Deduplicates chunks with the same ``chunk_id``.
4. Favours document diversity: when scores are close, prefer chunks from
   under-represented documents to avoid all context coming from one source.
5. Preserves the final relevance order for ties at the budget boundary.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.config.settings import Settings
from app.ingestion.chunker import estimate_tokens
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


def _score_with_diversity(
    result: RetrievalResult,
    doc_counts: dict[str, int],
) -> float:
    """Return the relevance score penalised for over-represented documents.

    Chunks from documents that already contributed several chunks are ranked
    slightly lower, so that when scores are close the selector prefers
    under-represented documents instead of filling context from one source.
    """
    doc_count = doc_counts.get(result.document_id, 0)
    # Penalise by doc_count so repeated documents are de-prioritised.
    # The 1/(n+1) multiplier keeps score ordering intact for clear winners
    # while letting close runner-ups from fresh documents win ties.
    return result.score * (1.0 / (doc_count + 1))


class EvidenceSelector:
    """Select evidence chunks within a token budget, favouring document diversity.

    Parameters
    ----------
    settings:
        Application settings containing ``FINAL_CONTEXT_K`` and ``MAX_CONTEXT_TOKENS``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()

    @property
    def final_context_k(self) -> int:
        """Maximum number of chunks to include in the final context."""
        return self._settings.final_context_k

    @property
    def max_context_tokens(self) -> int:
        """Maximum number of tokens allowed in the combined evidence."""
        return self._settings.max_context_tokens

    def select(
        self,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Select up to ``final_context_k`` chunks within ``max_context_tokens``.

        Parameters
        ----------
        candidates:
            Re-ranked :class:`RetrievalResult` list from the hybrid retriever
            (optionally re-ranked by the :class:`Reranker`).
        top_k:
            Override for ``final_context_k``. Use this when you need a different
            limit than the configured default.

        Returns
        -------
        list[RetrievalResult]
            Selected evidence chunks, deduplicated, token-budgeted, in
            selection order (diversity-adjusted relevance, stable for ties).
        """
        effective_k = top_k or self.final_context_k
        max_tokens = self.max_context_tokens

        if not candidates:
            return []

        # Step 1: deduplicate by chunk_id (keep first occurrence)
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for c in candidates:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                deduped.append(c)

        # Step 2: greedy selection within token budget, favouring doc
        # diversity. Each round picks the remaining chunk with the best
        # diversity-adjusted score that still fits the token budget, so a
        # slightly lower-scoring chunk from a fresh document beats a
        # same-document chunk when scores are close — while smaller fitting
        # chunks later in the list are still picked up (no early cutoff).
        selected: list[RetrievalResult] = []
        doc_counts: dict[str, int] = {}
        cumulative_tokens = 0
        pool = list(enumerate(deduped))  # (original index, chunk) for stable ties

        while pool and len(selected) < effective_k:
            best_pos: int | None = None
            best_key: tuple[float, int] | None = None
            for pos, (idx, result) in enumerate(pool):
                if cumulative_tokens + estimate_tokens(result.text) > max_tokens:
                    continue
                key = (_score_with_diversity(result, doc_counts), -idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pos = pos
            if best_pos is None:
                break  # nothing left fits the remaining budget
            _, picked = pool.pop(best_pos)
            selected.append(picked)
            doc_counts[picked.document_id] = doc_counts.get(picked.document_id, 0) + 1
            cumulative_tokens += estimate_tokens(picked.text)

        logger.debug(
            "Selected %d chunks (%d tokens) from %d documents",
            len(selected),
            cumulative_tokens,
            len(doc_counts),
        )
        return selected
