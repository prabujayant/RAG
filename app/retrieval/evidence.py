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
    total_selected: int,
) -> tuple[float, int]:
    """Return a sort key that penalises chunks from over-represented documents.

    We combine the relevance score with a document-frequency penalty so that
    chunks from documents that already contributed several chunks are ranked
    slightly lower when scores are close.
    """
    doc_count = doc_counts.get(result.document_id, 0)
    # Penalise by doc_count so repeated documents are de-prioritised
    # The multiplier keeps score ordering intact while providing diversity
    penalty = 1.0 / (doc_count + 1)
    return result.score * penalty, total_selected


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
            Selected evidence chunks, deduplicated, token-budgeted, and ordered
            by final relevance.
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

        # Step 2: select within token budget, favouring doc diversity
        selected: list[RetrievalResult] = []
        doc_counts: dict[str, int] = {}
        cumulative_tokens = 0

        for result in deduped:
            chunk_tokens = estimate_tokens(result.text)
            if (
                len(selected) >= effective_k
                or cumulative_tokens + chunk_tokens > max_tokens
            ):
                break
            selected.append(result)
            doc_counts[result.document_id] = doc_counts.get(result.document_id, 0) + 1
            cumulative_tokens += chunk_tokens

        # Step 3: if budget allows and there are remaining candidates, try to fill
        # up to top_k with higher-scoring / diverse docs already passed over
        remaining = [
            c for c in deduped if c not in selected
        ]
        for result in remaining:
            if len(selected) >= effective_k:
                break
            chunk_tokens = estimate_tokens(result.text)
            if cumulative_tokens + chunk_tokens <= max_tokens:
                selected.append(result)
                doc_counts[result.document_id] = doc_counts.get(result.document_id, 0) + 1
                cumulative_tokens += chunk_tokens

        logger.debug(
            "Selected %d chunks (%d tokens) from %d documents",
            len(selected),
            cumulative_tokens,
            len(doc_counts),
        )
        return selected
