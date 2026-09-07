"""Cross-encoder reranking using BAAI/bge-reranker-v2-m3.

The reranker re-scores query-document pairs using a cross-encoder model, which
provides more accurate relevance signals than the independent vector and BM25
scores used in the initial retrieval stage.

Only the top-K candidates from the hybrid retriever are passed to the reranker,
keeping inference cost predictable. When the reranker is disabled (via
``enable_reranker=False``) or the model is unavailable, the original ordering
is preserved and reranking is silently skipped.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.config.settings import Settings
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker wrapping ``BAAI/bge-reranker-v2-m3``.

    Parameters
    ----------
    settings:
        Application settings containing RERANKER_MODEL, ENABLE_RERANKER,
        and RERANK_TOP_K.
    model:
        Optional pre-constructed :class:`CrossEncoder` instance (useful for tests
        with fake models). When omitted the model is loaded lazily on first use.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: CrossEncoder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model
        self._loaded = model is not None

    @property
    def _reranker_model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model: %s", self._settings.reranker_model)
            self._model = CrossEncoder(
                self._settings.reranker_model,
                max_length=512,
            )
            self._loaded = True
        return self._model

    @property
    def is_enabled(self) -> bool:
        """True when reranking is enabled and the model is available."""
        return self._settings.enable_reranker

    @property
    def is_loaded(self) -> bool:
        """True when the model has been loaded (even if reranking is disabled)."""
        return self._loaded

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-score and reorder ``candidates`` for the given ``query``.

        Parameters
        ----------
        query:
            The original user query.
        candidates:
            List of :class:`RetrievalResult` from the hybrid retriever. These
            are typically the top ``rerank_top_k`` results from the fused list.
        top_k:
            Override for ``rerank_top_k`` from settings. Return only the top-K
            re-ranked results.

        Returns
        -------
        list[RetrievalResult]
            Re-ranked results. When reranking is disabled or the candidate list
            is empty, the original ordering is returned unchanged.
        """
        if not self.is_enabled:
            logger.debug("Reranking is disabled; returning original ordering")
            return candidates

        if not candidates:
            return candidates

        effective_top_k = top_k or self._settings.rerank_top_k

        try:
            model = self._reranker_model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load reranker model, returning original ordering: %s", exc)
            return candidates

        # Build query-document pairs for the cross-encoder
        pairs = [(query, candidate.text) for candidate in candidates]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker prediction failed, returning original ordering: %s", exc)
            return candidates

        # Attach cross-encoder scores to results and re-sort
        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: list[RetrievalResult] = []
        for rank, (result, score) in enumerate(scored[:effective_top_k]):
            result.score = float(score)
            result.rank = rank
            result.metadata["rerank_score"] = float(score)
            reranked.append(result)

        return reranked
