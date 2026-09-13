"""Cross-encoder reranking using a small multilingual MiniLM model.

The default model is ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` (~118M
params). It was chosen over ``BAAI/bge-reranker-v2-m3`` (568M) after
benchmarking both on this project's CPU-only target:

======================  ===========  ======================
model                   per pair     ranking sanity check
======================  ===========  ======================
bge-reranker-v2-m3      7898 ms      pass
mmarco-mMiniLMv2-L12     404 ms      pass (EN/DE/ID)
ms-marco-MiniLM-L-6      221 ms      weak on DE (negative)
======================  ===========  ======================

The MiniLM model is ~20x faster while still ranking relevant passages above
irrelevant ones across English, German and Indonesian. Set ``RERANKER_MODEL``
to ``BAAI/bge-reranker-v2-m3`` if maximum accuracy is required and the extra
latency is acceptable.

The reranker re-scores query-document pairs, providing more accurate relevance
signals than the independent vector and BM25 scores used in the initial
retrieval stage.

Only candidates passed in are scored; when the reranker is disabled (via
``enable_reranker=False``) or the model is unavailable, the original ordering is
preserved and reranking is silently skipped.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.config.settings import Settings
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level model singleton (loaded once, shared across all Reranker
# instances). Route handlers construct a Reranker per request — without this
# cache every query would pay the full cross-encoder load cost (~15-25s).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_shared_reranker(model_name: str) -> CrossEncoder:
    """Load and cache the cross-encoder model at module scope."""
    logger.info("Loading cross-encoder model: %s", model_name)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


class Reranker:
    """Cross-encoder reranker (default: multilingual MiniLM, ~118M params).

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
        """Return the injected model, else the shared cross-encoder
        (loaded once at module scope)."""
        if self._model is None:
            self._model = _get_shared_reranker(self._settings.reranker_model)
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

    def warmup(self) -> bool:
        """Pre-load the cross-encoder model; return True on success.

        Used at app startup so the first reranked query doesn't pay the
        model-load cost. Returns False (without raising) when reranking is
        disabled or the model cannot be loaded.
        """
        if not self.is_enabled:
            return False
        try:
            _ = self._reranker_model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker warm-up failed: %s", exc)
            return False
        return True

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
