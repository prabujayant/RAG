"""BGE-M3 embeddings wrapper.

Loads the model lazily on first use to keep startup time fast, and exposes
two distinct methods — :meth:`Embedder.embed_documents` (storage) and
:meth:`Embedder.embed_queries` (retrieval) — that mirror the recommended
asymmetric pattern for BGE-M3.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings
from app.config.settings import Settings
from app.retrieval.vector import VectorStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# BGE-M3 returns 1024-d vectors; keep an explicit constant so tests can
# detect dimension drift before the real model is loaded.
BGE_M3_DIM = 1024


# ---------------------------------------------------------------------------
# Module-level model singleton (loaded once, shared across all Embedder
# instances).  Using an lru_cache here means every Embedder instance — even
# ones created per-request in route handlers — will reuse the same
# SentenceTransformer object without repeatedly loading it from disk.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_shared_model(model_name: str) -> SentenceTransformer:
    """Load and cache the sentence-transformer model at module scope.

    Attempts a local-only load first. The model is normally already present in
    the HuggingFace cache, and skipping the hub update check cuts load time
    from ~50s to a few seconds. Falls back to a normal (downloading) load when
    the model is not cached yet, so a fresh machine still works.
    """
    from sentence_transformers import SentenceTransformer

    # CPU encode scales with threads + batch size. Batch comes from
    # EMBEDDING_BATCH_SIZE (now 64); threads default to all cores so a
    # 150-chunk PDF encodes in one pass instead of ten.
    try:
        import os

        import torch

        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass

    logger.info("Loading embedding model %s …", model_name)
    start = time.perf_counter()
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "%s not available in the local cache (%s); loading from HuggingFace",
            model_name,
            exc,
        )
        model = SentenceTransformer(model_name)
    logger.info(
        "Embedding model %s loaded in %.1fs", model_name, time.perf_counter() - start
    )
    return model


class Embedder:
    """Sentence-transformers wrapper for BGE-M3 embeddings.

    Parameters
    ----------
    settings:
        Application settings with EMBEDDING_MODEL, EMBEDDING_DIM,
        EMBEDDING_BATCH_SIZE.
    vector_store:
        Optional pre-built :class:`VectorStore` to upsert into. When ``None``,
        a new instance is created lazily from settings on first upsert.
    model:
        Optional pre-constructed ``SentenceTransformer`` instance (useful for
        tests with fake models). When omitted the shared module-level model
        is loaded lazily on first use.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        s = settings or get_settings()
        self.model_name = s.embedding_model
        self.batch_size = s.embedding_batch_size
        self.vector_size = s.embedding_dim
        self._vector_store = vector_store
        self._model = model

    @property
    def model(self) -> SentenceTransformer:
        """Return the injected model, else the shared sentence-transformer
        model (loaded once)."""
        if self._model is None:
            self._model = _get_shared_model(self.model_name)
        return self._model

    def _check_dim(self, vectors: list[list[float]]) -> None:
        """Fail fast when the model output disagrees with configured EMBEDDING_DIM."""
        if not vectors:
            return
        actual = len(vectors[0])
        if actual != self.vector_size:
            raise ValueError(
                f"Embedding dim mismatch: model produced {actual}, "
                f"EMBEDDING_DIM={self.vector_size}. Set EMBEDDING_DIM to {actual}."
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        self._check_dim(vectors)
        return vectors

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed *documents* for storage. BGE-M3 prefers the default prompt."""
        text_list = list(texts)
        if not text_list:
            return []
        return self._encode(text_list)

    def embed_queries(self, queries: Iterable[str]) -> list[list[float]]:
        """Embed *queries* for retrieval.

        BGE-M3 documents an asymmetric encoding strategy; we follow it with
        an explicit query prompt. Older sentence-transformers builds may not
        support ``prompt=``; we fall back to the default encoding in that case.
        """
        query_list = list(queries)
        if not query_list:
            return []
        try:
            vectors = self.model.encode(
                query_list,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                prompt="Represent this sentence for searching relevant passages: ",
            ).tolist()
        except TypeError:
            vectors = self._encode(query_list)
        self._check_dim(vectors)
        return vectors

    # Thin alias for callers that don't care about doc/query asymmetry.
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_and_upsert(self, chunks: list[Chunk]) -> int:
        """Embed chunks and upsert them (with their full payload) into Qdrant.

        Returns the number of chunks persisted.
        """
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = self.embed_documents(texts)
        vs = self._vector_store or VectorStore()
        vs.upsert(chunks, vectors=vectors)
        logger.info("Embedded and upserted %d chunks to Qdrant", len(chunks))
        return len(chunks)


__all__ = ["Embedder", "BGE_M3_DIM"]
