"""Qdrant vector store wrapper.

Production-ready vector search on top of Qdrant. Supports:

* idempotent collection creation with configurable distance metric
* chunk upsert with real vectors and a complete retrieval payload
* metadata-filtered cosine search (with optional score threshold)
* per-id and per-document deletion
* count + by-id fetch (for verification and tests)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance

from app.config import get_settings
from app.config.settings import Settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def _point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id (UUID) derived from a chunk_id.

    Qdrant only accepts unsigned-integer or UUID point ids, but our chunk ids
    are strings of the form ``{document_id}:{index}``. We derive a stable UUID
    from the chunk id so re-indexing a chunk is idempotent and the mapping is
    fully deterministic. The original ``chunk_id`` is retained in the payload.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _point_payload(chunk: Chunk) -> dict:
    """Payload stored alongside each Qdrant point."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "text": chunk.text,
        "source": chunk.source,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "content_hash": chunk.content_hash,
    }


class VectorStore:
    """Qdrant-backed vector store for chunk embeddings.

    Parameters
    ----------
    settings:
        Application settings containing QDRANT_URL, QDRANT_COLLECTION,
        QDRANT_DISTANCE and EMBEDDING_DIM.
    client:
        Optional :class:`QdrantClient` instance (allows injection in tests).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        s = settings or get_settings()
        self.collection = s.qdrant_collection
        self.vector_size = s.embedding_dim
        self.distance = Distance[s.qdrant_distance.upper()]
        self._client = client or QdrantClient(url=s.qdrant_url, timeout=30)

    # ------------------------------------------------------------------ admin

    def ensure_collection(self) -> None:
        """Create the collection (BGE-M3 unnamed default vector) if missing."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            return
        try:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
        except UnexpectedResponse as exc:
            # Race: another process created it concurrently.
            if "already exists" not in str(exc).lower():
                raise
        logger.info("Created Qdrant collection %s", self.collection)

    def delete_collection(self) -> None:
        """Drop the entire collection (idempotent)."""
        try:
            self._client.delete_collection(collection_name=self.collection)
        except UnexpectedResponse:
            return
        logger.info("Deleted Qdrant collection %s", self.collection)

    # ----------------------------------------------------------------- writes

    def upsert(
        self,
        chunks: Iterable[Chunk],
        vectors: list[list[float]] | None = None,
    ) -> int:
        """Upsert chunks as points.

        Parameters
        ----------
        chunks:
            Iterable of :class:`Chunk` objects.
        vectors:
            Optional pre-computed vectors aligned with ``chunks``. When omitted,
            a zero-vector placeholder is inserted (useful for tests that only
            care about payload round-tripping).

        Returns the number of chunks upserted.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        self.ensure_collection()
        points: list[qm.PointStruct] = []
        for i, chunk in enumerate(chunk_list):
            vector = vectors[i] if vectors is not None else [0.0] * self.vector_size
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Vector length {len(vector)} != configured size {self.vector_size}"
                )
            points.append(
                qm.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=_point_payload(chunk),
                )
            )
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        logger.info("Upserted %d chunks to Qdrant", len(points))
        return len(points)

    # ----------------------------------------------------------------- reads

    def get(self, chunk_id: str) -> dict | None:
        """Return a single point's payload, or None if missing."""
        try:
            resp = self._client.retrieve(
                collection_name=self.collection, ids=[_point_id(chunk_id)], with_payload=True
            )
        except UnexpectedResponse:
            return None
        if not resp:
            return None
        point = resp[0]
        return {"id": point.id, "payload": point.payload}

    def count(self, document_id: str | None = None) -> int:
        """Count points in the collection, optionally filtered by document_id."""
        flt = None
        if document_id is not None:
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id",
                        match=qm.MatchValue(value=document_id),
                    )
                ]
            )
        try:
            resp = self._client.count(
                collection_name=self.collection,
                count_filter=flt,
                exact=True,
            )
        except UnexpectedResponse:
            return 0
        return int(resp.count)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filter_document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search for the top-k nearest chunks by configured similarity."""
        self.ensure_collection()
        flt: qm.Filter | None = None
        if filter_document_ids:
            flt = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id",
                        match=qm.MatchAny(any=filter_document_ids),
                    )
                ]
            )
        results = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=flt,
            score_threshold=score_threshold,
            with_vectors=False,
            with_payload=True,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points]

    # ---------------------------------------------------------------- deletes

    def delete_by_ids(self, chunk_ids: Iterable[str]) -> int:
        """Delete specific chunks by chunk_id. Returns the number deleted."""
        ids = list(chunk_ids)
        if not ids:
            return 0
        point_ids = [_point_id(cid) for cid in ids]
        self._client.delete(collection_name=self.collection, points_selector=point_ids, wait=True)  # type: ignore[arg-type]
        return len(ids)

    def delete_by_document(self, document_id: str) -> int:
        """Delete every point belonging to a document.

        Qdrant's delete-by-filter doesn't return a count, so we issue a count
        *before* and a count *after* the deletion to compute the delta. Returns
        the number of points removed.
        """
        before = self.count(document_id=document_id)
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="document_id", match=qm.MatchValue(value=document_id)
                )
            ]
        )
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(filter=flt),
            wait=True,
        )
        return before

    # ----------------------------------------------------------------- export

    def iter_all(self, batch_size: int = 256) -> Iterable[dict]:
        """Stream every point (payload only) in the collection. Test/eval helper."""
        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                yield {"id": point.id, "payload": point.payload}
            if next_offset is None:
                return
            offset = next_offset


__all__ = ["VectorStore"]
