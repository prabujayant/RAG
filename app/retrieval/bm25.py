"""OpenSearch BM25 full-text search wrapper."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from app.config import get_settings
from app.config.settings import Settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def _chunk_payload(chunk: Chunk) -> dict:
    """Build the OpenSearch document for a single chunk."""
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


class BM25Indexer:
    """OpenSearch-backed BM25 indexer.

    Parameters
    ----------
    settings:
        Application settings containing OPENSEARCH_URL, OPENSEARCH_USERNAME,
        OPENSEARCH_PASSWORD, OPENSEARCH_INDEX.
    client:
        Optional OpenSearch client (allows injection in tests).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenSearch | None = None,
    ) -> None:
        s = settings or get_settings()
        self.index = s.opensearch_index
        self._client = client or OpenSearch(
            [s.opensearch_url],
            basic_auth=(s.opensearch_username, s.opensearch_password),
            verify_certs=False,
            timeout=30,
        )

    def ensure_index(self) -> None:
        """Create the BM25 index with standard analyzer if it does not exist."""
        if self._client.indices.exists(index=self.index):
            logger.info("OpenSearch index %s already exists", self.index)
            return
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "document_name": {"type": "text"},
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "section": {"type": "text"},
                    "content_hash": {"type": "keyword"},
                }
            },
        }
        self._client.indices.create(index=self.index, body=body)
        logger.info("Created OpenSearch index %s", self.index)

    def index_documents(self, chunks: list[Chunk]) -> None:
        """Bulk-index a list of chunks into OpenSearch (idempotent on chunk_id)."""
        if not chunks:
            return
        self.ensure_index()
        bulk_body: list[dict] = []
        for chunk in chunks:
            bulk_body.append({"index": {"_index": self.index, "_id": chunk.chunk_id}})
            bulk_body.append(_chunk_payload(chunk))
        self._client.bulk(body=bulk_body, refresh=True)
        logger.info("BM25-indexed %d chunks to OpenSearch", len(chunks))

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        """Alias for :meth:`index_documents` (parity with :class:`VectorStore`).

        Returns the number of chunks indexed.
        """
        chunk_list = list(chunks)
        self.index_documents(chunk_list)
        return len(chunk_list)

    def get(self, chunk_id: str) -> dict | None:
        """Return the indexed document for a single chunk, or None if missing."""
        try:
            resp = self._client.get(index=self.index, id=chunk_id)
        except NotFoundError:
            return None
        return resp.get("_source")

    def count(self, document_id: str | None = None) -> int:
        """Count documents in the index, optionally filtered by document_id."""
        body: dict = {"query": {"match_all": {}}} if document_id is None else {
            "query": {"term": {"document_id": document_id}}
        }
        try:
            resp = self._client.count(index=self.index, body=body)
        except NotFoundError:
            return 0
        return int(resp.get("count", 0))

    def delete_by_document(self, document_id: str, refresh: bool = True) -> int:
        """Delete all chunks of a single document. Returns the number deleted."""
        try:
            resp = self._client.delete_by_query(
                index=self.index,
                body={"query": {"term": {"document_id": document_id}}},
                refresh=refresh,
                conflicts="proceed",
            )
        except NotFoundError:
            return 0
        return int(resp.get("deleted", 0))

    def delete_by_ids(self, chunk_ids: Iterable[str], refresh: bool = True) -> int:
        """Delete specific chunks by chunk_id. Returns the number deleted."""
        ids = list(chunk_ids)
        if not ids:
            return 0
        body: list[dict] = []
        for cid in ids:
            body.append({"delete": {"_index": self.index, "_id": cid}})
        resp = self._client.bulk(body=body, refresh=refresh)
        if not resp.get("errors"):
            return len(ids)
        # Count successful deletes when some failed.
        return sum(
            1
            for item in resp.get("items", [])
            if item.get("delete", {}).get("status", 500) < 300
        )

    def delete_index(self) -> None:
        """Drop the entire index (idempotent)."""
        try:
            self._client.indices.delete(index=self.index)
        except NotFoundError:
            return
        logger.info("Deleted OpenSearch index %s", self.index)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_document_ids: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        """Search for the top-k chunks matching the query.

        Parameters
        ----------
        query:
            Free-text query string.
        top_k:
            Number of results to return.
        filter_document_ids:
            Optional document_id filter.
        fields:
            Fields searched by ``multi_match``. Defaults to ``["text"]``;
            pass e.g. ``["text", "section"]`` to expand recall.

        Returns
        -------
        list[dict]
            Each dict has ``id``, ``score``, and ``payload`` (chunk fields).
        """
        self.ensure_index()
        search_fields = fields or ["text"]
        must_clause: list[dict] = [{"multi_match": {"query": query, "fields": search_fields}}]
        filter_clause: list[dict] = (
            [{"terms": {"document_id": filter_document_ids}}] if filter_document_ids else []
        )
        body = {
            "query": {"bool": {"must": must_clause, "filter": filter_clause}},
            "size": top_k,
        }
        resp = self._client.search(index=self.index, body=body)
        hits = resp["hits"]["hits"]
        return [
            {"id": h["_id"], "score": h["_score"], "payload": h["_source"]} for h in hits
        ]
