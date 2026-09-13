"""Live-Postgres smoke test for tsvector keyword search.

Requires a reachable PostgreSQL (local `docker compose up -d` or CI
services). Uses an isolated ``smoke_*`` document id and cleans up after
itself, so golden corpora and evals are unaffected.

Covers what fakes cannot: the trigger fills ``tsv``, GIN ranking orders
``ts_rank_cd`` results, and filters work end to end.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest
from app.ingestion.chunker import Chunk, content_hash
from app.retrieval.bm25 import BM25Indexer

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

DOC = f"smoke-{uuid.uuid4().hex[:8]}"


def _chunks() -> list[Chunk]:
    texts = [
        "OAuth access tokens expire after 60 minutes.",
        "Rate limits reset every hour.",
        "Use HTTPS for all production traffic.",
    ]
    return [
        Chunk(
            chunk_id=f"{DOC}:{i}",
            document_id=DOC,
            document_name=DOC,
            text=t,
            source=f"md/{DOC}.md",
            page_number=None,
            section="Authentication" if i == 0 else None,
            index=i,
            content_hash=content_hash(t),
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def indexer():
    try:
        idx = BM25Indexer()
        idx.ensure_index()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL unreachable: {exc}")
    yield idx
    with contextlib.suppress(Exception):
        idx.delete_by_document(DOC)


def test_live_round_trip(indexer: BM25Indexer) -> None:
    indexer.index_documents(_chunks())
    assert indexer.count(document_id=DOC) == 3

    res = indexer.search("OAuth tokens", top_k=5, filter_document_ids=[DOC])
    assert res, "expected at least one hit"
    assert res[0]["id"] == f"{DOC}:0"
    assert res[0]["score"] > 0
    assert res[0]["payload"]["document_id"] == DOC

    # Section-weighted: 'Authentication' section matches via weight B.
    res = indexer.search("Authentication", top_k=5, filter_document_ids=[DOC])
    assert any(r["id"] == f"{DOC}:0" for r in res)

    assert indexer.get(f"{DOC}:1")["text"].startswith("Rate limits")
    assert indexer.get("missing") is None

    assert indexer.delete_by_ids([f"{DOC}:2"]) == 1
    assert indexer.count(document_id=DOC) == 2
    assert indexer.delete_by_document(DOC) == 2
    assert indexer.count(document_id=DOC) == 0


def test_live_filters(indexer: BM25Indexer) -> None:
    indexer.index_documents(_chunks())
    other = [
        Chunk(
            chunk_id=f"{DOC}-other:0",
            document_id=f"{DOC}-other",
            document_name=f"{DOC}-other",
            text="OAuth tokens for service accounts.",
            source="md/x.md",
            page_number=None,
            section=None,
            index=0,
            content_hash=content_hash("OAuth tokens for service accounts."),
        )
    ]
    try:
        indexer.index_documents(other)
        res = indexer.search("OAuth", filter_document_ids=[DOC])
        assert {r["payload"]["document_id"] for r in res} == {DOC}
        res = indexer.search("OAuth", exclude_document_ids=[DOC])
        assert all(r["payload"]["document_id"] != DOC for r in res)
    finally:
        indexer.delete_by_document(f"{DOC}-other")
