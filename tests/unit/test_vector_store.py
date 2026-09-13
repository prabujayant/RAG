"""Unit tests for the Qdrant-backed VectorStore.

These tests use a fully in-memory fake Qdrant client — no network, no
docker. They exercise every public method and verify payloads, filters,
deletion, and dimensional validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest
from app.ingestion.chunker import Chunk, content_hash
from app.retrieval.vector import VectorStore, _point_id

POINT = _point_id  # Qdrant point id derived from a chunk_id

# ---------------------------------------------------------------- fakes

@dataclass
class _FakePoint:
    id: str
    vector: list[float]
    payload: dict

@dataclass
class _FakeCollection:
    name: str
    size: int
    distance: str
    points: dict[str, _FakePoint] = field(default_factory=dict)

class _UnexpectedResponse(Exception):
    pass

class _FakeQdrantClient:
    """In-memory stand-in for qdrant_client.QdrantClient."""

    UnexpectedResponse = _UnexpectedResponse

    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}
        self.calls: list[tuple[str, dict]] = []  # (op, kwargs) for assertions
        self._scroll_offsets: dict[str, int | None] = {}

    # admin
    def get_collections(self):
        self.calls.append(("get_collections", {}))
        return type("_R", (), {"collections": [type("_C", (), {"name": n})() for n in self.collections]})()

    def create_collection(self, collection_name, vectors_config):
        self.calls.append(("create_collection", {"name": collection_name}))
        if collection_name in self.collections:
            raise _UnexpectedResponse(f"collection {collection_name!r} already exists")
        size = vectors_config.size if hasattr(vectors_config, "size") else vectors_config["size"]
        distance = (
            vectors_config.distance
            if hasattr(vectors_config, "distance")
            else vectors_config["distance"]
        )
        self.collections[collection_name] = _FakeCollection(
            name=collection_name, size=size, distance=distance
        )

    def delete_collection(self, collection_name):
        self.calls.append(("delete_collection", {"name": collection_name}))
        if collection_name in self.collections:
            del self.collections[collection_name]

    # writes
    def upsert(self, collection_name, points, wait=True):
        coll = self.collections[collection_name]
        for p in points:
            coll.points[p.id] = _FakePoint(id=p.id, vector=list(p.vector), payload=dict(p.payload))
        return type("_R", (), {"status": "completed"})()

    # reads
    def retrieve(self, collection_name, ids, with_payload=True):
        if collection_name not in self.collections:
            raise _UnexpectedResponse("missing")
        coll = self.collections[collection_name]
        out = []
        for cid in ids:
            p = coll.points.get(cid)
            if p is not None:
                out.append(type("_P", (), {"id": p.id, "payload": p.payload})())
        return out

    def count(self, collection_name, count_filter=None, exact=True):  # noqa: ARG002
        if collection_name not in self.collections:
            raise _UnexpectedResponse("missing")
        coll = self.collections[collection_name]
        if count_filter is None:
            n = len(coll.points)
        else:
            n = 0
            # Naive but accurate for our usage: document_id == value
            for cond in count_filter.must:
                if cond.key == "document_id":
                    target = cond.match.value
                    n += sum(1 for p in coll.points.values() if p.payload.get("document_id") == target)
        return type("_R", (), {"count": n})()

    def query_points(
        self,
        collection_name,
        query,
        limit,
        query_filter=None,
        score_threshold=None,
        with_vectors=False,
        with_payload=True,
        **kwargs,  # noqa: ARG002
    ):
        coll = self.collections[collection_name]
        scored = []
        for p in coll.points.values():
            score = _cosine(query, p.vector)
            if score_threshold is not None and score < score_threshold:
                continue
            if query_filter is not None:
                wanted_docs = _extract_match_any(query_filter)
                if wanted_docs is not None and p.payload.get("document_id") not in wanted_docs:
                    continue
                banned_docs = _extract_must_not(query_filter)
                if p.payload.get("document_id") in banned_docs:
                    continue
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:limit]
        points = [
            type("_ScoredPoint", (), {"id": p.id, "score": s, "payload": p.payload})()
            for s, p in scored
        ]
        return type("_QueryResponse", (), {"points": points})()

    def delete(self, collection_name, points=None, points_selector=None, wait=True):  # noqa: ARG002
        coll = self.collections[collection_name]
        deleted = 0
        if points:
            for pid in points:
                if pid in coll.points:
                    del coll.points[pid]
                    deleted += 1
        if points_selector is not None and isinstance(points_selector, list):
            for pid in points_selector:
                if pid in coll.points:
                    del coll.points[pid]
                    deleted += 1
        if points_selector is not None and getattr(points_selector, "filter", None) is not None:
            flt = points_selector.filter
            for cond in flt.must:
                if cond.key == "document_id":
                    target = cond.match.value
                    to_delete = [
                        pid
                        for pid, p in coll.points.items()
                        if p.payload.get("document_id") == target
                    ]
                    for pid in to_delete:
                        del coll.points[pid]
                    deleted += len(to_delete)
        return {"result": {"operation_id": "op"}}

    def scroll(self, collection_name, limit, offset, with_payload=True, with_vectors=False):  # noqa: ARG002
        coll = self.collections[collection_name]
        ids = list(coll.points.keys())
        start = offset or 0
        end = start + limit
        batch = ids[start:end]
        next_offset = end if end < len(ids) else None
        return (
            [type("_P", (), {"id": coll.points[i].id, "payload": coll.points[i].payload})() for i in batch],
            next_offset,
        )

# helpers

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)

def _extract_match_any(flt) -> set[str] | None:
    for cond in getattr(flt, "must", None) or []:
        m = getattr(cond, "match", None)
        if m is not None and hasattr(m, "any"):
            return set(m.any)
    return None

def _extract_must_not(flt) -> set[str]:
    out: set[str] = set()
    for cond in getattr(flt, "must_not", None) or []:
        m = getattr(cond, "match", None)
        if m is not None and hasattr(m, "any"):
            out.update(m.any)
    return out

# ---------------------------------------------------------------- fixtures

@pytest.fixture
def fake_client() -> _FakeQdrantClient:
    return _FakeQdrantClient()

@pytest.fixture
def store(fake_client) -> VectorStore:
    return VectorStore(client=fake_client)

def _make_chunks(document_id: str, count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{document_id}:{i}",
            document_id=document_id,
            document_name=document_id,
            text=f"text {i}",
            source=f"md/{document_id}.md",
            page_number=None,
            section="s" if i == 0 else None,
            index=i,
            content_hash=content_hash(f"text {i}"),
        )
        for i in range(count)
    ]

# ---------------------------------------------------------------- tests

def test_ensure_collection_idempotent(store, fake_client) -> None:
    store.ensure_collection()
    store.ensure_collection()  # should not raise
    assert "askmydocs_chunks" in fake_client.collections

def test_ensure_collection_not_skipped_for_recycled_client_id() -> None:
    """A brand-new client must not be mistaken for an already-ensured one.

    The memoization used to be keyed on ``id(client)``. CPython recycles
    ``id()`` values after garbage collection, so a later client could reuse a
    dead client's id and have collection creation wrongly skipped — producing
    ``KeyError: 'askmydocs_chunks'``. Keys are now weak references to the
    client itself.
    """
    created: list[object] = []

    # Hold each client only weakly from the store's perspective: drop our
    # reference so the next client is likely to reuse the same memory address.
    for _ in range(30):
        client = _FakeQdrantClient()
        created.append(client)
        VectorStore(client=client).ensure_collection()
        assert "askmydocs_chunks" in client.collections

    # Force garbage collection, then ensure a fresh client still gets created.
    del created
    import gc

    gc.collect()

    fresh = _FakeQdrantClient()
    VectorStore(client=fresh).ensure_collection()
    assert "askmydocs_chunks" in fresh.collections, (
        "ensure_collection was skipped for a fresh client — memoization key "
        "collision (id() reuse)."
    )

def test_upsert_uses_real_vectors_and_payload(store, fake_client) -> None:
    chunks = _make_chunks("doc-1", 2)
    vectors = [[0.1] * 1024, [0.2] * 1024]
    n = store.upsert(chunks, vectors=vectors)
    assert n == 2
    coll = fake_client.collections["askmydocs_chunks"]
    assert len(coll.points) == 2
    for c, v in zip(chunks, vectors, strict=True):
        assert coll.points[POINT(c.chunk_id)].vector == v
        assert coll.points[POINT(c.chunk_id)].payload["text"] == c.text

def test_upsert_zero_vector_fallback(store, fake_client) -> None:
    chunks = _make_chunks("doc-1", 1)
    n = store.upsert(chunks)
    assert n == 1
    assert fake_client.collections["askmydocs_chunks"].points[POINT("doc-1:0")].vector == [0.0] * 1024

def test_upsert_rejects_wrong_vector_length(store) -> None:
    chunks = _make_chunks("doc-1", 1)
    with pytest.raises(ValueError, match="Vector length"):
        store.upsert(chunks, vectors=[[0.0] * 10])

def test_upsert_empty_noop(store, fake_client) -> None:
    n = store.upsert([])
    assert n == 0
    assert fake_client.collections == {}

def test_search_returns_scored_results(store, fake_client) -> None:
    store.ensure_collection()
    chunks = _make_chunks("doc-1", 3)
    # Construct distinct per-chunk vectors so ranking is deterministic.
    vecs = [
        [1.0] + [0.0] * 1023,
        [0.5, 1.0] + [0.0] * 1022,
        [0.0, 0.0, 1.0] + [0.0] * 1021,
    ]
    store.upsert(chunks, vectors=vecs)
    q = [1.0] + [0.0] * 1023
    results = store.search(q, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == POINT("doc-1:0")
    assert results[0]["score"] >= results[1]["score"]

def test_search_filter_document_ids(store) -> None:
    chunks_a = _make_chunks("doc-a", 2)
    chunks_b = _make_chunks("doc-b", 2)
    store.upsert(chunks_a, vectors=[[0.0] * 1024] * 2)
    store.upsert(chunks_b, vectors=[[0.0] * 1024] * 2)
    results = store.search([0.0] * 1024, top_k=10, filter_document_ids=["doc-a"])
    assert {r["payload"]["document_id"] for r in results} == {"doc-a"}

def test_search_exclude_document_ids(store) -> None:
    chunks_a = _make_chunks("doc-a", 2)
    chunks_b = _make_chunks("doc-b", 2)
    store.upsert(chunks_a, vectors=[[0.0] * 1024] * 2)
    store.upsert(chunks_b, vectors=[[0.0] * 1024] * 2)
    results = store.search([0.0] * 1024, top_k=10, exclude_document_ids=["doc-b"])
    assert {r["payload"]["document_id"] for r in results} == {"doc-a"}

def test_count_total_and_by_document(store) -> None:
    store.upsert(_make_chunks("doc-a", 2), vectors=[[0.0] * 1024] * 2)
    store.upsert(_make_chunks("doc-b", 3), vectors=[[0.0] * 1024] * 3)
    assert store.count() == 5
    assert store.count(document_id="doc-a") == 2
    assert store.count(document_id="doc-b") == 3

def test_get_returns_payload(store) -> None:
    chunks = _make_chunks("doc-1", 1)
    store.upsert(chunks, vectors=[[0.0] * 1024])
    res = store.get("doc-1:0")
    assert res is not None
    assert res["payload"]["text"] == "text 0"

def test_get_missing_returns_none(store) -> None:
    store.ensure_collection()
    assert store.get("does-not-exist") is None

def test_delete_by_ids(store) -> None:
    chunks = _make_chunks("doc-1", 3)
    store.upsert(chunks, vectors=[[0.0] * 1024] * 3)
    n = store.delete_by_ids(["doc-1:0", "doc-1:2"])
    assert n == 2
    assert store.count() == 1

def test_delete_by_document(store) -> None:
    store.upsert(_make_chunks("doc-a", 2), vectors=[[0.0] * 1024] * 2)
    store.upsert(_make_chunks("doc-b", 3), vectors=[[0.0] * 1024] * 3)
    store.delete_by_document("doc-a")
    assert store.count(document_id="doc-a") == 0
    assert store.count(document_id="doc-b") == 3

def test_iter_all_streams_payloads(store) -> None:
    chunks = _make_chunks("doc-1", 4)
    store.upsert(chunks, vectors=[[0.0] * 1024] * 4)
    seen = {row["id"] for row in store.iter_all(batch_size=2)}
    assert seen == {POINT(c.chunk_id) for c in chunks}
