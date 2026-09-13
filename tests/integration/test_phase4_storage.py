"""End-to-end Phase 4 integration tests.

These tests wire the full Phase 4 stack together:

    chunker → embedder → VectorStore (Qdrant) → BM25Indexer (Postgres tsvector)

They use **in-memory fakes** for Qdrant and keyword postings (no docker) so
the suite runs in CI without infra. The embedder uses a tiny deterministic
stub encoder (no BGE-M3 download) but preserves the real shape of the
data flowing through the stack. Real tsvector ranking is covered by the
live-Postgres smoke test (requires_docker).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest
from app.embeddings.embedder import BGE_M3_DIM, Embedder
from app.ingestion.chunker import Chunker, ChunkingStrategy
from app.retrieval.vector import VectorStore

# ---------------------------------------------------------------- fakes

@dataclass
class _Point:
    id: str
    vector: list[float]
    payload: dict

@dataclass
class _Coll:
    name: str
    size: int
    distance: str
    points: dict[str, _Point] = field(default_factory=dict)

class _UnexpectedResponse(Exception):
    pass

class _FakeQdrant:
    UnexpectedResponse = _UnexpectedResponse

    def __init__(self) -> None:
        self.collections: dict[str, _Coll] = {}

    def get_collections(self):
        return type(
            "_R", (), {"collections": [type("_C", (), {"name": n})() for n in self.collections]}
        )()

    def create_collection(self, collection_name, vectors_config):
        if collection_name in self.collections:
            raise _UnexpectedResponse("already exists")
        self.collections[collection_name] = _Coll(
            name=collection_name,
            size=vectors_config.size,
            distance=vectors_config.distance,
        )

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def upsert(self, collection_name, points, wait=True):  # noqa: ARG002
        coll = self.collections[collection_name]
        for p in points:
            coll.points[p.id] = _Point(p.id, list(p.vector), dict(p.payload))

    def retrieve(self, collection_name, ids, with_payload=True):  # noqa: ARG002
        coll = self.collections[collection_name]
        return [
            type("_P", (), {"id": coll.points[i].id, "payload": coll.points[i].payload})()
            for i in ids if i in coll.points
        ]

    def count(self, collection_name, count_filter=None, exact=True):  # noqa: ARG002
        coll = self.collections[collection_name]
        n = len(coll.points)
        if count_filter is not None:
            for cond in count_filter.must:
                if cond.key == "document_id":
                    target = cond.match.value
                    n = sum(1 for p in coll.points.values() if p.payload.get("document_id") == target)
                    break
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
        wanted = None
        if query_filter is not None:
            for cond in query_filter.must:
                if cond.key == "document_id" and hasattr(cond.match, "any"):
                    wanted = set(cond.match.any)
        scored = []
        for p in coll.points.values():
            if wanted is not None and p.payload.get("document_id") not in wanted:
                continue
            score = _cosine(query, p.vector)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        points = [
            type("_ScoredPoint", (), {"id": p.id, "score": s, "payload": p.payload})()
            for s, p in scored[:limit]
        ]
        return type("_QueryResponse", (), {"points": points})()

    def delete(self, collection_name, points=None, points_selector=None, wait=True):  # noqa: ARG002
        coll = self.collections[collection_name]
        if points:
            for pid in points:
                coll.points.pop(pid, None)
        if points_selector is not None and isinstance(points_selector, list):
            for pid in points_selector:
                coll.points.pop(pid, None)
        elif points_selector is not None:
            flt = points_selector.filter
            for cond in flt.must:
                if cond.key == "document_id":
                    target = cond.match.value
                    for pid in [
                        pid
                        for pid, p in coll.points.items()
                        if p.payload.get("document_id") == target
                    ]:
                        coll.points.pop(pid, None)

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=False, **kwargs):  # noqa: ARG002
        coll = self.collections[collection_name]
        all_ids = list(coll.points.keys())
        start = 0
        if offset is not None:
            try:
                start = all_ids.index(offset) + 1
            except ValueError:
                start = 0
        page_ids = all_ids[start : start + limit]
        next_offset = all_ids[start + limit] if start + limit < len(all_ids) else None
        points = [coll.points[pid] for pid in page_ids]
        return points, next_offset

def _cosine(a, b):  # noqa: ANN001
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)

class _FakeKeyword:
    """In-memory stand-in with the BM25Indexer method surface.

    Tests pipeline wiring (every chunk reaches every store); real ranking
    lives in Postgres and is covered by the live-PG smoke test.
    """

    def __init__(self) -> None:
        self.postings: dict[str, dict] = {}

    def ensure_index(self) -> None:
        return None

    def index_documents(self, chunks, refresh: bool = True) -> None:  # noqa: ARG002, ANN001
        for c in chunks:
            self.postings[c.chunk_id] = {
                "document_id": c.document_id,
                "text": c.text,
                "source": c.source,
                "page_number": c.page_number,
                "section": c.section,
            }

    def upsert(self, chunks) -> int:  # noqa: ANN001
        items = list(chunks)
        self.index_documents(items)
        return len(items)

    def search(self, query, top_k=20, filter_document_ids=None, exclude_document_ids=None, fields=None):  # noqa: ANN001
        q = (query or "").lower()
        hits = []
        for pid, src in self.postings.items():
            if filter_document_ids and src["document_id"] not in filter_document_ids:
                continue
            if exclude_document_ids and src["document_id"] in exclude_document_ids:
                continue
            text = (src.get("text") or "").lower()
            score = float(text.count(q)) if q and q in text else 0.0
            if score > 0:
                hits.append({"id": pid, "score": score, "payload": dict(src)})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    def get(self, chunk_id):
        src = self.postings.get(chunk_id)
        return dict(src) if src else None

    def count(self, document_id=None):
        if document_id is None:
            return len(self.postings)
        return sum(1 for s in self.postings.values() if s["document_id"] == document_id)

    def delete_by_document(self, document_id, refresh: bool = True) -> int:  # noqa: ARG002
        doomed = [p for p, s in self.postings.items() if s["document_id"] == document_id]
        for p in doomed:
            del self.postings[p]
        return len(doomed)

    def delete_by_ids(self, chunk_ids, refresh: bool = True) -> int:  # noqa: ARG002
        n = 0
        for cid in chunk_ids:
            if self.postings.pop(cid, None) is not None:
                n += 1
        return n

    def delete_index(self) -> None:
        self.postings.clear()

# ---------------------------------------------------------------- embedder stub

class _StubEncoder:
    def __init__(self, dim: int = BGE_M3_DIM) -> None:
        self.dim = dim

    def encode(self, texts, batch_size=16, normalize_embeddings=True,
               show_progress_bar=False, prompt=None):  # noqa: ANN001
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            # Deterministic dense vector: project text tokens into fixed dims.
            for j, ch in enumerate(t[: self.dim]):
                out[i, (j * 31 + ord(ch)) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)
            out = out / norms
        return out

# ---------------------------------------------------------------- fixtures

@pytest.fixture
def qdrant() -> _FakeQdrant:
    return _FakeQdrant()

@pytest.fixture
def keyword() -> _FakeKeyword:
    return _FakeKeyword()

@pytest.fixture
def embedder(qdrant, keyword):  # noqa: ARG001
    e = Embedder()
    e._model = _StubEncoder()
    return e

@pytest.fixture
def vector_store(qdrant) -> VectorStore:
    return VectorStore(client=qdrant)

@pytest.fixture
def bm25(keyword) -> _FakeKeyword:
    return keyword

# -------------------------------------------------------------- helpers

def _ingest(embedder, vector_store, bm25, doc_id: str, raw_text: str):  # noqa: ANN001
    chunker = Chunker(strategy=ChunkingStrategy.FIXED, chunk_size=80, chunk_overlap=10)
    chunks = chunker.chunk_text(
        raw_text,
        document_id=doc_id,
        document_name=doc_id,
        source=f"md/{doc_id}.md",
    )
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)
    vector_store.upsert(chunks, vectors=vectors)
    bm25.index_documents(chunks)
    return chunks

# ---------------------------------------------------------------- tests

def test_ensure_collections_are_idempotent(vector_store, bm25) -> None:
    vector_store.ensure_collection()
    vector_store.ensure_collection()
    bm25.ensure_index()
    bm25.ensure_index()

def test_end_to_end_chunk_embed_index(vector_store, bm25, embedder) -> None:
    chunks = _ingest(
        embedder,
        vector_store,
        bm25,
        "auth-guide",
        "OAuth access tokens expire after 60 minutes. "
        "Rate limits are hourly. Always use HTTPS for production traffic.",
    )
    # Every chunk is in every store.
    assert vector_store.count() == len(chunks)
    assert bm25.count() == len(chunks)
    assert {c.chunk_id for c in chunks}.issubset(
        {p["id"] for p in vector_store.iter_all()}
    )

def test_vector_search_returns_relevant_chunks(vector_store, bm25, embedder) -> None:
    _ingest(
        embedder,
        vector_store,
        bm25,
        "auth",
        "OAuth access tokens expire after 60 minutes. Use HTTPS in production.",
    )
    _ingest(
        embedder,
        vector_store,
        bm25,
        "monitoring",
        "PagerDuty alerts fire when latency exceeds 2 seconds over 5 minutes.",
    )
    # Build a query vector that should match "auth" — concatenate token chars.
    q = embedder.embed_queries(["How long do OAuth tokens last?"])[0]
    results = vector_store.search(q, top_k=2, filter_document_ids=["auth"])
    assert results
    assert all(r["payload"]["document_id"] == "auth" for r in results)

def test_bm25_search_finds_keyword_match(vector_store, bm25, embedder) -> None:
    _ingest(
        embedder,
        vector_store,
        bm25,
        "auth",
        "OAuth access tokens expire after 60 minutes. Use HTTPS in production.",
    )
    _ingest(
        embedder,
        vector_store,
        bm25,
        "monitoring",
        "PagerDuty alerts fire when latency exceeds 2 seconds.",
    )
    res = bm25.search("PagerDuty", top_k=2)
    assert res
    assert res[0]["payload"]["document_id"] == "monitoring"

def test_delete_chunks_by_document_clears_all_stores(vector_store, bm25, embedder) -> None:
    _ingest(embedder, vector_store, bm25, "auth", "OAuth access tokens expire after 60 minutes.")
    _ingest(embedder, vector_store, bm25, "monitoring", "PagerDuty alerts fire when latency is high.")
    assert vector_store.count() == 2 and bm25.count() == 2
    vector_store.delete_by_document("auth")
    bm25.delete_by_document("auth")
    assert vector_store.count(document_id="auth") == 0
    assert bm25.count(document_id="auth") == 0
    assert vector_store.count() == 1 and bm25.count() == 1

def test_ingestion_pipeline_writes_to_all_stores(  # noqa: E501
    monkeypatch, qdrant, keyword, embedder, vector_store, bm25
) -> None:
    """End-to-end through the real IngestionPipeline but with fake stores."""
    # Patch the DB engine to an in-memory SQLite so the test does not need PG.
    from app.db import session as db_session
    from app.db.models import Base
    from app.ingestion.pipeline import IngestionPipeline
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False)

    monkeypatch.setattr(db_session, "engine", test_engine)
    monkeypatch.setattr(db_session, "SessionLocal", TestSession)

    # Patch the store classes in the modules they are imported from inside pipeline.
    from app.retrieval import bm25 as b_mod
    from app.retrieval import vector as v_mod

    monkeypatch.setattr(v_mod, "VectorStore", lambda *_a, **_k: VectorStore(settings=None, client=qdrant))
    monkeypatch.setattr(b_mod, "BM25Indexer", lambda *_a, **_k: keyword)

    pipeline = IngestionPipeline(embedder=embedder)
    res = pipeline.ingest("data/corpus/markdown/authentication-guide.md", title="Authentication Guide")
    assert res.chunk_count > 0
    assert vector_store.count() == res.chunk_count
    assert bm25.count() == res.chunk_count
