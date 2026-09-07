"""Unit tests for the OpenSearch-backed BM25Indexer.

A fully in-memory fake OpenSearch client is used — no network, no docker.
We exercise the public API (ensure/index/search/get/count/delete/multi_match).
"""

from __future__ import annotations

from collections import Counter

import pytest
from app.ingestion.chunker import Chunk, content_hash
from app.retrieval.bm25 import BM25Indexer
from opensearchpy.exceptions import NotFoundError

# ---------------------------------------------------------------- fake OS

class _FakeOpenSearch:
    """Minimal OpenSearch stand-in: supports indices, bulk, get, count, search."""

    # Use the real opensearchpy NotFoundError so production code paths that
    # catch it also work against the fake.
    NotFoundError = NotFoundError

    def __init__(self) -> None:
        self.indices_map: dict[str, dict] = {}
        self.docs: dict[str, dict[str, dict]] = {}  # index -> id -> source
        self.last_bulk: list[dict] = []

    class _IndicesClient:
        def __init__(self, parent: _FakeOpenSearch) -> None:
            self.parent = parent

        def exists(self, index):
            return index in self.parent.indices_map

        def create(self, index, body):
            if index in self.parent.indices_map:
                return {}
            self.parent.indices_map[index] = body
            self.parent.docs.setdefault(index, {})
            return {}

        def delete(self, index):
            self.parent.indices_map.pop(index, None)
            self.parent.docs.pop(index, None)
            return {}

    @property
    def indices(self) -> _IndicesClient:
        return _FakeOpenSearch._IndicesClient(self)

    # ---- docs
    def bulk(self, body, refresh=False):  # noqa: ARG002
        self.last_bulk = body
        items = []
        i = 0
        while i < len(body):
            op = body[i]
            if "index" in op:
                meta = op["index"]
                idx = meta["_index"]
                doc_id = meta["_id"]
                source = body[i + 1]
                self.docs.setdefault(idx, {})[doc_id] = source
                items.append({"index": {"status": 201}})
                i += 2
            elif "delete" in op:
                meta = op["delete"]
                idx = meta["_index"]
                doc_id = meta["_id"]
                existed = self.docs.get(idx, {}).pop(doc_id, None) is not None
                items.append({"delete": {"status": 200 if existed else 404}})
                i += 1
            else:
                i += 1
        return {"errors": False, "items": items}

    def get(self, index, id):
        if index not in self.indices_map or id not in self.docs.get(index, {}):
            raise self.NotFoundError()
        return {"_source": self.docs[index][id]}

    def count(self, index, body):
        if index not in self.indices_map:
            raise self.NotFoundError()
        if body["query"] == {"match_all": {}}:
            return {"count": len(self.docs.get(index, {}))}
        # term: document_id
        term = body["query"].get("term", {}).get("document_id")
        n = sum(1 for d in self.docs[index].values() if d.get("document_id") == term)
        return {"count": n}

    def search(self, index, body):
        if index not in self.indices_map:
            return {"hits": {"hits": []}}
        size = body.get("size", 10)
        must = body["query"]["bool"]["must"]
        flt = body["query"]["bool"].get("filter", [])
        query = must[0]["multi_match"]["query"]
        fields = must[0]["multi_match"]["fields"]
        wanted: set[str] = set()
        for d in flt:
            if "terms" in d:
                val = d["terms"]["document_id"]
                if isinstance(val, list):
                    wanted.update(val)
                else:
                    wanted.add(val)
        wanted = set(next(iter(wanted))) if wanted else None

        scored = []
        for doc_id, src in self.docs[index].items():
            if wanted is not None and src.get("document_id") not in wanted:
                continue
            score = _bm25_score(query, fields, src)
            scored.append((score, doc_id, src))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:size]
        return {
            "hits": {
                "hits": [
                    {"_id": did, "_score": s, "_source": src} for s, did, src in scored
                ]
            }
        }

    def delete_by_query(self, index, body, refresh=True, conflicts="proceed"):  # noqa: ARG002
        if index not in self.indices_map:
            raise self.NotFoundError()
        term = body["query"].get("term", {}).get("document_id")
        to_delete = [d for d, s in self.docs[index].items() if s.get("document_id") == term]
        for d in to_delete:
            del self.docs[index][d]
        return {"deleted": len(to_delete)}

# ---------------------------------------------------------------- helpers

def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]

def _bm25_score(query: str, fields: list[str], doc: dict) -> float:
    """Toy BM25-like score so our tests are deterministic & inspectable."""
    q_tokens = _tokens(query)
    score = 0.0
    k1, b = 1.5, 0.75
    doc_tokens: list[str] = []
    for f in fields:
        v = doc.get(f)
        if isinstance(v, str):
            doc_tokens.extend(_tokens(v))
    if not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    avg_dl = 1.0
    for qt in q_tokens:
        f = tf.get(qt, 0)
        if f == 0:
            continue
        denom = f + k1 * (1 - b + b * doc_len / max(avg_dl, 1e-9))
        score += (f * (k1 + 1)) / max(denom, 1e-9)
    return score

# ---------------------------------------------------------------- fixtures

@pytest.fixture
def fake_os() -> _FakeOpenSearch:
    return _FakeOpenSearch()

@pytest.fixture
def indexer(fake_os) -> BM25Indexer:
    return BM25Indexer(client=fake_os)

def _make_chunks(document_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{document_id}:{i}",
            document_id=document_id,
            document_name=document_id,
            text=t,
            source=f"md/{document_id}.md",
            page_number=None,
            section=None,
            index=i,
            content_hash=content_hash(t),
        )
        for i, t in enumerate(texts)
    ]

# ---------------------------------------------------------------- tests

def test_ensure_index_creates_standard_analyzer(indexer, fake_os) -> None:
    indexer.ensure_index()
    body = fake_os.indices_map["askmydocs_chunks"]
    assert body["mappings"]["properties"]["text"]["type"] == "text"
    assert body["settings"]["analysis"]["analyzer"]["default"]["type"] == "standard"

def test_ensure_index_idempotent(indexer) -> None:
    indexer.ensure_index()
    indexer.ensure_index()  # should not raise

def test_index_documents_writes_payload(indexer, fake_os) -> None:
    chunks = _make_chunks("auth", ["OAuth tokens expire after 60 minutes.", "Rate limits are hourly."])
    indexer.index_documents(chunks)
    docs = fake_os.docs["askmydocs_chunks"]
    assert set(docs.keys()) == {"auth:0", "auth:1"}
    assert docs["auth:0"]["text"] == "OAuth tokens expire after 60 minutes."

def test_index_documents_empty_noop(indexer, fake_os) -> None:
    indexer.index_documents([])
    assert fake_os.docs == {}

def test_upsert_returns_count(indexer) -> None:
    chunks = _make_chunks("auth", ["a", "b", "c"])
    assert indexer.upsert(chunks) == 3

def test_search_basic(indexer) -> None:
    chunks = _make_chunks(
        "auth",
        [
            "OAuth access tokens expire after 60 minutes.",
            "Rate limits reset every hour.",
            "Use HTTPS for all production traffic.",
        ],
    )
    indexer.index_documents(chunks)
    res = indexer.search("OAuth tokens")
    assert len(res) >= 1
    assert res[0]["id"] == "auth:0"
    assert res[0]["score"] > 0
    assert "OAuth" in res[0]["payload"]["text"]

def test_search_filter_document_ids(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["OAuth tokens"]))
    indexer.index_documents(_make_chunks("b", ["OAuth tokens"]))
    res = indexer.search("OAuth", filter_document_ids=["a"])
    assert {r["payload"]["document_id"] for r in res} == {"a"}

def test_search_multi_match_fields(indexer) -> None:
    chunks = _make_chunks("a", ["plain body", "another body"])
    # Force the second chunk's section field to match the query.
    chunks[1].section = "OAuth troubleshooting"
    indexer.index_documents(chunks)
    res = indexer.search("OAuth", fields=["text", "section"])
    assert res[0]["id"] == "a:1"

def test_count_and_by_document(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["x", "y"]))
    indexer.index_documents(_make_chunks("b", ["z"]))
    assert indexer.count() == 3
    assert indexer.count(document_id="a") == 2

def test_get_returns_source(indexer) -> None:
    chunks = _make_chunks("a", ["hello world"])
    indexer.index_documents(chunks)
    src = indexer.get("a:0")
    assert src is not None
    assert src["text"] == "hello world"

def test_get_missing_returns_none(indexer) -> None:
    assert indexer.get("nope") is None

def test_delete_by_document(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["x", "y"]))
    indexer.index_documents(_make_chunks("b", ["z"]))
    assert indexer.delete_by_document("a") == 2
    assert indexer.count(document_id="a") == 0
    assert indexer.count() == 1

def test_delete_by_ids(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["x", "y", "z"]))
    assert indexer.delete_by_ids(["a:0", "a:2"]) == 2
    assert indexer.count() == 1

def test_delete_by_ids_empty(indexer) -> None:
    assert indexer.delete_by_ids([]) == 0

def test_delete_index_is_idempotent(indexer, fake_os) -> None:
    indexer.ensure_index()
    indexer.delete_index()
    indexer.delete_index()  # no raise
    assert "askmydocs_chunks" not in fake_os.indices_map

def test_count_when_index_missing_returns_zero(indexer) -> None:
    assert indexer.count() == 0
    assert indexer.delete_by_document("nope") == 0
