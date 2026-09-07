"""Unit tests for the retrieval domain models (Phase 2)."""

from __future__ import annotations

from app.retrieval.models import RetrievalResult, RetrieverType


def test_retriever_type_values() -> None:
    assert RetrieverType.VECTOR.value == "vector"
    assert RetrieverType.BM25.value == "bm25"
    assert RetrieverType.HYBRID.value == "hybrid"

def test_retrieval_result_defaults() -> None:
    r = RetrievalResult(
        chunk_id="doc-1:0",
        document_id="doc-1",
        text="hello",
        score=0.9,
        source="md/guide.md",
    )
    assert r.page_number is None
    assert r.section is None
    assert r.retriever is RetrieverType.VECTOR
    assert r.rank == 0
    assert r.metadata == {}

def test_retrieval_result_all_fields() -> None:
    r = RetrievalResult(
        chunk_id="doc-1:3",
        document_id="doc-1",
        text="some text",
        score=0.5,
        source="pdf/a.pdf",
        page_number=4,
        section="Auth",
        retriever=RetrieverType.BM25,
        rank=2,
        metadata={"content_hash": "abc"},
    )
    assert r.page_number == 4
    assert r.section == "Auth"
    assert r.retriever is RetrieverType.BM25
    assert r.rank == 2
    assert r.metadata["content_hash"] == "abc"

def test_to_dict_shape() -> None:
    r = RetrievalResult(
        chunk_id="doc-1:0",
        document_id="doc-1",
        text="hello",
        score=0.9,
        source="md/guide.md",
        retriever=RetrieverType.HYBRID,
        rank=1,
    )
    d = r.to_dict()
    assert d["chunk_id"] == "doc-1:0"
    assert d["document_id"] == "doc-1"
    assert d["text"] == "hello"
    assert d["score"] == 0.9
    assert d["source"] == "md/guide.md"
    assert d["retriever"] == "hybrid"
    assert d["rank"] == 1
    assert d["metadata"] == {}

def test_round_trip() -> None:
    r = RetrievalResult(
        chunk_id="doc-1:2",
        document_id="doc-1",
        text="round trip",
        score=0.42,
        source="html/page.html",
        page_number=7,
        section="Errors",
        retriever=RetrieverType.BM25,
        rank=3,
        metadata={"key": "value"},
    )
    restored = RetrievalResult.from_dict(r.to_dict())
    assert restored == r
    assert restored.retriever is RetrieverType.BM25

def test_from_dict_defaults_retriever() -> None:
    d = {
        "chunk_id": "doc-1:0",
        "document_id": "doc-1",
        "text": "x",
        "score": 0.1,
        "source": "s",
    }
    restored = RetrievalResult.from_dict(d)
    assert restored.retriever is RetrieverType.VECTOR
    assert restored.page_number is None
    assert restored.section is None
    assert restored.rank == 0
