"""Unit tests for the BGE-M3 embedding wrapper.

These tests do NOT load the real model (4+ GB). They inject a stub
sentence-transformer encoder so we can exercise the public API end-to-end
including dimension validation, doc/query asymmetry, and the Qdrant upsert
round-trip via a fake VectorStore.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.config import get_settings
from app.embeddings.embedder import BGE_M3_DIM, Embedder
from app.ingestion.chunker import Chunk, Chunker, ChunkingStrategy, content_hash


class _StubEncoder:
    """Mimics sentence_transformers.SentenceTransformer.encode()."""

    def __init__(self, dim: int = BGE_M3_DIM) -> None:
        self.dim = dim
        self.last_prompt: str | None = None
        self.call_count = 0

    def encode(self, texts, batch_size=16, normalize_embeddings=True,
               show_progress_bar=False, prompt=None):  # noqa: ANN001
        self.call_count += 1
        self.last_prompt = prompt
        # Deterministic, normalized fake vectors seeded by text length.
        rng = np.random.default_rng(seed=len(texts))
        out = rng.standard_normal((len(texts), self.dim)).astype("float32")
        if normalize_embeddings:
            out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)
        return out

@pytest.fixture
def stub_encoder() -> _StubEncoder:
    return _StubEncoder()

@pytest.fixture
def embedder(stub_encoder) -> Embedder:
    e = Embedder()
    e._model = stub_encoder  # inject the stub
    return e

@pytest.fixture
def sample_chunks() -> list[Chunk]:
    chunker = Chunker(strategy=ChunkingStrategy.FIXED, chunk_size=40, chunk_overlap=4)
    return chunker.chunk_text(
        "OAuth tokens expire after 60 minutes. Rate limits reset hourly. "
        "Always use HTTPS for production traffic. Backup before deploys.",
        document_id="auth-guide",
        document_name="Auth Guide",
        source="md/auth.md",
    )

# ---------------------------------------------------------------- defaults

def test_default_settings_loaded() -> None:
    e = Embedder()
    assert e.model_name == "BAAI/bge-m3"
    assert e.vector_size == get_settings().embedding_dim == 1024

def test_model_lazy_loaded_on_access(embedder) -> None:
    # Accessing .model should NOT trigger reload (it was injected).
    assert isinstance(embedder.model, _StubEncoder)

# --------------------------------------------------------------- embed_xxx

def test_embed_documents_returns_one_vector_per_text(embedder, stub_encoder) -> None:
    out = embedder.embed_documents(["hello", "world"])
    assert len(out) == 2
    assert all(len(v) == BGE_M3_DIM for v in out)
    assert stub_encoder.call_count == 1

def test_embed_queries_uses_query_prompt(embedder, stub_encoder) -> None:
    embedder.embed_queries(["how do I authenticate?"])
    assert stub_encoder.last_prompt is not None
    assert "searching" in stub_encoder.last_prompt.lower()

def test_embed_queries_falls_back_when_prompt_unsupported() -> None:
    """Older sentence-transformers don't accept ``prompt=``; we must not crash."""

    class _NoPromptEncoder(_StubEncoder):
        def encode(self, texts, **kwargs):  # noqa: ANN001
            if "prompt" in kwargs:
                raise TypeError("unexpected kw: prompt")
            return super().encode(texts, **kwargs)

    e = Embedder()
    e._model = _NoPromptEncoder()
    out = e.embed_queries(["q1"])
    assert len(out) == 1

def test_embed_documents_empty_returns_empty(embedder, stub_encoder) -> None:
    assert embedder.embed_documents([]) == []
    assert embedder.embed_queries([]) == []
    assert stub_encoder.call_count == 0

def test_embed_alias_uses_document_mode(embedder, stub_encoder) -> None:
    embedder.embed(["a", "b"])
    assert stub_encoder.last_prompt is None  # default prompt for documents

# -------------------------------------------------------------- dim guard

def test_dim_mismatch_raises(embedder) -> None:
    # Force the configured size to disagree with what the stub produces.
    embedder.vector_size = BGE_M3_DIM + 1
    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        embedder.embed_documents(["hi"])

# ----------------------------------------------------------------- upsert

class _FakeVectorStore:
    def __init__(self) -> None:
        self.points: list[dict] = []
        self.calls = 0

    def upsert(self, chunks, vectors=None) -> int:  # noqa: ANN001
        self.calls += 1
        batch_size = 0
        for i, c in enumerate(chunks):
            self.points.append({"id": c.chunk_id, "vector": vectors[i] if vectors else None})
            batch_size += 1
        return batch_size

def test_embed_and_upsert_uses_real_vectors(embedder, sample_chunks) -> None:
    fake = _FakeVectorStore()
    embedder._vector_store = fake
    n = embedder.embed_and_upsert(sample_chunks)
    assert n == len(sample_chunks)
    assert len(fake.points) == len(sample_chunks)
    # Every point must carry the real vector, never None.
    for p in fake.points:
        assert p["vector"] is not None
        assert len(p["vector"]) == BGE_M3_DIM

def test_embed_and_upsert_empty_noop(embedder) -> None:
    fake = _FakeVectorStore()
    embedder._vector_store = fake
    assert embedder.embed_and_upsert([]) == 0
    assert fake.calls == 0

# ---------------------------------------------------------- content hash

def test_content_hash_used_in_payload() -> None:
    chunk = Chunk(
        chunk_id="doc-1:0",
        document_id="doc-1",
        document_name="Doc",
        text="hello",
        source="s",
        page_number=None,
        section=None,
        index=0,
        content_hash=content_hash("hello"),
    )
    assert chunk.content_hash == content_hash("hello")
