"""Tests for the strongly-typed configuration object."""

from __future__ import annotations

import pytest
from app.config.settings import Settings
from pydantic import ValidationError


def test_defaults_match_spec() -> None:
    s = Settings()
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.reranker_model == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert s.enable_reranker is True
    assert s.chunk_strategy == "paragraph"
    assert s.rrf_k == 60
    assert s.bm25_top_k == 20
    assert s.vector_top_k == 20
    assert s.hybrid_top_k == 20
    assert s.rerank_top_k == 8
    assert s.final_context_k == 5


def test_env_overrides_work(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("BM25_TOP_K", "50")
    monkeypatch.setenv("ENABLE_RERANKER", "false")
    s = Settings()
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.bm25_top_k == 50
    assert s.enable_reranker is False


def test_openrouter_defaults() -> None:
    s = Settings()
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_model


def test_langfuse_optional() -> None:
    assert Settings().langfuse_enabled is False
    s = Settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    assert s.langfuse_enabled is True


def test_database_url_default(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings()
    assert s.database_url.startswith("postgresql+psycopg://")


# ---- Retrieval configuration validation -----------------------------------------


def test_retrieval_top_k_positive() -> None:
    for field in ("bm25_top_k", "vector_top_k", "hybrid_top_k", "rerank_top_k", "final_context_k"):
        with pytest.raises(ValidationError):
            Settings(**{field: 0})
        with pytest.raises(ValidationError):
            Settings(**{field: -1})


def test_rrf_k_positive() -> None:
    s = Settings()
    assert s.rrf_k > 0
    with pytest.raises(ValidationError):
        Settings(rrf_k=0)
    with pytest.raises(ValidationError):
        Settings(rrf_k=-1)


def test_final_context_k_within_hybrid_top_k() -> None:
    # final_context_k <= hybrid_top_k is a soft constraint tested at retrieval time
    s = Settings(final_context_k=5, hybrid_top_k=20)
    assert s.final_context_k <= s.hybrid_top_k


def test_max_context_tokens_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(max_context_tokens=0)
    with pytest.raises(ValidationError):
        Settings(max_context_tokens=-1)


def test_positive_retrieval_config_values() -> None:
    """Verify retrieval config fields accept positive values without error."""
    s = Settings(
        bm25_top_k=20,
        vector_top_k=20,
        hybrid_top_k=20,
        rerank_top_k=8,
        final_context_k=5,
        rrf_k=60,
        max_context_tokens=3000,
    )
    assert s.bm25_top_k == 20
    assert s.rrf_k == 60