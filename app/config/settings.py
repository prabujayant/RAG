"""Strongly-typed, environment-driven configuration for AskMyDocs.

All configuration flows through a single :class:`Settings` object produced by
pydantic-settings. Every knobs named in the spec is covered here and can be
overridden through environment variables or a ``.env`` file, keeping the rest
of the codebase free of magic strings.

Example::

    settings = get_settings()
    settings.embedding_model          # -> "BAAI/bge-m3"
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application -------------------------------------------------------
    app_env: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="pretty",
        description="Log format: 'json' for production, 'pretty' for development",
    )
    enable_tracing: bool = Field(
        default=False,
        description="Master toggle for distributed tracing (Langfuse when creds present)",
    )
    enable_structured_logging: bool = Field(
        default=True,
        description="Emit structured (JSON) logs when log_format=json",
    )

    # ---- Metadata database (PostgreSQL) ------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg://askmydocs:askmydocs@localhost:5432/askmydocs",
        description="SQLAlchemy connection string for application metadata",
    )

    # ---- Qdrant (vector search) --------------------------------------------
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="askmydocs_chunks")
    qdrant_distance: str = Field(default="Cosine", description="Similarity metric (Cosine | Euclid | Dot)")

    # ---- OpenSearch (BM25) --------------------------------------------------
    opensearch_url: str = Field(default="http://localhost:9200")
    opensearch_username: str = Field(default="")
    opensearch_password: str = Field(default="")
    opensearch_index: str = Field(default="askmydocs_chunks")

    # ---- Embeddings ----------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)
    embedding_batch_size: int = Field(default=16)

    # ---- Re-ranker ------------------------------------------------------------
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    enable_reranker: bool = Field(default=True)
    rerank_top_k: int = Field(default=8, ge=1)

    # ---- Chunking -------------------------------------------------------------
    chunk_strategy: str = Field(default="paragraph")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ---- Retrieval -------------------------------------------------------------
    bm25_top_k: int = Field(default=20, ge=1)
    vector_top_k: int = Field(default=20, ge=1)
    hybrid_top_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    final_context_k: int = Field(default=5, ge=1)
    max_context_tokens: int = Field(default=3000, ge=1)

    # ---- LLM (OpenRouter) ------------------------------------------------------
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="openai/gpt-4o-mini")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_timeout_seconds: float = Field(default=60.0)
    llm_max_retries: int = Field(default=2)

    # ---- Grounding / answers ----------------------------------------------------
    max_answer_tokens: int = Field(default=400)
    confidence_threshold: float = Field(default=0.5)

    # ---- Observability (Langfuse) ------------------------------------------------
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="")
    langfuse_prefix: str = Field(default="askmydocs/")

    # ---- Evaluation ----------------------------------------------------------------
    eval_dataset_path: str = Field(default="evals/dataset/golden.jsonl")
    eval_rerank_enabled: bool = Field(default=True)


    @property
    def langfuse_enabled(self) -> bool:
        """True when Langfuse credentials are present (optional feature)."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


def reload_settings() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()