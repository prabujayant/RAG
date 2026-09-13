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
    cors_origins: str = Field(
        default="*",
        description=(
            "Comma-separated CORS origins (e.g. 'https://askmydocs.vercel.app'). "
            "Default '*' keeps local dev working; set explicitly in production."
        ),
    )
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

    # ---- Embeddings ----------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)
    embedding_batch_size: int = Field(default=64)
    warmup_models: bool = Field(
        default=True,
        description=(
            "Pre-load the embedding model during app startup so the first "
            "query doesn't pay the ~50s cold-start cost. Set to false to "
            "keep startup fast (e.g. in tests)."
        ),
    )

    # ---- Re-ranker ------------------------------------------------------------
    # Multilingual MiniLM cross-encoder (~118M params). Measured 19.6x faster
    # than BAAI/bge-reranker-v2-m3 (568M) on CPU: 404 ms/pair vs 7898 ms/pair,
    # with strong ranking margins across EN/DE/ID. The larger model is still
    # selectable via RERANKER_MODEL when maximum accuracy is required.
    reranker_model: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    enable_reranker: bool = Field(default=True)
    rerank_top_k: int = Field(default=8, ge=1)

    # ---- Chunking -------------------------------------------------------------
    chunk_strategy: str = Field(default="paragraph")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ---- Ingestion / uploads ----------------------------------------------------
    upload_dir: str = Field(
        default="data/uploads",
        description="Directory where uploaded documents are persisted before ingestion",
    )

    # ---- Retrieval -------------------------------------------------------------
    bm25_top_k: int = Field(default=20, ge=1)
    vector_top_k: int = Field(default=20, ge=1)
    hybrid_top_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    final_context_k: int = Field(default=5, ge=1)
    max_context_tokens: int = Field(default=3000, ge=1)

    # ---- LLM (OpenRouter) ------------------------------------------------------
    openrouter_api_key: str = Field(default="")
    # Fastest free model measured (median 1.6s, valid JSON + citations on 3/3
    # runs). See .env for the full free-model benchmark. Reasoning models emit
    # hidden reasoning tokens, so max_answer_tokens must stay high.
    openrouter_model: str = Field(default="nex-agi/nex-n2.5-mini:free")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_timeout_seconds: float = Field(
        default=180.0,
        description=(
            "HTTP timeout for LLM calls. Must exceed the time needed to emit "
            "MAX_ANSWER_TOKENS: a reasoning model producing ~16k tokens can "
            "take well over a minute. Was 60s, which risked timing out before "
            "a long answer finished."
        ),
    )
    llm_max_retries: int = Field(default=2)
    llm_reasoning_effort: str = Field(
        default="high",
        description=(
            "Reasoning effort for reasoning models (high|medium|low|none), sent "
            "as OpenRouter's `reasoning.effort`. Lower values cut latency a lot "
            "(reasoning tokens dominate generation time) but may hurt answer "
            "quality — A/B before lowering in production. 'high' sends no "
            "parameter (provider default), so the default path is unchanged."
        ),
    )
    llm_json_mode: bool = Field(
        default=True,
        description=(
            "Request OpenAI-style JSON mode (response_format=json_object) so "
            "the model returns a JSON payload in `content` instead of prose "
            "or chain-of-thought. Skipped automatically if the provider "
            "rejects the parameter."
        ),
    )

    # ---- Grounding / answers ----------------------------------------------------
    max_answer_tokens: int = Field(
        default=16384,
        description=(
            "Max completion tokens for answer generation. Must cover the "
            "model's reasoning tokens (if any) plus the JSON answer, or the "
            "response is truncated mid-JSON and cannot be parsed. Raised from "
            "8192 after observing finish_reason=length truncation in production."
        ),
    )
    confidence_threshold: float = Field(default=0.5)

    # ---- Observability (Langfuse) ------------------------------------------------
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="")
    langfuse_prefix: str = Field(default="askmydocs/")

    # ---- Evaluation ----------------------------------------------------------------
    eval_dataset_path: str = Field(default="evals/dataset/golden.jsonl")
    eval_rerank_enabled: bool = Field(default=True)

    # ---- Celery -------------------------------------------------------------------
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis broker URL for Celery task queue",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1",
        description="Redis backend URL for Celery results",
    )
    celery_task_track_started: bool = Field(
        default=True,
        description="Track task start time for progress monitoring",
    )
    celery_task_ignore_result: bool = Field(
        default=False,
        description="Store task results for retrieval",
    )
    celery_worker_prefetch_multiplier: int = Field(default=4, ge=1)
    celery_task_time_limit: int = Field(
        default=3600,
        description="Hard time limit per task (seconds)",
    )
    celery_task_soft_time_limit: int = Field(
        default=3000,
        description="Soft time limit per task (seconds)",
    )

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
