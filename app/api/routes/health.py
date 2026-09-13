"""Health and readiness routes."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.api.schemas.common import (
    ComponentStatus,
    DependencyStatus,
    HealthResponse,
    HealthStatus,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])

APP_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return overall application health (always healthy if the process is running)."""
    return HealthResponse(status=HealthStatus.HEALTHY, version=APP_VERSION)


@router.get("/metrics")
async def metrics() -> dict:
    """Return per-stage timing attribution plus LLM token/cost totals.

    Timings answer "what is slow" with numbers: ingest.parse/clean/chunk/
    postgres/embed/qdrant/keyword and query.retrieval/rerank/evidence/
    generation/grounding. ``llm_usage`` aggregates prompt/completion tokens
    and estimated USD cost per caller label (generation/grounding/agent).
    Process-local, resets on restart.
    """
    from app.observability.cost import usage_summary
    from app.observability.metrics import summary

    data = summary()
    data["llm_usage"] = usage_summary()
    return data


# ---------------------------------------------------------------------------
# GET /ready
# ---------------------------------------------------------------------------


async def _check_postgres() -> DependencyStatus:
    """Ping PostgreSQL with a simple SELECT 1."""
    try:
        from app.db.session import SessionLocal

        session = SessionLocal()
        try:
            start = time.perf_counter()
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
            latency = (time.perf_counter() - start) * 1000
            return DependencyStatus(
                name="postgresql",
                status=ComponentStatus.HEALTHY,
                latency_ms=latency,
            )
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="postgresql",
            status=ComponentStatus.UNHEALTHY,
            error=_safe_error_message(exc),
        )


async def _check_qdrant() -> DependencyStatus:
    """Check Qdrant is reachable via get_collections."""
    try:
        from app.config import get_settings
        from app.retrieval.vector import VectorStore

        settings = get_settings()
        start = time.perf_counter()
        store = VectorStore(settings=settings)
        store._client.get_collections()  # noqa: SLF001
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="qdrant",
            status=ComponentStatus.HEALTHY,
            latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="qdrant",
            status=ComponentStatus.UNHEALTHY,
            error=_safe_error_message(exc),
        )


async def _check_keyword_search() -> DependencyStatus:
    """Check the Postgres keyword postings (tsvector) are queryable."""
    try:
        from app.retrieval.bm25 import BM25Indexer

        start = time.perf_counter()
        indexer = BM25Indexer()
        indexer.ensure_index()
        n = indexer.count()
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="keyword",
            status=ComponentStatus.HEALTHY,
            latency_ms=latency,
            error=f"postings={n}" if n == 0 else None,
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="keyword",
            status=ComponentStatus.UNHEALTHY,
            error=_safe_error_message(exc),
        )


async def _check_redis() -> DependencyStatus:
    """Ping Redis (Celery broker). Degraded, not fatal: uploads fall back to sync."""
    try:
        import socket
        from urllib.parse import urlparse

        from app.config import get_settings

        broker = get_settings().celery_broker_url
        parts = urlparse(broker)
        host, port = parts.hostname or "localhost", parts.port or 6379
        start = time.perf_counter()
        with socket.create_connection((host, port), timeout=2):
            pass
        return DependencyStatus(
            name="redis",
            status=ComponentStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="redis",
            status=ComponentStatus.UNHEALTHY,
            error=f"{_safe_error_message(exc)} (background ingest falls back to sync)",
        )


async def _check_celery() -> DependencyStatus:
    """Ping Celery workers via the Redis broker (short timeout).

    Informational like Redis: uploads fall back to sync when no worker runs,
    so this never gates overall readiness — it tells ops whether background
    ingest is actually async.
    """
    try:
        import time as _time

        from app.celery_app import celery_app

        start = _time.perf_counter()
        replies = celery_app.control.inspect(timeout=2).ping() or {}
        latency = (_time.perf_counter() - start) * 1000
        workers = sorted(replies.keys())
        if not workers:
            return DependencyStatus(
                name="celery",
                status=ComponentStatus.UNHEALTHY,
                error="no workers replied (background ingest falls back to sync)",
            )
        return DependencyStatus(
            name="celery",
            status=ComponentStatus.HEALTHY,
            latency_ms=latency,
            error=None if len(workers) == 1 else f"workers: {', '.join(workers)}",
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="celery",
            status=ComponentStatus.UNHEALTHY,
            error=f"{_safe_error_message(exc)} (background ingest falls back to sync)",
        )


def _safe_error_message(exc: Exception) -> str:
    """Strip credentials and stack traces from exception messages."""
    msg = str(exc)
    # Remove common credential patterns
    for pattern in ("password", "secret", "api_key", "token", "credential"):
        msg = msg.replace(pattern, f"<{pattern}>")
    return msg


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    """Check all dependencies and return detailed readiness status.

    Returns 200 even when some dependencies are unhealthy so that load
    balancers can still poll this endpoint. The response body distinguishes
    healthy from unhealthy components.
    """
    pg, qdrant, keyword, redis, celery = (
        await _check_postgres(),
        await _check_qdrant(),
        await _check_keyword_search(),
        await _check_redis(),
        await _check_celery(),
    )

    # Redis + Celery gate only background ingest (sync fallback exists), so
    # they are reported but excluded from the overall readiness gate.
    checks = [pg, qdrant, keyword]
    all_deps = [*checks, redis, celery]
    all_healthy = all(c.status == ComponentStatus.HEALTHY for c in checks)
    any_unhealthy = any(c.status == ComponentStatus.UNHEALTHY for c in checks)

    if all_healthy:
        overall = HealthStatus.HEALTHY
    elif any_unhealthy:
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED

    return ReadinessResponse(
        status=overall,
        dependencies=all_deps,
    )
