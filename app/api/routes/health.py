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


async def _check_opensearch() -> DependencyStatus:
    """Check OpenSearch cluster health."""
    try:
        from app.config import get_settings
        from app.retrieval.bm25 import BM25Indexer

        settings = get_settings()
        start = time.perf_counter()
        indexer = BM25Indexer(settings=settings)
        health = indexer._client.cluster.health()  # noqa: SLF001
        latency = (time.perf_counter() - start) * 1000
        status_str = health.get("status", "unknown")
        status = (
            ComponentStatus.HEALTHY
            if status_str in ("green", "yellow")
            else ComponentStatus.UNHEALTHY
        )
        return DependencyStatus(
            name="opensearch",
            status=status,
            latency_ms=latency,
            error=None if status == ComponentStatus.HEALTHY else f"cluster status: {status_str}",
        )
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(
            name="opensearch",
            status=ComponentStatus.UNHEALTHY,
            error=_safe_error_message(exc),
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
    pg, qdrant, os_ = await _check_postgres(), await _check_qdrant(), await _check_opensearch()

    checks = [pg, qdrant, os_]
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
        dependencies=checks,
    )
