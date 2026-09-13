"""AskMyDocs FastAPI application entry point."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, query
from app.config import get_settings
from app.config.settings import Settings
from app.observability import (
    RequestContext,
    get_logger,
    get_tracer,
    setup_logging,
)
from app.observability.tracing import Tracer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic around the application lifetime."""
    settings = get_settings()
    setup_logging(settings)
    tracer = get_tracer(settings)
    logger.info(
        "AskMyDocs starting up -- env=%s, tracing=%s",
        settings.app_env,
        "enabled" if getattr(tracer, "enabled", False) else "disabled",
    )
    await _warmup_models(settings)
    yield
    logger.info("AskMyDocs shutting down")


async def _warmup_models(settings: Settings) -> None:
    """Pre-load the embedding model so the first query stays fast.

    BGE-M3 loads lazily on first use (~50s on CPU). Warming it here moves
    that cost to startup, where it belongs. Failures are non-fatal: the
    model will simply load on first use instead.
    """
    if not settings.warmup_models:
        logger.info("Model warm-up disabled (WARMUP_MODELS=false)")
        return
    try:
        from app.embeddings.embedder import Embedder

        start = time.perf_counter()
        # Blocking CPU work — keep it off the event loop.
        await asyncio.to_thread(Embedder(settings=settings).embed_queries, ["warmup"])
        logger.info(
            "Embedding model warmed up in %.1fs", time.perf_counter() - start
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Model warm-up failed; first query will load the model lazily: %s",
            exc,
        )

    # The cross-encoder is on the live /query path — warm it too so the
    # first reranked query doesn't pay the model-load cost.
    if not settings.enable_reranker:
        logger.info("Reranker warm-up skipped (ENABLE_RERANKER=false)")
        return
    try:
        from app.retrieval.reranker import Reranker

        start = time.perf_counter()
        loaded = await asyncio.to_thread(Reranker(settings=settings).warmup)
        logger.info(
            "Reranker warm-up finished in %.1fs (loaded=%s)",
            time.perf_counter() - start,
            loaded,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Reranker warm-up failed; first rerank will load lazily: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Request context middleware
# ---------------------------------------------------------------------------

async def _request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Establish a RequestContext for the lifetime of the request.

    Pulls ``X-Request-ID`` from the incoming request (or generates a fresh
    UUID4) and stores it in the thread-/task-local context so any log record
    or span created during the request can be correlated. The same id is
    also attached to the outgoing response header.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    with RequestContext(request_id):
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Tracing middleware (root span per request)
# ---------------------------------------------------------------------------

async def _tracing_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Open a root span around every request, when tracing is enabled."""
    tracer: Tracer = get_tracer()
    with tracer.span(
        "http.request",
        method=request.method,
        path=request.url.path,
    ):
        return await call_next(request)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

def _safe_error_dict(exc: Exception, request_id: str | None) -> dict:
    """Build a safe error dict without stack traces or credentials."""
    message = str(exc)
    # Strip credentials / secrets from error messages
    for pattern in ("password", "secret", "api_key", "token", "credential", "authorization"):
        message = message.replace(pattern, f"<{pattern}>")
    return {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": message,
            "request_id": request_id,
        }
    }


def _register_exception_handlers(app: FastAPI) -> None:
    """Register the global exception handler for unhandled errors."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_safe_error_dict(exc, request_id),
        )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AskMyDocs",
        version="0.1.0",
        description="Grounded RAG question-answering API",
        lifespan=lifespan,
    )

    # CORS — "*" locally; restrict via CORS_ORIGINS in production
    # (e.g. "https://askmydocs.vercel.app"). Comma-separated list supported.
    # NOTE: allow_credentials is auto-disabled for a wildcard origin: browsers
    # reject `Access-Control-Allow-Origin: *` combined with credentials, so
    # shipping both would silently break credentialed cross-origin requests.
    raw_origins = getattr(get_settings(), "cors_origins", "*")
    allow_origins = [o.strip() for o in str(raw_origins).split(",") if o.strip()] or ["*"]
    allow_credentials = allow_origins != ["*"]
    if not allow_credentials:
        logger.warning(
            "CORS origins are wildcarded; allow_credentials disabled "
            "(browsers reject '*' + credentials). Set CORS_ORIGINS explicitly "
            "in production to re-enable credentialed requests."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Tracing middleware (outermost: spans the whole request)
    app.middleware("http")(_tracing_middleware)

    # Request context middleware (must run inside the tracing span)
    app.middleware("http")(_request_context_middleware)

    # Exception handlers
    _register_exception_handlers(app)

    # Register routes
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    return app


# Default app instance
app = create_app()
