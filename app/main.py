"""AskMyDocs FastAPI application entry point."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, query
from app.config import get_settings
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
    yield
    logger.info("AskMyDocs shutting down")


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

    # CORS -- allow all origins for now (configure as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
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
