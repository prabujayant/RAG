"""Observability package: structured logging, request context, tracing, and timing.

Public API:
- :func:`setup_logging` -- configure root logger once at process startup
- :func:`get_logger` -- obtain a module-level logger with redaction applied
- :class:`RequestContext` -- thread-local request_id storage
- :func:`get_request_id` / :func:`set_request_id` / :func:`clear_request_id`
- :class:`Tracer` -- abstract tracing protocol
- :class:`NoOpTracer` -- default no-op implementation
- :class:`LangfuseTracer` -- Langfuse-backed implementation (when creds present)
- :func:`get_tracer` -- factory returning the active tracer based on settings
- :func:`time_operation` -- context manager for measuring operation latency
"""

from __future__ import annotations

from app.observability.logging import get_logger, setup_logging
from app.observability.request_context import (
    RequestContext,
    clear_request_id,
    get_request_id,
    set_request_id,
)
from app.observability.timing import TimeResult, time_operation
from app.observability.tracing import (
    LangfuseTracer,
    NoOpTracer,
    Tracer,
    get_tracer,
)

__all__ = [
    "LangfuseTracer",
    "NoOpTracer",
    "RequestContext",
    "TimeResult",
    "Tracer",
    "clear_request_id",
    "get_logger",
    "get_request_id",
    "get_tracer",
    "set_request_id",
    "setup_logging",
    "time_operation",
]
