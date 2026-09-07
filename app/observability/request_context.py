"""Thread-local request context for the current request_id.

This module stores the current request's identifier in a
:class:`contextvars.ContextVar` so any code running inside the request --
including deeply nested service code, background tasks, and observability
helpers -- can attach the request_id to log records and traces without the
caller having to thread it through every function signature.

The :class:`RequestContext` context manager sets the value on entry and
restores the previous value on exit, so nested contexts (e.g. a span
triggering an internal sub-request) work as expected.
"""

from __future__ import annotations

import contextvars
import uuid

# A ContextVar is the right tool: it's safe across threads and async tasks
# and is automatically scoped per asyncio task.
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestContext:
    """Context manager that sets the current request_id.

    Example::

        with RequestContext() as request_id:
            logger.info("processing")  # log record is auto-tagged

        # Or with an explicit id:
        with RequestContext("req-abc123"):
            ...
    """

    def __init__(self, request_id: str | None = None) -> None:
        self._explicit = request_id
        self._generated: str | None = None
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> str:
        rid = self._explicit or str(uuid.uuid4())
        if self._explicit is None:
            self._generated = rid
        self._token = _request_id_var.set(rid)
        return rid

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _request_id_var.reset(self._token)
            self._token = None


def get_request_id() -> str | None:
    """Return the current request_id, or None if no context is active."""
    return _request_id_var.get()


def set_request_id(request_id: str) -> RequestContext:
    """Return a context manager that sets the current request_id for its block.

    Convenience wrapper for use with ``with``::

        with set_request_id("req-123"):
            do_work()
    """
    return RequestContext(request_id)


def clear_request_id() -> None:
    """Reset the current request_id to None.

    Mainly useful in tests.
    """
    _request_id_var.set(None)


__all__ = [
    "RequestContext",
    "clear_request_id",
    "get_request_id",
    "set_request_id",
]
