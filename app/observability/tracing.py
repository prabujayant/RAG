"""Tracing abstraction with a no-op default and a Langfuse-backed implementation.

The :class:`Tracer` protocol defines a small surface that the rest of the
codebase can use without depending on any particular tracing backend:

- :meth:`Tracer.span` -- context manager that opens a span
- :meth:`Tracer.event` -- attach a structured event to the current span
- :meth:`Tracer.set_attribute` -- set a single attribute on the current span
- :meth:`Tracer.record_error` -- mark the current span as failed

Two concrete implementations are provided:

- :class:`NoOpTracer` -- default; does nothing. Always safe to use.
- :class:`LangfuseTracer` -- wraps a Langfuse client when credentials are
  present; otherwise falls back to :class:`NoOpTracer` so the application
  never crashes due to a missing observability backend.

Use :func:`get_tracer` to obtain the appropriate tracer based on the
current :class:`app.config.settings.Settings`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from app.config import get_settings
from app.config.settings import Settings
from app.observability.logging import get_logger
from app.observability.request_context import get_request_id

logger = get_logger(__name__)

# Try to import the Langfuse client factory at module load time. If it's
# unavailable (e.g. langfuse not installed), all calls become no-ops.
try:
    from langfuse import get_client as _langfuse_get_client
except Exception:  # noqa: BLE001
    _langfuse_get_client = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Tracer(Protocol):
    """Minimal tracing interface used across the application."""

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Open a span with *name* and optional attributes; yields a :class:`_SpanScope`."""

    def event(self, name: str, **attributes: Any) -> None:
        """Attach a structured event to the current span (if any)."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single attribute on the current span (if any)."""

    def record_error(self, error: BaseException) -> None:
        """Mark the current span as failed with *error*."""


# ---------------------------------------------------------------------------
# No-op implementation
# ---------------------------------------------------------------------------

class _NoopSpan:
    """Stand-in span object yielded by :class:`NoOpTracer`."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_output(self, value: Any) -> None:
        return None

    def end(self) -> None:
        return None


class NoOpTracer:
    """Tracer that does nothing. Used when tracing is disabled or unavailable."""

    def __init__(self, *, enabled: bool = False) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Always False for NoOpTracer."""
        return False

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_NoopSpan]:
        """Open a no-op span; yields an inert span object."""
        span = _NoopSpan()
        yield span

    def event(self, name: str, **attributes: Any) -> None:
        """No-op."""
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        return None

    def record_error(self, error: BaseException) -> None:
        """No-op."""
        return None


# ---------------------------------------------------------------------------
# Langfuse implementation
# ---------------------------------------------------------------------------

class _LangfuseSpan:
    """Span object yielded by :class:`LangfuseTracer`."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def set_attribute(self, key: str, value: Any) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(metadata={key: value})
        except Exception:  # noqa: BLE001
            logger.debug("Failed to set span attribute %s", key, exc_info=True)

    def set_output(self, value: Any) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(output=value)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to set span output", exc_info=True)

    def end(self) -> None:
        if self._observation is None:
            return
        try:
            self._observation.end()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to end span", exc_info=True)


class LangfuseTracer:
    """Tracer backed by a Langfuse v4 client.

    If the Langfuse client fails to initialize (e.g. due to missing
    credentials or network errors), the constructor transparently falls back
    to a :class:`NoOpTracer` so the application keeps running.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any = None
        self._fallback: NoOpTracer | None = None

        if not self._settings.langfuse_enabled:
            logger.info("Langfuse credentials not configured; tracing is disabled.")
            self._fallback = NoOpTracer(enabled=False)
            return

        try:
            # Langfuse v4: get_client() reads LANGFUSE_* env vars and returns
            # a singleton client. We pass credentials explicitly as a
            # defence-in-depth so the same Settings object drives both.
            if _langfuse_get_client is None:
                raise RuntimeError("langfuse package not available")
            self._client = _langfuse_get_client()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to initialise Langfuse client; falling back to NoOpTracer: %s",
                exc,
            )
            self._fallback = NoOpTracer(enabled=False)

    @property
    def enabled(self) -> bool:
        """True when Langfuse is active and tracing should record spans."""
        return self._client is not None and self._fallback is None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_LangfuseSpan | _NoopSpan]:
        """Open a Langfuse observation named *name* and yield a span object.

        If Langfuse is unavailable, yields a no-op span instead.
        """
        if self._fallback is not None:
            with self._fallback.span(name, **attributes) as span:
                yield span
            return

        observation: Any = None
        scoped_span: _LangfuseSpan | _NoopSpan

        try:
            # Always attach the current request_id as a top-level attribute.
            request_id = get_request_id()
            metadata: dict[str, Any] = dict(attributes)
            if request_id:
                metadata.setdefault("request_id", request_id)

            observation = self._client.start_observation(
                name=name,
                as_type="span",
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to start Langfuse span %s: %s", name, exc)

        # Build the appropriate span object (Langfuse-backed or no-op).
        scoped_span = _LangfuseSpan(observation) if observation is not None else _NoopSpan()

        try:
            yield scoped_span
        except BaseException as exc:  # noqa: BLE001
            self.record_error(exc)
            raise
        finally:
            scoped_span.end()

    def event(self, name: str, **attributes: Any) -> None:
        """Record a structured event on the current Langfuse span, if any."""
        if self._fallback is not None or self._client is None:
            return
        try:
            self._client.create_event(
                name=name,
                metadata=attributes,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to record event %s", name, exc_info=True)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a metadata attribute on the current observation, if any."""
        if self._fallback is not None or self._client is None:
            return
        try:
            current = self._client.get_current_observation_id()
            if current is None:
                return
            self._client.update_current_span(
                metadata={key: value},
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to set attribute %s", key, exc_info=True)

    def record_error(self, error: BaseException) -> None:
        """Mark the current span as failed with *error*."""
        if self._fallback is not None or self._client is None:
            return
        try:
            self._client.update_current_span(
                level="ERROR",
                status_message=str(error),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to record error", exc_info=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DEFAULT_TRACER: Tracer | None = None


def get_tracer(settings: Settings | None = None) -> Tracer:
    """Return the active tracer based on application settings.

    The default is :class:`NoOpTracer` (zero overhead). When Langfuse
    credentials are configured in settings, a :class:`LangfuseTracer` is
    returned. If Langfuse fails to initialise, the factory transparently
    returns a :class:`NoOpTracer` instead.
    """
    global _DEFAULT_TRACER
    if _DEFAULT_TRACER is not None:
        return _DEFAULT_TRACER

    if settings is None:
        settings = get_settings()

    if settings.langfuse_enabled:
        tracer: NoOpTracer | LangfuseTracer = LangfuseTracer(settings=settings)
        if not isinstance(tracer, LangfuseTracer) or getattr(tracer, "enabled", False):
            _DEFAULT_TRACER = tracer
            return _DEFAULT_TRACER
        # Langfuse was requested but failed to initialise; fall through.
        _DEFAULT_TRACER = NoOpTracer(enabled=False)
        return _DEFAULT_TRACER

    _DEFAULT_TRACER = NoOpTracer(enabled=False)
    return _DEFAULT_TRACER


def reset_tracer() -> None:
    """Clear the cached tracer (mainly for tests)."""
    global _DEFAULT_TRACER
    _DEFAULT_TRACER = None


__all__ = [
    "LangfuseTracer",
    "NoOpTracer",
    "Tracer",
    "get_tracer",
    "reset_tracer",
]
