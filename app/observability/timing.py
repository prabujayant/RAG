"""Timing helper: measure latency of an operation as a context manager.

Usage::

    with time_operation("retrieval") as t:
        do_retrieval()
    logger.info("retrieval done", extra={"duration_ms": t.duration_ms})
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimeResult:
    """Result of a :func:`time_operation` block.

    Fields are populated when the block exits. ``duration_ms`` is always
    set; ``error`` is set only when the wrapped block raised.
    """

    name: str
    duration_ms: float = 0.0
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        """True when no exception was raised inside the block."""
        return self.error is None


@contextmanager
def time_operation(name: str, *, log_on_exit: bool = True) -> Iterator[TimeResult]:
    """Measure wall-clock duration of the wrapped block.

    Parameters
    ----------
    name:
        Human-readable name of the operation (e.g. ``"retrieval"``).
    log_on_exit:
        When True, emits a structured log line on block exit with the
        measured duration. Errors are logged at WARNING.

    The yielded :class:`TimeResult` is mutated in-place; callers can read
    ``duration_ms`` and ``error`` from it both inside and after the block.
    """
    result = TimeResult(name=name)
    start = time.perf_counter()
    try:
        yield result
    except BaseException as exc:  # noqa: BLE001
        result.duration_ms = (time.perf_counter() - start) * 1000
        result.error = exc
        if log_on_exit:
            logger.warning(
                "operation failed: %s",
                name,
                extra={"operation": name, "duration_ms": result.duration_ms},
            )
        raise
    else:
        result.duration_ms = (time.perf_counter() - start) * 1000
        if log_on_exit:
            logger.debug(
                "operation completed: %s",
                name,
                extra={"operation": name, "duration_ms": result.duration_ms},
            )


__all__ = ["TimeResult", "time_operation"]
