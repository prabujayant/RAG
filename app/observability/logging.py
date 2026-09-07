"""Structured logging with secret redaction and per-request context.

The setup_logging() function is called once at process startup (e.g. from the
FastAPI lifespan) and configures the root logger with either:

- a JSON formatter for production (``log_format == "json"``), or
- a human-readable formatter for development.

Every log record automatically picks up the current request_id from
:mod:`app.observability.request_context` so logs can be correlated across
the request lifecycle without callers having to pass it explicitly.

Secret redaction
----------------
:func:`_SafeFormatter` and :class:`RedactingFilter` strip credential-like
substrings from log messages before they are emitted. The following patterns
are replaced with a placeholder:

    password, secret, api_key, token, credential, authorization,
    access_token, refresh_token, client_secret, private_key

LLM prompts, full document texts, and other large payloads are *never* logged
by this module -- callers should use :class:`app.observability.tracing.Tracer`
for those.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import UTC
from typing import Any

from app.config import get_settings
from app.config.settings import Settings
from app.observability.request_context import get_request_id

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Substrings (case-insensitive) that look like credentials. When found in a
# log message they are replaced with the placeholder ``<redacted>``.
_REDACTED_PATTERNS: tuple[str, ...] = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "token",
    "credential",
    "authorization",
    "private_key",
    "client_secret",
    "bearer",
)

# Pre-compile a single regex for efficiency.
_REDACTION_RE = re.compile(
    r"(?i)(" + "|".join(re.escape(p) for p in _REDACTED_PATTERNS) + r")\s*[:=]\s*[^\s,;}\]\"']+"
)


def _redact_message(message: str) -> str:
    """Replace credential-like substrings in *message* with ``<redacted>``.

    Two passes are applied:

    1. ``authorization=Bearer <token>`` and ``authorization=Basic <value>``
       multi-token patterns are caught first.  Running this BEFORE the key=value
       pass is critical: otherwise the key=value regex stops at the space after
       ``Bearer`` and leaves the token exposed.
    2. ``<key>=<value>`` / ``<key>: <value>`` patterns where ``<key>`` matches
       one of the known credential keywords (password, token, api_key, …).
    """
    # "Bearer <value>" and "Basic <value>" — must run FIRST so that
    # "Bearer" at the start of a value is caught before the key=value pass
    # consumes the whole line.
    out = re.sub(r"(?i)\b(Bearer|Basic)\s+\S+", r"\1 <redacted>", message)
    # <key>=<value> / <key>: <value> patterns
    out = _REDACTION_RE.sub(r"\1=<redacted>", out)
    return out


class RedactingFilter(logging.Filter):
    """Logging filter that redacts credential-like substrings from messages."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            original = record.getMessage()
            redacted = _redact_message(original)
            if redacted != original:
                # Replace the formatted message; args stay untouched.
                record.msg = redacted
                record.args = ()
        except Exception:  # noqa: BLE001
            # Never let a logging filter raise; just emit the record.
            pass
        return True


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    # Standard LogRecord attributes to include in JSON output.
    _STANDARD_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "request_id",
    }

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime

        # ISO-8601 timestamp in UTC with millisecond precision.
        ts = (
            datetime.fromtimestamp(record.created, tz=UTC)
            .strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{int(record.msecs):03d}Z"
        )
        payload: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Inject the current request id (if any).
        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id

        # Attach any extra attributes the caller stashed on the record.
        for key, value in record.__dict__.items():
            if key in self._STANDARD_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class _PrettyFormatter(logging.Formatter):
    """Human-readable formatter with request_id, used in development."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = get_request_id()
        prefix = f"[req={request_id}] " if request_id else ""
        message = record.getMessage()
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{ts} — {record.name} — {record.levelname} — {prefix}{message}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(settings: Settings | None = None) -> None:
    """Configure the root logger.

    Idempotent: if setup_logging has already configured handlers, the existing
    handlers are replaced.

    Parameters
    ----------
    settings:
        Application settings. If ``None``, the global settings singleton is
        used.
    """
    if settings is None:
        settings = get_settings()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_format = getattr(settings, "log_format", "json").lower()

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any pre-existing handlers (e.g. uvicorn's defaults).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    if log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_PrettyFormatter())
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)

    # Quiet down very chatty libraries.
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*.

    A thin wrapper around :func:`logging.getLogger` so callers don't need to
    import :mod:`logging` directly. The returned logger picks up the global
    redaction filter and request_id injection automatically.
    """
    return logging.getLogger(name)


__all__ = [
    "RedactingFilter",
    "get_logger",
    "setup_logging",
]
