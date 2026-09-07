"""Unit tests for the observability logging module."""

from __future__ import annotations

import io
import json
import logging

from app.observability.logging import (
    RedactingFilter,
    _JsonFormatter,
    _PrettyFormatter,
    _redact_message,
    get_logger,
    setup_logging,
)


class TestRedaction:
    def test_redacts_password_value(self) -> None:
        msg = "connecting with password=hunter2 to db"
        assert "password=<redacted>" in _redact_message(msg)
        assert "hunter2" not in _redact_message(msg)

    def test_redacts_api_key_value(self) -> None:
        msg = "request failed: api_key=sk-abc123def456"
        assert "api_key=<redacted>" in _redact_message(msg)
        assert "sk-abc123def456" not in _redact_message(msg)

    def test_redacts_token_value(self) -> None:
        msg = "auth token=eyJhbGciOi... stripped"
        out = _redact_message(msg)
        assert "token=<redacted>" in out
        assert "eyJhbGciOi" not in out

    def test_redacts_authorization_header(self) -> None:
        # When the credential value is a single token (no spaces), the
        # primary regex matches the whole ``key=value`` pair.
        msg = "headers: authorization=BearerXYZ"
        out = _redact_message(msg)
        assert "authorization=<redacted>" in out
        assert "BearerXYZ" not in out

    def test_redacts_bearer_with_token(self) -> None:
        # "Bearer <token>" pattern is handled by the secondary pass.
        # The first regex catches "Authorization=Bearer <token>", the second
        # pass catches "Authorization: Bearer <token>". In both cases the JWT
        # token value must be hidden.
        msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload"
        out = _redact_message(msg)
        # The credential value (JWT) must not appear in the redacted output.
        assert "eyJhbGciOiJIUzI1NiJ9" not in out

    def test_case_insensitive_redaction(self) -> None:
        msg = "API_KEY=xyz"
        out = _redact_message(msg)
        assert "xyz" not in out
        assert "<redacted>" in out

    def test_passes_through_normal_messages(self) -> None:
        msg = "user asked a question about OAuth"
        assert _redact_message(msg) == msg

    def test_redacting_filter_applies_to_log_record(self) -> None:
        flt = RedactingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="logging in with password=secret123",
            args=(),
            exc_info=None,
        )
        assert flt.filter(record) is True
        assert "secret123" not in record.msg
        assert "<redacted>" in record.msg

class TestJsonFormatter:
    def test_emits_valid_json(self) -> None:
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        out = fmt.format(record)
        data = json.loads(out)
        assert data["level"] == "INFO"
        assert data["logger"] == "app.test"
        assert data["message"] == "hello world"
        assert "timestamp" in data

    def test_includes_request_id_when_set(self) -> None:
        from app.observability.request_context import RequestContext

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="with request id",
            args=(),
            exc_info=None,
        )
        with RequestContext("req-test-123"):
            data = json.loads(fmt.format(record))
        assert data["request_id"] == "req-test-123"

    def test_extra_fields_included(self) -> None:
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="with extra",
            args=(),
            exc_info=None,
        )
        record.custom_field = "value-42"  # type: ignore[attr-defined]
        data = json.loads(fmt.format(record))
        assert data["custom_field"] == "value-42"

    def test_unserialisable_extras_are_repr(self) -> None:
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="unserialisable",
            args=(),
            exc_info=None,
        )
        record.complex = {"a", "b"}  # type: ignore[attr-defined]
        data = json.loads(fmt.format(record))
        # Set rendered as repr
        assert "{" in data["complex"] or "set(" in data["complex"]

class TestPrettyFormatter:
    def test_includes_request_id_when_set(self) -> None:
        from app.observability.request_context import RequestContext

        fmt = _PrettyFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        with RequestContext("req-pretty-1"):
            out = fmt.format(record)
        assert "[req=req-pretty-1]" in out

    def test_no_request_id_omits_bracket(self) -> None:
        fmt = _PrettyFormatter()
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        out = fmt.format(record)
        assert "[req=" not in out

class TestSetupLogging:
    def test_setup_logging_json(self) -> None:
        """When log_format=json, root logger gets a JSON formatter."""
        from app.config.settings import Settings

        settings = Settings(log_format="json", log_level="INFO", app_env="production")
        setup_logging(settings)
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(
            isinstance(h.formatter, _JsonFormatter) for h in root.handlers
        )

    def test_setup_logging_pretty(self) -> None:
        """When log_format=pretty, root logger gets a pretty formatter."""
        from app.config.settings import Settings

        settings = Settings(log_format="pretty", log_level="DEBUG", app_env="development")
        setup_logging(settings)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert any(
            isinstance(h.formatter, _PrettyFormatter) for h in root.handlers
        )

    def test_setup_logging_idempotent(self) -> None:
        """Calling setup_logging twice doesn't duplicate handlers."""
        from app.config.settings import Settings

        settings = Settings(log_format="json", log_level="INFO")
        setup_logging(settings)
        first_handlers = list(logging.getLogger().handlers)
        setup_logging(settings)
        second_handlers = list(logging.getLogger().handlers)
        assert len(first_handlers) == len(second_handlers)

    def test_setup_logging_redacts_secrets(self) -> None:
        """End-to-end: secret values never reach the log output."""
        from app.config.settings import Settings

        settings = Settings(log_format="json", log_level="INFO")
        setup_logging(settings)
        # Replace stderr stream with a buffer we can read.
        buffer = io.StringIO()
        logging.getLogger().handlers[0].stream = buffer  # type: ignore[attr-defined]
        logging.getLogger("app.test").info("connecting password=hunter2 now")
        out = buffer.getvalue()
        assert "hunter2" not in out
        assert "<redacted>" in out

class TestGetLogger:
    def test_returns_module_logger(self) -> None:
        log = get_logger("app.test.module")
        assert isinstance(log, logging.Logger)
        assert log.name == "app.test.module"
