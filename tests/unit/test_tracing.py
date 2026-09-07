"""Unit tests for the observability tracing module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.observability.tracing import (
    LangfuseTracer,
    NoOpTracer,
    _LangfuseSpan,
    get_tracer,
    reset_tracer,
)


@pytest.fixture(autouse=True)
def _reset_tracer_cache() -> None:
    """Reset the global tracer cache between tests."""
    reset_tracer()
    yield
    reset_tracer()

class TestNoOpTracer:
    def test_enabled_is_false(self) -> None:
        tracer = NoOpTracer()
        assert tracer.enabled is False

    def test_span_yields_inert_span(self) -> None:
        tracer = NoOpTracer()
        with tracer.span("test") as span:
            span.set_attribute("key", "value")  # must not raise
            span.set_output({"result": 1})
            span.end()
        # No assertions on side effects; just verify no exception.

    def test_event_and_set_attribute_are_safe(self) -> None:
        tracer = NoOpTracer()
        tracer.event("event", foo="bar")
        tracer.set_attribute("key", "value")
        tracer.record_error(ValueError("test"))
        # No assertions on side effects; just verify no exception.

class TestLangfuseTracerWithoutCreds:
    def test_falls_back_to_noop_when_disabled(self) -> None:
        """When langfuse_enabled is False, LangfuseTracer uses a NoOp fallback."""
        settings = MagicMock()
        settings.langfuse_enabled = False
        tracer = LangfuseTracer(settings=settings)
        # Should not raise and should be a no-op
        with tracer.span("test") as span:
            span.set_attribute("a", 1)
        assert tracer.enabled is False
        tracer.event("ev")
        tracer.set_attribute("k", "v")
        tracer.record_error(RuntimeError("x"))

    def test_falls_back_to_noop_on_init_error(self) -> None:
        """If get_client() raises, LangfuseTracer uses a NoOp fallback."""
        settings = MagicMock()
        settings.langfuse_enabled = True
        with patch("app.observability.tracing._langfuse_get_client", side_effect=RuntimeError("boom")):
            tracer = LangfuseTracer(settings=settings)
        # NoOp path: enabled is False; no exception.
        assert tracer.enabled is False
        with tracer.span("test") as span:
            span.set_attribute("x", 1)

class TestLangfuseTracerWithMockedClient:
    def test_span_uses_client(self) -> None:
        """A successful start_observation call yields a real span."""
        settings = MagicMock()
        settings.langfuse_enabled = True
        mock_obs = MagicMock()
        mock_client = MagicMock()
        mock_client.start_observation.return_value = mock_obs

        with patch("app.observability.tracing._langfuse_get_client", return_value=mock_client):
            tracer = LangfuseTracer(settings=settings)
        assert tracer.enabled is True

        with tracer.span("ingestion", document_id="d1") as span:
            assert isinstance(span, _LangfuseSpan)
            span.set_attribute("chunks", 42)
            span.set_output({"ok": True})
        mock_obs.end.assert_called_once()

    def test_span_records_error_on_exception(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = True
        mock_obs = MagicMock()
        mock_client = MagicMock()
        mock_client.start_observation.return_value = mock_obs

        with patch("app.observability.tracing._langfuse_get_client", return_value=mock_client):
            tracer = LangfuseTracer(settings=settings)

        with pytest.raises(ValueError, match="boom"), tracer.span("fail"):
            raise ValueError("boom")

        mock_obs.end.assert_called_once()
        mock_client.update_current_span.assert_called()

    def test_set_attribute_safe_when_no_current_span(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = True
        mock_client = MagicMock()
        mock_client.get_current_observation_id.return_value = None
        with patch("app.observability.tracing._langfuse_get_client", return_value=mock_client):
            tracer = LangfuseTracer(settings=settings)
        # Should not raise.
        tracer.set_attribute("a", 1)
        mock_client.update_current_span.assert_not_called()

    def test_event_swallows_errors(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = True
        mock_client = MagicMock()
        mock_client.create_event.side_effect = RuntimeError("network down")
        with patch("app.observability.tracing._langfuse_get_client", return_value=mock_client):
            tracer = LangfuseTracer(settings=settings)
        # Should not raise.
        tracer.event("ev", k="v")

class TestGetTracerFactory:
    def test_returns_noop_when_no_creds(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = False
        tracer = get_tracer(settings)
        assert isinstance(tracer, NoOpTracer)
        # Cached on second call
        assert get_tracer(settings) is tracer

    def test_returns_langfuse_when_creds_and_init_ok(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = True
        mock_client = MagicMock()
        with patch("app.observability.tracing._langfuse_get_client", return_value=mock_client):
            tracer = get_tracer(settings)
        assert isinstance(tracer, LangfuseTracer)
        assert tracer.enabled is True

    def test_returns_noop_when_creds_but_init_fails(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = True
        with patch(
            "app.observability.tracing._langfuse_get_client",
            side_effect=RuntimeError("init failed"),
        ):
            tracer = get_tracer(settings)
        assert isinstance(tracer, NoOpTracer)

    def test_reset_clears_cache(self) -> None:
        settings = MagicMock()
        settings.langfuse_enabled = False
        first = get_tracer(settings)
        reset_tracer()
        second = get_tracer(settings)
        assert first is not second
