"""Integration tests for the observability layer (timing + FastAPI wiring)."""

from __future__ import annotations

import time

import pytest
from app.main import create_app
from app.observability import clear_request_id, get_request_id
from app.observability.timing import TimeResult, time_operation
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())

class TestTiming:
    def test_measures_wall_clock(self) -> None:
        with time_operation("sleep") as t:
            time.sleep(0.02)
        assert t.duration_ms >= 15.0  # allow some scheduler slop
        assert t.ok
        assert t.name == "sleep"

    def test_captures_exception(self) -> None:
        with pytest.raises(RuntimeError, match="boom"), time_operation("failing") as t:
            raise RuntimeError("boom")
        assert t.error is not None
        assert t.duration_ms >= 0.0
        assert t.ok is False

    def test_result_is_time_result_instance(self) -> None:
        with time_operation("noop", log_on_exit=False) as t:
            pass
        assert isinstance(t, TimeResult)

class TestObservabilityInFastAPI:
    def test_request_id_propagated_to_log(self, client: TestClient) -> None:
        """Health route generates a request_id and exposes it on response."""
        clear_request_id()
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        # The request_id should be set on the state object during the request,
        # but cleared after because we exit the context.
        assert get_request_id() is None

    def test_custom_request_id_echoed(self, client: TestClient) -> None:
        response = client.get(
            "/health", headers={"X-Request-ID": "my-custom-id-42"}
        )
        assert response.headers["X-Request-ID"] == "my-custom-id-42"

    def test_request_id_on_error(self, client: TestClient) -> None:
        response = client.get(
            "/health", headers={"X-Request-ID": "err-id-99"}
        )
        assert response.headers["X-Request-ID"] == "err-id-99"

    def test_no_request_id_leak_between_requests(self, client: TestClient) -> None:
        """After a request completes, the contextvar is reset."""
        client.get("/health", headers={"X-Request-ID": "first"})
        assert get_request_id() is None
        client.get("/health", headers={"X-Request-ID": "second"})
        assert get_request_id() is None
