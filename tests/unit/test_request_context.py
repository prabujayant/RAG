"""Unit tests for the request context module."""

from __future__ import annotations

import pytest
from app.observability.request_context import (
    RequestContext,
    clear_request_id,
    get_request_id,
    set_request_id,
)


@pytest.fixture(autouse=True)
def _clear_context() -> None:
    """Ensure no request_id leaks between tests."""
    clear_request_id()
    yield
    clear_request_id()

class TestRequestContext:
    def test_generates_uuid_when_none_given(self) -> None:
        with RequestContext() as rid:
            assert rid is not None
            assert len(rid) >= 16  # UUID4
            assert get_request_id() == rid

    def test_uses_explicit_id(self) -> None:
        with RequestContext("explicit-id-1") as rid:
            assert rid == "explicit-id-1"
            assert get_request_id() == "explicit-id-1"

    def test_clears_on_exit(self) -> None:
        with RequestContext("ctx-1"):
            pass
        assert get_request_id() is None

    def test_restores_previous_value_on_exit(self) -> None:
        with RequestContext("outer"):
            with RequestContext("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"
        assert get_request_id() is None

    def test_restores_to_none_when_no_outer(self) -> None:
        clear_request_id()
        with RequestContext("first"):
            pass
        assert get_request_id() is None

class TestGetRequestId:
    def test_returns_none_outside_context(self) -> None:
        clear_request_id()
        assert get_request_id() is None

class TestSetRequestId:
    def test_returns_context_manager(self) -> None:
        ctx = set_request_id("from-set")
        assert isinstance(ctx, RequestContext)
        with ctx:
            assert get_request_id() == "from-set"

class TestClearRequestId:
    def test_clears_active_id(self) -> None:
        with RequestContext("temp"):
            clear_request_id()
        assert get_request_id() is None

    def test_clear_outside_context_is_safe(self) -> None:
        clear_request_id()
        clear_request_id()  # no error
        assert get_request_id() is None
