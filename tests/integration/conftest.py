"""Integration test configuration.

Patches sentence_transformers before it can be imported to avoid a pyarrow
access violation on Windows.  The patch is applied at session scope so it is
in place before any test module is collected.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: Integration tests (multi-component)")


@pytest.fixture(scope="session", autouse=True)
def _disable_query_persistence_in_api_tests():
    """Keep API tests isolated from an unavailable PostgreSQL service.

    Query endpoint tests mock retrieval and generation, so persisting the
    resulting query record would introduce an unintended live database
    dependency and can block while the database connection times out.
    Database persistence is covered by the database/integration tests.
    """
    from unittest.mock import patch

    with patch("app.api.routes.query._persist_query_record"):
        yield


@pytest.fixture(scope="session", autouse=True)
def _patch_sentence_transformers() -> None:
    """Replace sentence_transformers with a no-op mock before import."""
    mock_module = MagicMock()
    mock_module.CrossEncoder = MagicMock()
    sys.modules["sentence_transformers"] = mock_module
    # Also mock common sub-modules that may be imported during the chain
    for sub in (
        "sentence_transformers.CrossEncoder",
        "sentence_transformers.models",
        "sentence_transformers.util",
        "sentence_transformers.similarity",
        "sentence_transformers.retrieval",
    ):
        sys.modules[sub] = mock_module
