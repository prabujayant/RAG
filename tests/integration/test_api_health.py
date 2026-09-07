"""Integration tests for health and readiness endpoints.

All dependency checks (PostgreSQL, Qdrant, OpenSearch) are mocked so the
tests run deterministically without requiring live Docker services.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from app.api.schemas.common import ComponentStatus, DependencyStatus
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())

def _healthy_dep(name: str) -> DependencyStatus:
    return DependencyStatus(
        name=name,
        status=ComponentStatus.HEALTHY,
        latency_ms=1.0,
    )

def _unhealthy_dep(name: str, error: str) -> DependencyStatus:
    return DependencyStatus(
        name=name,
        status=ComponentStatus.UNHEALTHY,
        error=error,
    )

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_does_not_require_db(self, client: TestClient) -> None:
        """Health check should succeed even when DB is unavailable."""
        response = client.get("/health")
        assert response.status_code == 200

class TestReadinessEndpoint:
    @patch("app.api.routes.health._check_postgres")
    @patch("app.api.routes.health._check_qdrant")
    @patch("app.api.routes.health._check_opensearch")
    def test_ready_returns_200(
        self, mock_os, mock_qdrant, mock_pg, client: TestClient
    ) -> None:
        mock_pg.return_value = _healthy_dep("postgresql")
        mock_qdrant.return_value = _healthy_dep("qdrant")
        mock_os.return_value = _healthy_dep("opensearch")
        response = client.get("/ready")
        assert response.status_code == 200

    @patch("app.api.routes.health._check_postgres")
    @patch("app.api.routes.health._check_qdrant")
    @patch("app.api.routes.health._check_opensearch")
    def test_ready_returns_readiness_status(
        self, mock_os, mock_qdrant, mock_pg, client: TestClient
    ) -> None:
        mock_pg.return_value = _healthy_dep("postgresql")
        mock_qdrant.return_value = _healthy_dep("qdrant")
        mock_os.return_value = _healthy_dep("opensearch")
        response = client.get("/ready")
        data = response.json()
        assert "status" in data
        assert "dependencies" in data
        assert "checked_at" in data

    @patch("app.api.routes.health._check_postgres")
    @patch("app.api.routes.health._check_qdrant")
    @patch("app.api.routes.health._check_opensearch")
    def test_ready_checks_all_dependencies(
        self, mock_os, mock_qdrant, mock_pg, client: TestClient
    ) -> None:
        mock_pg.return_value = _healthy_dep("postgresql")
        mock_qdrant.return_value = _healthy_dep("qdrant")
        mock_os.return_value = _healthy_dep("opensearch")
        response = client.get("/ready")
        deps = response.json()["dependencies"]
        names = {d["name"] for d in deps}
        assert names == {"postgresql", "qdrant", "opensearch"}

    @patch("app.api.routes.health._check_postgres")
    @patch("app.api.routes.health._check_qdrant")
    @patch("app.api.routes.health._check_opensearch")
    def test_ready_all_healthy(
        self, mock_os, mock_qdrant, mock_pg, client: TestClient
    ) -> None:
        mock_pg.return_value = _healthy_dep("postgresql")
        mock_qdrant.return_value = _healthy_dep("qdrant")
        mock_os.return_value = _healthy_dep("opensearch")
        response = client.get("/ready")
        assert response.json()["status"] == "healthy"

    @patch("app.api.routes.health._check_postgres")
    @patch("app.api.routes.health._check_qdrant")
    @patch("app.api.routes.health._check_opensearch")
    def test_ready_any_unhealthy_is_unhealthy(
        self, mock_os, mock_qdrant, mock_pg, client: TestClient
    ) -> None:
        mock_pg.return_value = _healthy_dep("postgresql")
        mock_qdrant.return_value = _unhealthy_dep("qdrant", "connection refused")
        mock_os.return_value = _healthy_dep("opensearch")
        response = client.get("/ready")
        assert response.json()["status"] == "unhealthy"

    @patch("app.api.routes.health._check_postgres")
    @patch("app.api.routes.health._check_qdrant")
    @patch("app.api.routes.health._check_opensearch")
    def test_ready_error_is_safe_and_non_empty(
        self, mock_os, mock_qdrant, mock_pg, client: TestClient
    ) -> None:
        mock_pg.return_value = _unhealthy_dep("postgresql", "connection refused")
        mock_qdrant.return_value = _unhealthy_dep("qdrant", "timeout")
        mock_os.return_value = _unhealthy_dep("opensearch", "cluster red")
        response = client.get("/ready")
        data = response.json()
        for dep in data["dependencies"]:
            assert dep["status"] == "unhealthy"
            assert dep["error"]  # safe message present, no secrets

class TestRequestIDPropagation:
    def test_request_id_header_returned(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"]

    def test_request_id_header_propagated(self, client: TestClient) -> None:
        custom_id = "my-custom-request-id-123"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id

    def test_request_id_in_error_response(self, client: TestClient) -> None:
        custom_id = "error-req-456"
        response = client.post(
            "/query",
            json={"question": "hi"},  # too short → validation error
            headers={"X-Request-ID": custom_id},
        )
        assert response.status_code == 422
        assert response.headers.get("X-Request-ID") == custom_id
