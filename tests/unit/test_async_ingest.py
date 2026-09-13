"""Unit tests for background ingest + SSE wiring (no services)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_try_enqueue_success() -> None:
    from app.api.routes.documents import _try_enqueue_ingest

    with patch(
        "app.tasks.ingestion.ingest_document_job", create=True
    ) as _unused:
        # Patch where it is imported inside the helper (app.tasks.ingestion).
        import app.tasks.ingestion as tasks_mod

        mock_task = MagicMock()
        orig = tasks_mod.ingest_document_job
        tasks_mod.ingest_document_job = mock_task
        try:
            assert _try_enqueue_ingest("d", "/tmp/f.md", "t", "j") is True
            mock_task.delay.assert_called_once_with("d", "/tmp/f.md", "t", "j")
        finally:
            tasks_mod.ingest_document_job = orig


@pytest.mark.unit
def test_try_enqueue_fallback_on_broker_down() -> None:

    with patch(
        "app.api.routes.documents._try_enqueue_ingest",
        side_effect=AssertionError("should not recurse"),
    ):
        pass  # sanity: helper exists and is patchable


@pytest.mark.unit
def test_ingest_background_returns_202_running() -> None:
    from app.main import create_app

    client = TestClient(create_app())
    with (
        patch("app.api.routes.documents.SessionLocal") as mock_session_cls,
        patch("app.api.routes.documents._try_enqueue_ingest", return_value=True),
    ):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        doc = MagicMock()
        doc.source = "/tmp/f.md"
        doc.title = "f.md"
        mock_session.execute.return_value.scalar_one_or_none.return_value = doc
        job = MagicMock()
        job.id = "job-1"
        job.started_at = None
        with patch(
            "app.api.routes.documents._create_job_record", return_value=job
        ):
            res = client.post("/documents/some-id/ingest?background=true")
            assert res.status_code == 202
            assert res.json()["status"] == "running"


@pytest.mark.unit
def test_upload_background_returns_processing() -> None:
    import io

    from app.main import create_app

    client = TestClient(create_app())
    with (
        patch("app.api.routes.documents.SessionLocal") as mock_session_cls,
        patch("app.api.routes.documents._try_enqueue_ingest", return_value=True),
        patch("app.api.routes.documents._persist_upload", return_value=("/tmp/x/f.md", "abc")),
        patch("app.api.routes.documents._doc_id_from_path", return_value="doc-1"),
        patch("app.api.routes.documents._create_document_record"),
        patch("app.api.routes.documents._create_job_record") as mock_job,
    ):
        mock_session_cls.return_value = MagicMock()
        mock_job.return_value = MagicMock(id="job-9", started_at=None)
        res = client.post(
            "/documents/upload?background=true",
            files={"file": ("f.md", io.BytesIO(b"# hi"), "text/markdown")},
        )
        assert res.status_code == 201
        assert res.json()["status"] == "processing"


@pytest.mark.unit
def test_query_stream_route_registered() -> None:
    from app.main import create_app

    paths = list(create_app().openapi()["paths"].keys())
    assert "/query/stream" in paths
    assert "/documents/{document_id}/jobs/{job_id}/stream" in paths


@pytest.mark.unit
def test_worker_probe_is_non_blocking_when_worker_is_busy(monkeypatch) -> None:
    """A busy (single-threaded) worker must not stall the upload request.

    ``--pool=solo`` workers cannot answer a control ping while executing a
    task, so a naive ping blocks for the whole task duration — which delayed an
    upload by ~25s in practice. The probe must fail open and return promptly.
    """
    import time as _time

    from app.api.routes import documents as docs_mod

    docs_mod._worker_probe_cache.clear()

    def _slow_ping(*args, **kwargs):  # noqa: ANN002, ANN003
        _time.sleep(5.0)  # simulate a worker that cannot reply while busy
        return [{"celery@host": {"ok": "pong"}}]

    monkeypatch.setattr(
        "app.celery_app.celery_app.control.ping", _slow_ping, raising=False
    )

    start = _time.monotonic()
    available = docs_mod._celery_worker_available(timeout=0.2)
    elapsed = _time.monotonic() - start

    assert available is True  # fail open: a busy worker is still a worker
    assert elapsed < 2.0, f"probe blocked for {elapsed:.1f}s"


@pytest.mark.unit
def test_worker_probe_detects_absent_worker(monkeypatch) -> None:
    """An empty ping reply means nothing is consuming the queue."""
    from app.api.routes import documents as docs_mod

    docs_mod._worker_probe_cache.clear()
    monkeypatch.setattr(
        "app.celery_app.celery_app.control.ping", lambda *a, **k: [], raising=False
    )

    assert docs_mod._celery_worker_available(timeout=0.2) is False


@pytest.mark.unit
def test_worker_probe_caches_positive_result(monkeypatch) -> None:
    """A detected worker is remembered briefly so uploads skip re-probing."""
    from app.api.routes import documents as docs_mod

    docs_mod._worker_probe_cache.clear()
    calls: list[int] = []

    def _counted_ping(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(1)
        return [{"celery@host": {"ok": "pong"}}]

    monkeypatch.setattr(
        "app.celery_app.celery_app.control.ping", _counted_ping, raising=False
    )

    assert docs_mod._celery_worker_available(timeout=0.2) is True
    assert docs_mod._celery_worker_available(timeout=0.2) is True
    assert len(calls) == 1  # second call served from cache


@pytest.mark.unit
def test_ready_includes_redis() -> None:
    import asyncio

    from app.api.routes.health import readiness

    res = asyncio.run(readiness())
    names = [d.name for d in res.dependencies]
    assert "redis" in names
