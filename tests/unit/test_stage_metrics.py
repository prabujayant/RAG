"""Unit tests for stage-timing metrics registry."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_record_and_summary() -> None:
    from app.observability import metrics

    metrics.reset()
    metrics.record("ingest.embed", 100.0)
    metrics.record("ingest.embed", 200.0)
    summary = metrics.summary()
    assert summary["stages"]["ingest.embed"]["count"] == 2
    assert summary["stages"]["ingest.embed"]["avg_ms"] == 150.0
    assert summary["stages"]["ingest.embed"]["max_ms"] == 200.0
    assert summary["uptime_seconds"] >= 0
    metrics.reset()


@pytest.mark.unit
def test_metrics_endpoint_registered() -> None:
    from app.main import create_app

    assert "/metrics" in list(create_app().openapi()["paths"].keys())


@pytest.mark.unit
def test_ingest_result_carries_stage_ms() -> None:
    from app.ingestion.pipeline import IngestResult

    result = IngestResult(
        document_id="d", title="t", chunk_count=1, job_id="j",
        stage_ms={"parse": 1.0},
    )
    assert result.stage_ms == {"parse": 1.0}
