"""
Celery tasks for benchmark report generation.

These tasks generate the benchmark.md report from experiment results.
"""

from __future__ import annotations

import logging

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def run_full_benchmark(self) -> dict:
    """
    Run the full benchmark pipeline: evaluate all 4 configurations and generate report.

    This is the nightly Celery Beat job that runs all four experiment configurations
    (vector-only, BM25-only, hybrid, grounded) and regenerates ``evals/reports/benchmark.md``.

    Returns:
        Summary dict with keys: status, report_path, experiment_count, duration_seconds.
    """
    import time

    from scripts.generate_benchmark import generate_report

    start = time.monotonic()

    try:
        report_path = generate_report()

        duration = time.monotonic() - start
        logger.info(
            "run_full_benchmark completed",
            extra={"report_path": report_path, "duration": duration},
        )

        return {
            "status": "success",
            "report_path": report_path,
            "experiment_count": 4,
            "duration_seconds": round(duration, 2),
        }
    except Exception as exc:
        logger.exception("run_full_benchmark failed")
        raise self.retry(exc=exc) from exc


@celery_app.task(bind=True)
def generate_benchmark_report(self, experiment_dir: str | None = None) -> dict:
    """
    Generate benchmark.md from experiment result files in a directory.

    Args:
        experiment_dir: Directory containing experiment JSON files.
                       Defaults to ``evals/results/``.

    Returns:
        Summary dict with keys: status, report_path, experiment_count.
    """
    import time

    from scripts.generate_benchmark import generate_report

    start = time.monotonic()

    try:
        report_path = generate_report(results_dir=experiment_dir)

        return {
            "status": "success",
            "report_path": report_path,
            "experiment_count": None,  # generated inside the script
            "duration_seconds": round(time.monotonic() - start, 2),
        }
    except Exception as exc:
        logger.exception("generate_benchmark_report failed")
        return {"status": "error", "error": str(exc)}
