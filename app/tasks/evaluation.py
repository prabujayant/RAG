"""
Celery tasks for asynchronous evaluation runs.

These tasks run RAG evaluation without blocking API requests.
"""

from __future__ import annotations

import logging

from app.celery_app import celery_app
from app.config import get_settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2)
def run_evaluation(
    self,
    dataset_path: str | None = None,
    experiment_name: str | None = None,
    mock: bool = False,
) -> dict:
    """
    Run a full evaluation on the golden dataset.

    Args:
        dataset_path: Optional path to golden dataset JSONL. Defaults to settings.
        experiment_name: Name for this experiment run (used in results filename).
        mock: If True, use MockLLM for deterministic evaluation.

    Returns:
        Summary dict with keys: experiment_name, questions_passed, total, duration_seconds.
    """
    import time

    from app.evaluation.runner import EvaluationRunner

    start = time.monotonic()

    settings = get_settings()
    dataset = dataset_path or settings.eval_dataset_path
    name = experiment_name or f"experiment_{self.request.id}"

    try:
        runner = EvaluationRunner(mock=mock)
        result = runner.run(dataset_path=dataset)

        duration = time.monotonic() - start
        summary = result.get("summary", {})
        logger.info(
            "run_evaluation completed",
            extra={
                "experiment": name,
                "passed": summary.get("questions_passed", 0),
                "total": summary.get("total_questions", 0),
                "duration": duration,
            },
        )

        return {
            "status": "success",
            "experiment_name": name,
            "questions_passed": summary.get("questions_passed", 0),
            "total_questions": summary.get("total_questions", 0),
            "retrieval_precision": summary.get("retrieval_precision"),
            "retrieval_recall": summary.get("retrieval_recall"),
            "citation_precision": summary.get("citation_precision"),
            "citation_recall": summary.get("citation_recall"),
            "grounding_accuracy": summary.get("grounding_accuracy"),
            "duration_seconds": round(duration, 2),
            "results_path": result.get("results_path"),
        }
    except Exception as exc:
        logger.exception("run_evaluation failed")
        raise self.retry(exc=exc) from exc


@celery_app.task(bind=True)
def check_regression(self, experiment_path: str, baseline_path: str | None = None) -> dict:
    """
    Check experiment results against baseline for regressions.

    Args:
        experiment_path: Path to the experiment results JSON file.
        baseline_path: Optional baseline JSON path. Defaults to
            ``evals/baselines/baseline.json``.

    Returns:
        Summary dict with keys: status, has_regressions, regressions.
    """
    import json

    from app.evaluation.regression import compare_baseline, load_baseline

    try:
        with open(experiment_path) as f:
            experiment_data = json.load(f)

        resolved_baseline = baseline_path or "evals/baselines/baseline.json"
        baseline = load_baseline(resolved_baseline)

        # Thresholds come from evals/thresholds.yaml (single source of truth).
        report = compare_baseline(experiment_data, baseline)

        return {
            "status": "success",
            "has_regressions": report.has_regression(),
            "regressions": [
                {
                    "metric": entry.metric,
                    "current": entry.current,
                    "baseline": entry.baseline,
                    "absolute_gap": entry.absolute_gap,
                    "relative_gap": entry.relative_gap,
                }
                for entry in report.entries
            ],
        }
    except Exception as exc:
        logger.exception("check_regression failed")
        return {"status": "error", "error": str(exc)}
