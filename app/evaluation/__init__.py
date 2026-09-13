"""Evaluation pipeline for AskMyDocs.

Provides:
- Dataset loading and validation (``datasets``)
- Pure retrieval metric functions (``retrieval_metrics``)
- Citation / grounding metric computation (``citation_metrics``)
- Ragas integration with no-network fallback (``ragas_eval``)
- End-to-end runner and CLI entry point (``runner``, ``run``)
- Baseline and regression detection (``regression``)
"""

from app.evaluation.citation_metrics import (
    CitationMetricSet,
    compute_citation_metrics,
)
from app.evaluation.datasets import (
    Difficulty,
    GoldenDataset,
    GoldenExample,
    load_golden_dataset,
)
from app.evaluation.ragas_eval import (
    RagasEvaluator,
)
from app.evaluation.regression import (
    MetricThreshold,
    RegressionEntry,
    RegressionReport,
    ThresholdSet,
    check_regression,
    compare_baseline,
    load_baseline,
    load_thresholds,
    resolve_thresholds,
    save_baseline,
)
from app.evaluation.retrieval_metrics import (
    RetrievalMetricSet,
    aggregate_retrieval_metrics,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.run import main as run_cli
from app.evaluation.runner import (
    EXPERIMENTS,
    EvaluationRunner,
    ExperimentConfig,
    ExperimentResults,
    QuestionResult,
    run_evaluation,
)

__all__ = [
    # datasets
    "Difficulty",
    "GoldenExample",
    "GoldenDataset",
    "load_golden_dataset",
    # retrieval metrics
    "RetrievalMetricSet",
    "aggregate_retrieval_metrics",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    # citation metrics
    "CitationMetricSet",
    "compute_citation_metrics",
    # ragas
    "RagasEvaluator",
    # runner
    "EvaluationRunner",
    "ExperimentConfig",
    "EXPERIMENTS",
    "QuestionResult",
    "ExperimentResults",
    "run_evaluation",
    # regression
    "MetricThreshold",
    "ThresholdSet",
    "RegressionEntry",
    "RegressionReport",
    "check_regression",
    "compare_baseline",
    "load_baseline",
    "load_thresholds",
    "resolve_thresholds",
    "save_baseline",
    # CLI
    "run_cli",
]
