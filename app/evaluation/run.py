"""
CLI entry point for running evaluations.

Usage::

    python -m app.evaluation.run --experiment final --mock --question-limit 10
    python -m app.evaluation.run --experiment hybrid --output evals/results/myrun.json

Environment variables (from Settings):
    OPENROUTER_API_KEY   — required for Ragas evaluation
    OPENSEARCH_URL       — defaults to http://localhost:9200
    QDRANT_URL           — defaults to http://localhost:6333
    EVAL_DATASET_PATH    — defaults to evals/dataset/golden.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings

from .regression import check_regression
from .runner import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        "-e",
        choices=["vector", "bm25", "hybrid", "final"],
        default="final",
        help="Experiment to run (default: final)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLM instead of real LLM (no API key required)",
    )
    parser.add_argument(
        "--question-limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (for quick smoke tests)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results JSON (default: evals/results/)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline JSON for regression check after the run",
    )
    parser.add_argument(
        "--no-regression-check",
        action="store_true",
        help="Skip post-run regression check even when --baseline is provided",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    get_settings()

    logger.info(f"Starting evaluation experiment={args.experiment} mock={args.mock}")

    try:
        results = run_evaluation(
            experiment=args.experiment,
            mock=args.mock,
            question_limit=args.question_limit,
            output_dir=args.output,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

    logger.info(
        f"Evaluation complete — {results.question_count} questions, "
        f"{results.duration_seconds:.1f}s, "
        f"recall@10={results.aggregate_retrieval.get('recall_at_k', 0):.3f}"
    )

    # Optional regression check against a stored baseline
    if args.baseline and not args.no_regression_check:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            logger.warning(f"Baseline not found: {baseline_path} — skipping regression check")
        else:
            logger.info(f"Checking regression against baseline: {baseline_path}")
            check_regression(results, baseline_path, verbose=True)


if __name__ == "__main__":
    main()
