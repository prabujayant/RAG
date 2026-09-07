"""
Update the stored evaluation baseline with the latest experiment results.

This script should be run MANUALLY after a successful evaluation run
when you want to record the current implementation's performance as the
new golden baseline for regression detection.

Run:
    python scripts/update_baseline.py
    python scripts/update_baseline.py --experiment hybrid --output evals/baselines/baseline.json

Exit codes:
    0 — baseline updated successfully
    1 — no results found or error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.regression import save_baseline
from app.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "evals" / "baselines" / "baseline.json"
RESULTS_DIR = ROOT / "evals" / "results"


def _latest_result(experiment: str) -> Path | None:
    results_exp_dir = RESULTS_DIR / experiment
    if not results_exp_dir.exists():
        return None
    files = sorted(results_exp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update the stored evaluation baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        "-e",
        choices=["vector", "bm25", "hybrid", "final"],
        default="final",
        help="Which experiment result to promote to baseline",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Where to write the baseline JSON",
    )
    parser.add_argument(
        "--from-run",
        action="store_true",
        help="Run evaluation first and use those results (slow), "
        "otherwise use the latest stored results",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the baseline and print a summary without updating anything. "
        "Exits 0 if the baseline exists, 1 if it is missing.",
    )
    args = parser.parse_args()

    # --check: validate baseline exists and print a summary without updating
    if args.check:
        baseline_path = Path(args.output)
        if not baseline_path.exists():
            print(f"Baseline not found: {baseline_path}", file=sys.stderr)
            return 1
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"Baseline exists: {baseline_path}")
        print(f"  Experiment : {data.get('experiment_name', 'unknown')}")
        print(f"  Questions  : {data.get('question_count', '?')}")
        agg_ret = data.get("aggregate_retrieval", {})
        print(f"  Recall@10  : {agg_ret.get('recall_at_k', 'N/A')}")
        print(f"  Hit Rate   : {agg_ret.get('hit_rate', 'N/A')}")
        agg_cit = data.get("aggregate_citation", {})
        print(f"  Citation   : {agg_cit.get('citation_correctness', 'N/A')}")
        return 0

    # Find the results to promote
    if args.from_run:
        print(f"Running evaluation experiment={args.experiment} (mock mode) ...")
        results = run_evaluation(experiment=args.experiment, mock=True)
        output_path = Path(args.output)
        save_baseline(results, output_path)
        print(f"Baseline updated from fresh run: {output_path}")
        return 0

    result_path = _latest_result(args.experiment)
    if result_path is None:
        print(
            f"No results found for experiment={args.experiment} in {RESULTS_DIR}",
            file=sys.stderr,
        )
        print("Run 'make eval' first, or use --from-run to evaluate now.", file=sys.stderr)
        return 1

    data = json.loads(result_path.read_text(encoding="utf-8"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data["_saved_at"] = datetime.now(UTC).isoformat()  # UTC is already imported
    data["_source_file"] = str(result_path)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Baseline updated: {output_path}")
    print(f"  Source: {result_path.name}")
    print(f"  Experiment: {data.get('experiment_name')}")
    print(f"  Questions: {data.get('question_count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
