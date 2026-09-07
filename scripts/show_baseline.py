"""
Display the current stored evaluation baseline.

Run:
    python scripts/show_baseline.py
    python scripts/show_baseline.py --path evals/baselines/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "evals" / "baselines" / "baseline.json"


def _fmt(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _print_section(title: str) -> None:
    print(f"\n## {title}")
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the current evaluation baseline.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_BASELINE),
        help="Path to baseline JSON file",
    )
    args = parser.parse_args()

    baseline_path = Path(args.path)
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}", file=sys.stderr)
        return 1

    data = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(f"Baseline: {baseline_path}")
    print(f"Saved at: {data.get('_saved_at', 'unknown')}")
    print(f"Experiment: {data.get('experiment_name', 'unknown')}")
    print(f"Questions: {data.get('question_count', '?')}")

    _print_section("Retrieval Metrics")
    for key, val in sorted(data.get("aggregate_retrieval", {}).items()):
        print(f"  {key:30s} {_fmt(val)}")

    _print_section("Citation / Grounding Metrics")
    for key, val in sorted(data.get("aggregate_citation", {}).items()):
        print(f"  {key:30s} {_fmt(val)}")

    ragas = data.get("aggregate_ragas", {})
    if ragas:
        _print_section("Ragas Metrics")
        for key, val in sorted(ragas.items()):
            print(f"  {key:30s} {_fmt(val)}")

    per_diff = data.get("per_difficulty", {})
    if per_diff:
        _print_section("Per-Difficulty")
        for diff_name in ["easy", "medium", "hard", "unanswerable"]:
            dd = per_diff.get(diff_name)
            if not dd:
                continue
            print(f"\n  [{diff_name}]")
            for key, val in sorted(dd.items()):
                if key == "count":
                    continue
                print(f"    {key:30s} {_fmt(val)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
