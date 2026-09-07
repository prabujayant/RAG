"""
Generate evals/reports/benchmark.md — a human-readable comparison of all experiments.

Run:
    python scripts/generate_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"
REPORT_PATH = ROOT / "evals" / "reports" / "benchmark.md"

EXPERIMENTS = ["vector", "bm25", "hybrid", "final"]


def _load_results(experiment: str) -> dict | None:
    """Load the latest results JSON for the given experiment name.

    Results are saved to evals/results/experiment_{experiment}_{id}.json
    (e.g. experiment_final_a1b2c3d4.json), so we glob for that pattern
    directly rather than looking in a subdirectory.
    """
    if not RESULTS_DIR.exists():
        return None
    # Find all JSON files matching the experiment pattern and pick the latest
    pattern = f"experiment_{experiment}_*.json"
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _section(title: str) -> str:
    return f"\n## {title}\n"


def _table_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _table_header(cols: list[str]) -> str:
    return "| " + " | ".join(cols) + " |\n" + "| " + " | ".join("---" for _ in cols) + " |"


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# RAG Evaluation Benchmark",
        "",
        _section("Experiment Summary"),
        _table_header([
            "Experiment", "Questions", "Duration (s)",
            "Recall@10", "Hit Rate@5", "Citation Correctness",
        ]),
        "",
    ]

    experiments_data: dict[str, dict] = {}
    for exp in EXPERIMENTS:
        data = _load_results(exp)
        if data is None:
            lines.append(
                _table_row([exp, "—", "—", "—", "—", "—", "No results found"])
            )
            continue
        experiments_data[exp] = data
        agg = data.get("aggregate_retrieval", {})
        cit = data.get("aggregate_citation", {})
        lines.append(
            _table_row([
                exp,
                str(data.get("question_count", "?")),
                f"{data.get('duration_seconds', 0):.1f}",
                _fmt(agg.get("recall_at_k")),
                _fmt(agg.get("hit_rate")),
                _fmt(cit.get("citation_correctness")),
            ])
        )

    lines.append("")

    # Per-difficulty breakdown
    if experiments_data:
        lines.append(_section("Per-Difficulty Breakdown"))
        for exp, data in experiments_data.items():
            per_diff = data.get("per_difficulty", {})
            if not per_diff:
                continue
            lines.append(f"\n### {exp}\n")
            lines.append(_table_header([
                "Difficulty", "Count", "Recall@10",
                "Hit Rate", "Citation Correctness",
            ]))
            for diff_name in ["easy", "medium", "hard", "unanswerable"]:
                dd = per_diff.get(diff_name, {})
                if not dd:
                    continue
                lines.append(
                    _table_row([
                        diff_name,
                        str(dd.get("count", "?")),
                        _fmt(dd.get("recall_at_k")),
                        _fmt(dd.get("hit_rate")),
                        _fmt(dd.get("citation_correctness")),
                    ])
                )
            lines.append("")

    # Ragas metrics
    ragas_exp = experiments_data.get("final", {})
    ragas_metrics = ragas_exp.get("aggregate_ragas", {})
    if ragas_metrics:
        lines.append(_section("Ragas Metrics (Experiment D — Full Pipeline)"))
        lines.append(_table_header(["Metric", "Value"]))
        for key, val in sorted(ragas_metrics.items()):
            lines.append(_table_row([key, _fmt(val)]))

    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Benchmark written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
