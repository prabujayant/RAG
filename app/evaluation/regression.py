"""
Baseline comparison and regression detection for evaluation results.

Provides two public functions:

* compare_baseline(results, baseline_path) — compare current results vs a stored
  baseline and return a detailed diff.
* check_regression(results, baseline_path, thresholds) — returns True if any
  metric has regressed beyond the configured tolerance, suitable for CI gates.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .runner import ExperimentResults

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class MetricThreshold:
    """Absolute minimum and relative tolerance (fraction) for a single metric."""

    metric: str
    minimum: float
    tolerance: float = 0.05  # 5 % relative tolerance by default


@dataclass
class RegressionEntry:
    """A single metric that has regressed."""

    metric: str
    baseline: float
    current: float
    absolute_gap: float
    relative_gap: float
    regressed: bool


@dataclass
class RegressionReport:
    """Full regression analysis for an experiment."""

    experiment_name: str
    timestamp: str
    total_metrics: int
    regressed_metrics: int
    entries: list[RegressionEntry] = field(default_factory=list)
    difficulty_regressions: dict[str, list[RegressionEntry]] = field(default_factory=dict)

    def has_regression(self) -> bool:
        return self.regressed_metrics > 0

    def summary(self) -> str:
        status = "❌ REGRESSION" if self.has_regression() else "✅ PASS"
        lines = [
            f"{status} — {self.experiment_name} @ {self.timestamp}",
            f"  Metrics checked : {self.total_metrics}",
            f"  Regressed       : {self.regressed_metrics}",
        ]
        if self.entries:
            lines.append("  Regressed metrics:")
            for e in self.entries:
                lines.append(
                    f"    {e.metric}: {e.baseline:.4f} → {e.current:.4f} "
                    f"(gap={e.absolute_gap:+.4f}, {e.relative_gap:+.1%})"
                )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Default thresholds (applied when None is passed to check_regression)
# ----------------------------------------------------------------------

DEFAULT_THRESHOLDS: list[MetricThreshold] = [
    MetricThreshold("recall_at_k", minimum=0.30, tolerance=0.05),
    MetricThreshold("precision_at_k", minimum=0.15, tolerance=0.05),
    MetricThreshold("mrr", minimum=0.20, tolerance=0.05),
    MetricThreshold("ndcg_at_k", minimum=0.20, tolerance=0.05),
    MetricThreshold("hit_rate", minimum=0.30, tolerance=0.05),
    MetricThreshold("citation_correctness", minimum=0.50, tolerance=0.10),
    MetricThreshold("citation_completeness", minimum=0.40, tolerance=0.10),
    MetricThreshold("grounded_answer_rate", minimum=0.50, tolerance=0.10),
    MetricThreshold("refusal_correctness", minimum=0.70, tolerance=0.10),
    MetricThreshold("unsupported_claim_rate", minimum=0.00, tolerance=0.00),
]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def load_baseline(baseline_path: str | Path) -> dict[str, Any]:
    """Load a baseline JSON file and return it as a dict."""
    p = Path(baseline_path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(results: ExperimentResults, baseline_path: str | Path) -> None:
    """
    Persist an ExperimentResults object as the new baseline.

    Call this explicitly (e.g. ``make baseline-update``) to record the
    current implementation as the new golden baseline.
    """
    p = Path(baseline_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = results.to_dict()
    d["_saved_at"] = datetime.now(UTC).isoformat()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    logger.info(f"Baseline saved to {p}")


def compare_baseline(
    current: ExperimentResults | dict[str, Any],
    baseline: ExperimentResults | dict[str, Any] | Path | str,
    thresholds: list[MetricThreshold] | None = None,
) -> RegressionReport:
    """
    Compare current results against a stored baseline.

    Parameters
    ----------
    current : ExperimentResults or dict
        Results from the current experiment run.
    baseline : ExperimentResults, dict, Path, or str
        Previously stored baseline (loaded from path if str/Path).
    thresholds : list[MetricThreshold] | None
        Metrics + minimum values + tolerances to check.
        Defaults to DEFAULT_THRESHOLDS if None.

    Returns
    -------
    RegressionReport with per-metric regression analysis.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Resolve baseline to dict
    if isinstance(baseline, (str, Path)):
        baseline = load_baseline(baseline)
    if hasattr(baseline, "to_dict"):
        baseline = baseline.to_dict()

    # Resolve current to dict
    if hasattr(current, "to_dict"):
        current = current.to_dict()

    # Collect metric namespaces to check
    retrieval_a = current.get("aggregate_retrieval", {})
    citation_a = current.get("aggregate_citation", {})
    ragas_a = current.get("aggregate_ragas", {})
    baseline_retrieval = baseline.get("aggregate_retrieval", {})
    baseline_citation = baseline.get("aggregate_citation", {})
    baseline_ragas = baseline.get("aggregate_ragas", {})

    all_pairs: list[tuple[str, dict[str, float], dict[str, float]]] = [
        ("retrieval", retrieval_a, baseline_retrieval),
        ("citation", citation_a, baseline_citation),
        ("ragas", ragas_a, baseline_ragas),
    ]

    entries: list[RegressionEntry] = []
    difficulty_regressions: dict[str, list[RegressionEntry]] = {}
    checked = 0

    for ns, curr_metrics, base_metrics in all_pairs:
        for threshold in thresholds:
            key = threshold.metric
            # Check in current namespace first, then fallback to cross-namespace key
            curr_val = curr_metrics.get(key, base_metrics.get(key))
            base_val = base_metrics.get(key)
            if curr_val is None or base_val is None:
                continue

            checked += 1
            abs_gap = curr_val - base_val
            rel_gap = abs_gap / base_val if base_val != 0 else 0.0

            # Regressed if: below absolute minimum OR (relative drop beyond tolerance AND not an improvement)
            regressed = curr_val < threshold.minimum or (
                (abs_gap < 0) and (abs(rel_gap) > threshold.tolerance)
            )

            entry = RegressionEntry(
                metric=f"{ns}.{key}" if ns != "retrieval" else key,
                baseline=base_val,
                current=curr_val,
                absolute_gap=abs_gap,
                relative_gap=rel_gap,
                regressed=regressed,
            )
            if regressed:
                entries.append(entry)

    # Per-difficulty check
    per_diff = current.get("per_difficulty", {})
    base_diff = baseline.get("per_difficulty", {})
    for diff_name, diff_metrics in per_diff.items():
        b_diff = base_diff.get(diff_name, {})
        for key in ["recall_at_k", "hit_rate", "citation_correctness"]:
            curr_val = diff_metrics.get(key)
            base_val = b_diff.get(key)
            if curr_val is None or base_val is None:
                continue
            abs_gap = curr_val - base_val
            rel_gap = abs_gap / base_val if base_val != 0 else 0.0
            diff_threshold = next((t for t in thresholds if t.metric == key), None)
            tol = diff_threshold.tolerance if diff_threshold is not None else 0.05
            regressed = (abs_gap < 0) and (abs(rel_gap) > tol)
            if regressed:
                entry = RegressionEntry(
                    metric=f"{diff_name}.{key}",
                    baseline=base_val,
                    current=curr_val,
                    absolute_gap=abs_gap,
                    relative_gap=rel_gap,
                    regressed=True,
                )
                difficulty_regressions.setdefault(diff_name, []).append(entry)

    return RegressionReport(
        experiment_name=current.get("experiment_name", "unknown"),
        timestamp=datetime.now(UTC).isoformat(),
        total_metrics=checked,
        regressed_metrics=len(entries),
        entries=entries,
        difficulty_regressions=difficulty_regressions,
    )


def check_regression(
    current: ExperimentResults | dict[str, Any],
    baseline_path: str | Path,
    thresholds: list[MetricThreshold] | None = None,
    verbose: bool = True,
) -> bool:
    """
    Gate function for CI: returns True if any metric has regressed.

    Exits with code 1 (sys.exit(1)) when regressions are detected so that
    CI pipelines fail appropriately. Set ``verbose=False`` to suppress
    the printed report.
    """
    baseline = load_baseline(baseline_path)
    report = compare_baseline(current, baseline, thresholds=thresholds)

    if verbose:
        print(report.summary())

    if report.has_regression():
        if verbose:
            logger.error(f"Regression detected in {report.regressed_metrics} metric(s)")
        sys.exit(1)
        return True  # unreachable after sys.exit

    return False
