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

try:  # PyYAML is a declared dependency; degrade gracefully rather than crash on import.
    import yaml
except ImportError:  # pragma: no cover - exercised only in stripped-down installs
    yaml = None

from .runner import ExperimentResults

logger = logging.getLogger(__name__)

# ``evals/thresholds.yaml`` relative to the repository root (app/evaluation/regression.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "evals" / "thresholds.yaml"

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
class ThresholdSet:
    """Aggregate thresholds plus the separate per-difficulty thresholds.

    The two are kept distinct because the same metric name (e.g. ``recall_at_k``)
    legitimately carries a looser tolerance in the per-difficulty split.
    """

    aggregate: list[MetricThreshold] = field(default_factory=list)
    per_difficulty: list[MetricThreshold] = field(default_factory=list)

    def find(self, namespace: str, metric: str) -> MetricThreshold | None:
        """Return the threshold for ``metric`` in ``aggregate`` or ``per_difficulty``."""
        pool = self.aggregate if namespace == "aggregate" else self.per_difficulty
        return next((t for t in pool if t.metric == metric), None)


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
        # ASCII only: this is printed to CI logs and Windows consoles, where
        # non-cp1252 glyphs (✅/❌/→) raise UnicodeEncodeError.
        status = "[FAIL] REGRESSION" if self.has_regression() else "[OK] PASS"
        lines = [
            f"{status} — {self.experiment_name} @ {self.timestamp}",
            f"  Metrics checked : {self.total_metrics}",
            f"  Regressed       : {self.regressed_metrics}",
        ]
        if self.entries:
            lines.append("  Regressed metrics:")
            for e in self.entries:
                lines.append(
                    f"    {e.metric}: {e.baseline:.4f} -> {e.current:.4f} "
                    f"(gap={e.absolute_gap:+.4f}, {e.relative_gap:+.1%})"
                )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Threshold loading — evals/thresholds.yaml is the single source of truth
# ----------------------------------------------------------------------

# Fallback only. Used when thresholds.yaml cannot be read (e.g. the package is
# installed without the repo's evals/ directory). tests/unit/test_regression.py
# asserts these stay in sync with the YAML so the fallback cannot silently drift.
DEFAULT_THRESHOLDS: list[MetricThreshold] = [
    MetricThreshold("answer_correctness", minimum=0.50, tolerance=0.10),
    MetricThreshold("answer_relevancy", minimum=0.60, tolerance=0.10),
    MetricThreshold("context_precision", minimum=0.50, tolerance=0.10),
    MetricThreshold("context_recall", minimum=0.50, tolerance=0.10),
    MetricThreshold("faithfulness", minimum=0.55, tolerance=0.10),
    MetricThreshold("hit_rate", minimum=0.60, tolerance=0.05),
    MetricThreshold("mrr", minimum=0.45, tolerance=0.05),
    MetricThreshold("recall_at_k", minimum=0.55, tolerance=0.05),
    MetricThreshold("precision_at_k", minimum=0.25, tolerance=0.05),
    MetricThreshold("ndcg_at_k", minimum=0.40, tolerance=0.05),
    MetricThreshold("citation_correctness", minimum=0.65, tolerance=0.05),
    MetricThreshold("citation_completeness", minimum=0.55, tolerance=0.10),
    MetricThreshold("grounded_answer_rate", minimum=0.70, tolerance=0.10),
    MetricThreshold("refusal_correctness", minimum=0.75, tolerance=0.10),
    MetricThreshold("unsupported_claim_rate", minimum=0.00, tolerance=0.00),
]

DEFAULT_PER_DIFFICULTY_THRESHOLDS: list[MetricThreshold] = [
    MetricThreshold("recall_at_k", minimum=0.35, tolerance=0.10),
    MetricThreshold("hit_rate", minimum=0.45, tolerance=0.10),
    MetricThreshold("citation_correctness", minimum=0.55, tolerance=0.05),
]

# Aggregate namespaces present in thresholds.yaml.
_AGGREGATE_NAMESPACES = ("ragas", "retrieval", "citation")


def _parse_threshold_block(block: dict[str, Any]) -> list[MetricThreshold]:
    """Convert a ``{metric: {minimum, tolerance}}`` mapping into MetricThresholds.

    Duplicate metric names are collapsed to their first occurrence so lookups are
    deterministic (aggregate checks iterate thresholds and take the first match).
    """
    parsed: list[MetricThreshold] = []
    seen: set[str] = set()
    for metric, cfg in (block or {}).items():
        if not isinstance(cfg, dict) or "minimum" not in cfg or metric in seen:
            continue
        seen.add(metric)
        parsed.append(
            MetricThreshold(
                metric=metric,
                minimum=float(cfg["minimum"]),
                tolerance=float(cfg.get("tolerance", 0.05)),
            )
        )
    return parsed


def load_thresholds(path: str | Path | None = None) -> ThresholdSet:
    """Load thresholds from ``evals/thresholds.yaml`` (the authoritative source).

    Falls back to DEFAULT_THRESHOLDS / DEFAULT_PER_DIFFICULTY_THRESHOLDS with a
    warning if PyYAML is unavailable or the file is missing or malformed.
    """
    thresholds_path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH

    def _fallback(reason: str) -> ThresholdSet:
        logger.warning(
            "Using built-in default thresholds (%s); source of truth is %s",
            reason,
            thresholds_path,
        )
        return ThresholdSet(
            aggregate=list(DEFAULT_THRESHOLDS),
            per_difficulty=list(DEFAULT_PER_DIFFICULTY_THRESHOLDS),
        )

    if yaml is None:
        return _fallback("PyYAML not installed")
    if not thresholds_path.exists():
        return _fallback("file not found")

    try:
        data = yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - corrupt file
        return _fallback(f"unreadable: {exc}")

    if not isinstance(data, dict):  # pragma: no cover - malformed top level
        return _fallback("unexpected document structure")

    aggregate: list[MetricThreshold] = []
    seen: set[str] = set()
    for namespace in _AGGREGATE_NAMESPACES:
        for threshold in _parse_threshold_block(data.get(namespace) or {}):
            if threshold.metric in seen:
                continue
            seen.add(threshold.metric)
            aggregate.append(threshold)

    per_difficulty = _parse_threshold_block(data.get("per_difficulty") or {})

    if not aggregate:  # pragma: no cover - empty/misnamed namespaces
        return _fallback("no aggregate metrics parsed")

    return ThresholdSet(aggregate=aggregate, per_difficulty=per_difficulty)


def resolve_thresholds(
    thresholds: list[MetricThreshold] | ThresholdSet | None,
    per_difficulty_thresholds: list[MetricThreshold] | None = None,
) -> ThresholdSet:
    """Normalise any accepted threshold input into a ThresholdSet.

    ``None`` loads the YAML (authoritative). A plain list is treated as the
    aggregate set, preserving the original public API for callers and tests that
    pass explicit thresholds.
    """
    if isinstance(thresholds, ThresholdSet):
        if per_difficulty_thresholds is None:
            return thresholds
        return ThresholdSet(
            aggregate=list(thresholds.aggregate),
            per_difficulty=list(per_difficulty_thresholds),
        )

    if thresholds is None:
        loaded = load_thresholds()
        if per_difficulty_thresholds is None:
            return loaded
        return ThresholdSet(
            aggregate=list(loaded.aggregate),
            per_difficulty=list(per_difficulty_thresholds),
        )

    return ThresholdSet(
        aggregate=list(thresholds),
        per_difficulty=list(per_difficulty_thresholds or []),
    )


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
    thresholds: list[MetricThreshold] | ThresholdSet | None = None,
    per_difficulty_thresholds: list[MetricThreshold] | None = None,
) -> RegressionReport:
    """
    Compare current results against a stored baseline.

    Parameters
    ----------
    current : ExperimentResults or dict
        Results from the current experiment run.
    baseline : ExperimentResults, dict, Path, or str
        Previously stored baseline (loaded from path if str/Path).
    thresholds : list[MetricThreshold] | ThresholdSet | None
        Metrics + minimum values + tolerances to check. ``None`` (the default)
        loads ``evals/thresholds.yaml``, the single source of truth.
    per_difficulty_thresholds : list[MetricThreshold] | None
        Override for the per-difficulty split tolerances. When omitted, they come
        from the same thresholds source as ``thresholds``.

    Returns
    -------
    RegressionReport with per-metric regression analysis.
    """
    resolved = resolve_thresholds(thresholds, per_difficulty_thresholds)

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
        for threshold in resolved.aggregate:
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
            diff_threshold = resolved.find("per_difficulty", key)
            if diff_threshold is None:
                diff_threshold = resolved.find("aggregate", key)
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
    thresholds: list[MetricThreshold] | ThresholdSet | None = None,
    verbose: bool = True,
    per_difficulty_thresholds: list[MetricThreshold] | None = None,
) -> bool:
    """
    Gate function for CI: returns True if any metric has regressed.

    Exits with code 1 (sys.exit(1)) when regressions are detected so that
    CI pipelines fail appropriately. Set ``verbose=False`` to suppress
    the printed report.

    ``thresholds=None`` loads ``evals/thresholds.yaml`` (single source of truth).
    """
    baseline = load_baseline(baseline_path)
    report = compare_baseline(
        current,
        baseline,
        thresholds=thresholds,
        per_difficulty_thresholds=per_difficulty_thresholds,
    )

    if verbose:
        print(report.summary())

    if report.has_regression():
        if verbose:
            logger.error(f"Regression detected in {report.regressed_metrics} metric(s)")
        sys.exit(1)
        return True  # unreachable after sys.exit

    return False
