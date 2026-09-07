"""Unit tests for app.evaluation.regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.evaluation.regression import (
    RegressionEntry,
    RegressionReport,
    check_regression,
    compare_baseline,
    load_baseline,
    save_baseline,
)

BASELINE_DATA = {
    "experiment_name": "final",
    "question_count": 50,
    "aggregate_retrieval": {
        "recall_at_k": 0.60,
        "precision_at_k": 0.30,
        "mrr": 0.50,
        "ndcg_at_k": 0.45,
        "hit_rate": 0.60,
    },
    "aggregate_citation": {
        "citation_correctness": 0.70,
        "citation_completeness": 0.60,
        "grounded_answer_rate": 0.75,
        "refusal_correctness": 0.80,
    },
    "aggregate_ragas": {},
    "per_difficulty": {
        "easy": {"recall_at_k": 0.80, "hit_rate": 0.80, "citation_correctness": 0.90},
        "medium": {"recall_at_k": 0.60, "hit_rate": 0.60, "citation_correctness": 0.70},
    },
}

class TestLoadSaveBaseline:
    def test_save_and_load(self, tmp_path: Path) -> None:
        p = tmp_path / "baseline.json"

        class FakeResults:
            def to_dict(self):
                return BASELINE_DATA

        save_baseline(FakeResults(), p)
        loaded = load_baseline(p)
        assert loaded["experiment_name"] == "final"
        assert loaded["aggregate_retrieval"]["recall_at_k"] == 0.60

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")

class TestCompareBaseline:
    def test_no_regression_when_improved(self) -> None:
        current = {
            "experiment_name": "final",
            "question_count": 50,
            "aggregate_retrieval": {
                "recall_at_k": 0.70,  # improved
                "precision_at_k": 0.30,
                "mrr": 0.50,
                "ndcg_at_k": 0.45,
                "hit_rate": 0.60,
            },
            "aggregate_citation": BASELINE_DATA["aggregate_citation"],
            "aggregate_ragas": {},
            "per_difficulty": {},
        }
        report = compare_baseline(current, BASELINE_DATA)
        assert report.regressed_metrics == 0
        assert not report.has_regression()

    def test_regression_on_absolute_minimum(self) -> None:
        """If recall drops below minimum threshold, it's a regression."""
        current = {
            "experiment_name": "final",
            "question_count": 50,
            "aggregate_retrieval": {
                "recall_at_k": 0.20,  # below minimum of 0.30
                "precision_at_k": 0.30,
                "mrr": 0.50,
                "ndcg_at_k": 0.45,
                "hit_rate": 0.60,
            },
            "aggregate_citation": BASELINE_DATA["aggregate_citation"],
            "aggregate_ragas": {},
            "per_difficulty": {},
        }
        report = compare_baseline(current, BASELINE_DATA)
        assert report.regressed_metrics > 0
        assert any("recall_at_k" in e.metric for e in report.entries)

    def test_regression_within_tolerance(self) -> None:
        """Small drops within tolerance are not regressions."""
        current = {
            "experiment_name": "final",
            "question_count": 50,
            "aggregate_retrieval": {
                "recall_at_k": 0.59,  # 1% drop within 5% tolerance
                "precision_at_k": 0.30,
                "mrr": 0.50,
                "ndcg_at_k": 0.45,
                "hit_rate": 0.60,
            },
            "aggregate_citation": BASELINE_DATA["aggregate_citation"],
            "aggregate_ragas": {},
            "per_difficulty": {},
        }
        report = compare_baseline(current, BASELINE_DATA)
        assert report.regressed_metrics == 0

    def test_regression_beyond_tolerance(self) -> None:
        """Drop beyond tolerance counts as regression."""
        current = {
            "experiment_name": "final",
            "question_count": 50,
            "aggregate_retrieval": {
                "recall_at_k": 0.50,  # ~17% drop, beyond 5% tolerance
                "precision_at_k": 0.30,
                "mrr": 0.50,
                "ndcg_at_k": 0.45,
                "hit_rate": 0.60,
            },
            "aggregate_citation": BASELINE_DATA["aggregate_citation"],
            "aggregate_ragas": {},
            "per_difficulty": {},
        }
        report = compare_baseline(current, BASELINE_DATA)
        assert report.regressed_metrics > 0

    def test_difficulty_regression(self) -> None:
        current = {
            "experiment_name": "final",
            "question_count": 50,
            "aggregate_retrieval": {
                "recall_at_k": 0.60,
                "precision_at_k": 0.30,
                "mrr": 0.50,
                "ndcg_at_k": 0.45,
                "hit_rate": 0.60,
            },
            "aggregate_citation": BASELINE_DATA["aggregate_citation"],
            "aggregate_ragas": {},
            "per_difficulty": {
                "easy": {
                    "recall_at_k": 0.50,  # was 0.80
                    "hit_rate": 0.50,
                    "citation_correctness": 0.90,
                },
                "medium": BASELINE_DATA["per_difficulty"]["medium"],
            },
        }
        report = compare_baseline(current, BASELINE_DATA)
        # difficulty_regressions is a dict of lists keyed by difficulty name
        assert "easy" in report.difficulty_regressions

class TestRegressionReport:
    def test_summary_no_regression(self) -> None:
        report = RegressionReport(
            experiment_name="final",
            timestamp="2024-01-01T00:00:00Z",
            total_metrics=10,
            regressed_metrics=0,
            entries=[],
        )
        summary = report.summary()
        assert "PASS" in summary

    def test_summary_with_regression(self) -> None:
        entry = RegressionEntry(
            metric="recall_at_k",
            baseline=0.60,
            current=0.50,
            absolute_gap=-0.10,
            relative_gap=-0.1667,
            regressed=True,
        )
        report = RegressionReport(
            experiment_name="final",
            timestamp="2024-01-01T00:00:00Z",
            total_metrics=10,
            regressed_metrics=1,
            entries=[entry],
        )
        summary = report.summary()
        assert "REGRESSION" in summary
        assert "recall_at_k" in summary
        assert "-0.1000" in summary

class TestCheckRegression:
    def test_check_regression_exits_on_regression(self, tmp_path: Path) -> None:
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(BASELINE_DATA), encoding="utf-8")

        current = {
            "experiment_name": "final",
            "question_count": 50,
            "aggregate_retrieval": {
                "recall_at_k": 0.20,
                "precision_at_k": 0.30,
                "mrr": 0.50,
                "ndcg_at_k": 0.45,
                "hit_rate": 0.60,
            },
            "aggregate_citation": BASELINE_DATA["aggregate_citation"],
            "aggregate_ragas": {},
            "per_difficulty": {},
        }
        with pytest.raises(SystemExit):
            check_regression(current, p, verbose=False)
