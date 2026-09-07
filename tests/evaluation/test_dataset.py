"""Tests for the golden evaluation dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_dataset import (
    DIFFICULTIES,
    PER_DIFFICULTY_MIN,
    GoldenDataset,
    GoldenExample,
    validate_distribution,
)

SCRIPT_BASEDIR = Path(__file__).resolve().parents[1] / "scripts"

@pytest.fixture(scope="module")
def golden(repo_root: Path) -> GoldenDataset:
    return GoldenDataset.from_jsonl(repo_root / "evals" / "dataset" / "golden.jsonl")

def test_dataset_loads(golden: GoldenDataset) -> None:
    assert len(golden.examples) >= 50

def test_difficulty_distribution(golden: GoldenDataset) -> None:
    problems = validate_distribution(golden)
    assert problems == []
    summary = golden.summary()
    for difficulty in DIFFICULTIES:
        assert summary[difficulty] >= PER_DIFFICULTY_MIN[difficulty]

def test_ids_unique(golden: GoldenDataset) -> None:
    ids = [ex.id for ex in golden.examples]
    assert len(ids) == len(set(ids))

def test_answerable_flag_consistent_with_difficulty(golden: GoldenDataset) -> None:
    for ex in golden.examples:
        if ex.difficulty == "unanswerable":
            assert not ex.answerable
            assert ex.relevant_chunk_ids == [], f"{ex.id} has chunk refs"
        else:
            assert ex.answerable

def test_answerable_questions_have_relevant_chunks(golden: GoldenDataset) -> None:
    for ex in golden.examples:
        if ex.answerable:
            assert len(ex.relevant_chunk_ids) >= 1, f"{ex.id} has no relevant chunks"

def test_unanswerable_questions_have_no_expected_answer(golden: GoldenDataset) -> None:
    for ex in golden.examples:
        if not ex.answerable:
            assert ex.expected_answer in ("", None)

def test_questions_are_not_heading_copies(golden: GoldenDataset) -> None:
    """Ensure questions are genuine questions, not just document headings."""
    for ex in golden.examples:
        q = ex.question.strip().lower()
        assert q.endswith("?") or q.startswith(("what", "how", "which", "compare", "list", "walk"))

def test_validate_distribution_reports_shortfall() -> None:
    ds = GoldenDataset(examples=[GoldenExample(id="x1", question="?", difficulty="easy")])
    problems = validate_distribution(ds)
    assert any("easy" in p for p in problems)
    assert any("medium" in p for p in problems)
    assert any("unanswerable" in p for p in problems)
