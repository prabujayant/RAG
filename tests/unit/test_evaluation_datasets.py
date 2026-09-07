"""Unit tests for app.evaluation.datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.evaluation.datasets import (
    Difficulty,
    GoldenDataset,
    GoldenExample,
    _parse_example,
)

# ---------------------------------------------------------------------------
# GoldenExample construction
# ---------------------------------------------------------------------------

def test_golden_example_defaults() -> None:
    ex = GoldenExample(
        id="q001",
        question="What is Python?",
        expected_answer="A programming language.",
        relevant_chunk_ids=["doc1:0"],
        answerable=True,
        difficulty=Difficulty.EASY,
    )
    assert ex.id == "q001"
    assert ex.answerable is True
    assert ex.difficulty == Difficulty.EASY

def test_golden_example_str_difficulty() -> None:
    """String difficulty values are coerced to Difficulty enum."""
    ex = GoldenExample(
        id="q002",
        question="What is Python?",
        expected_answer="",
        relevant_chunk_ids=[],
        answerable=False,
        difficulty="unanswerable",
    )
    assert ex.difficulty == Difficulty.UNANSWERABLE
    assert ex.answerable is False

# ---------------------------------------------------------------------------
# GoldenDataset
# ---------------------------------------------------------------------------

def test_golden_dataset_filter_answerable() -> None:
    examples = [
        GoldenExample(id="q001", question="Q1", expected_answer="A",
                      relevant_chunk_ids=["c:0"], answerable=True, difficulty=Difficulty.EASY),
        GoldenExample(id="q002", question="Q2", expected_answer="A",
                      relevant_chunk_ids=["c:1"], answerable=True, difficulty=Difficulty.MEDIUM),
        GoldenExample(id="q003", question="Q3", expected_answer="",
                      relevant_chunk_ids=[], answerable=False, difficulty=Difficulty.UNANSWERABLE),
    ]
    ds = GoldenDataset(examples)

    answerable = ds.filter(answerable=True)
    assert len(answerable) == 2
    assert all(e.answerable for e in answerable)

    unanswerable = ds.filter(answerable=False)
    assert len(unanswerable) == 1
    assert unanswerable[0].id == "q003"

def test_golden_dataset_filter_difficulty() -> None:
    examples = [
        GoldenExample(id="q001", question="Q1", expected_answer="A",
                      relevant_chunk_ids=["c:0"], answerable=True, difficulty=Difficulty.EASY),
        GoldenExample(id="q002", question="Q2", expected_answer="A",
                      relevant_chunk_ids=["c:1"], answerable=True, difficulty=Difficulty.MEDIUM),
    ]
    ds = GoldenDataset(examples)
    filtered = ds.filter(difficulty=Difficulty.MEDIUM)
    assert len(filtered) == 1
    assert filtered[0].id == "q002"

def test_golden_dataset_limit() -> None:
    examples = [
        GoldenExample(id=f"q{i:03d}", question=f"Q{i}", expected_answer="A",
                      relevant_chunk_ids=[f"c:{i}"], answerable=True, difficulty=Difficulty.EASY)
        for i in range(10)
    ]
    ds = GoldenDataset(examples)
    limited = ds.limit(3)
    assert len(limited) == 3
    assert limited[0].id == "q000"

def test_golden_dataset_summary() -> None:
    examples = [
        GoldenExample(id="q001", question="Q1", expected_answer="A",
                      relevant_chunk_ids=["c:0"], answerable=True, difficulty=Difficulty.EASY),
        GoldenExample(id="q002", question="Q2", expected_answer="A",
                      relevant_chunk_ids=["c:1"], answerable=True, difficulty=Difficulty.MEDIUM),
        GoldenExample(id="q003", question="Q3", expected_answer="",
                      relevant_chunk_ids=[], answerable=False, difficulty=Difficulty.UNANSWERABLE),
    ]
    ds = GoldenDataset(examples)
    s = ds.summary()
    assert s.total == 3
    assert s.answerable == 2
    assert s.unanswerable == 1
    assert s.by_difficulty["easy"] == 1
    assert s.by_difficulty["medium"] == 1
    assert s.by_difficulty["unanswerable"] == 1

# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------

def test_golden_dataset_jsonl_roundtrip(tmp_path: Path) -> None:
    import json

    examples = [
        GoldenExample(
            id="q001",
            question="What is Python?",
            expected_answer="A programming language.",
            relevant_chunk_ids=["doc1:0", "doc1:1"],
            answerable=True,
            difficulty=Difficulty.MEDIUM,
        ),
        GoldenExample(
            id="q002",
            question="What is unreachable?",
            expected_answer="",
            relevant_chunk_ids=[],
            answerable=False,
            difficulty=Difficulty.UNANSWERABLE,
        ),
    ]
    p = tmp_path / "golden.jsonl"
    p.write_text(
        "\n".join(json.dumps(ex.to_dict()) for ex in examples) + "\n",
        encoding="utf-8",
    )

    loaded = GoldenDataset.load(p)
    assert len(loaded) == 2
    assert loaded[0].id == "q001"
    assert loaded[1].answerable is False

def test_golden_dataset_duplicate_id_raises(tmp_path: Path) -> None:
    p = tmp_path / "golden.jsonl"
    p.write_text(
        '{"id":"q001","question":"Q1","expected_answer":"A","relevant_chunk_ids":[],"answerable":true,"difficulty":"easy"}\n'
        '{"id":"q001","question":"Q2","expected_answer":"B","relevant_chunk_ids":[],"answerable":true,"difficulty":"easy"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        GoldenDataset.load(p)

def test_parse_example_validates_chunk_id_format() -> None:
    obj = {
        "id": "q001",
        "question": "Q",
        "expected_answer": "A",
        "relevant_chunk_ids": ["invalid-format"],
        "answerable": True,
        "difficulty": "easy",
    }
    with pytest.raises(ValueError, match="chunk id"):
        _parse_example(obj, line_number=1, validate_chunk_ids=True)
