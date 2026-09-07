"""Integration tests for EvaluationRunner (app.evaluation.runner)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from app.evaluation.datasets import Difficulty, GoldenDataset, GoldenExample
from app.evaluation.runner import (
    EXPERIMENTS,
    EvaluationRunner,
    ExperimentConfig,
    ExperimentResults,
    QuestionResult,
)
from app.retrieval.models import RetrievalResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_example(
    qid: str,
    difficulty: Difficulty = Difficulty.EASY,
    answerable: bool = True,
) -> GoldenExample:
    return GoldenExample(
        id=qid,
        question=f"What is {qid}?",
        expected_answer=f"The answer about {qid}.",
        relevant_chunk_ids=["doc1:0", "doc1:1"],  # must match _CHUNK_ID_RE pattern
        answerable=answerable,
        difficulty=difficulty,
    )

def _make_retrieval_result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"Chunk content for {chunk_id}",
        score=0.9,
        source="tests/fixtures/test.pdf",
    )

# ---------------------------------------------------------------------------
# Tests for ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.top_k == 20  # default from class definition
        assert cfg.experiment_name == "final"  # experiment_name is the preset key, not the label

    def test_experiment_presets(self) -> None:
        assert "vector" in EXPERIMENTS
        assert "bm25" in EXPERIMENTS
        assert "hybrid" in EXPERIMENTS
        assert "final" in EXPERIMENTS

# ---------------------------------------------------------------------------
# Tests for EvaluationRunner construction
# ---------------------------------------------------------------------------

class TestEvaluationRunnerConstruction:
    def test_runner_construction_with_mock(self) -> None:
        """Runner can be constructed with mock=True and a preset name."""
        runner = EvaluationRunner(experiment="final", mock=True)
        assert runner is not None
        # EXPERIMENTS["final"] has experiment_name="D — Full Pipeline"
        assert runner.cfg.experiment_name == "D — Full Pipeline"
        assert runner.mock is True

    def test_runner_lazy_init_skips_real_components(self) -> None:
        """mock=True should skip real component initialization."""
        runner = EvaluationRunner(experiment="final", mock=True)
        # Services remain None until explicitly initialized
        assert runner._embedder is None
        assert runner._vector_store is None
        assert runner._bm25 is None
        assert runner._gen_client is None

# ---------------------------------------------------------------------------
# Tests for QuestionResult
# ---------------------------------------------------------------------------

class TestQuestionResult:
    def test_question_result_attributes(self) -> None:
        qid = "q001"
        ex = _make_example(qid)
        result = QuestionResult(
            id=qid,
            example=ex,
            retrieved_chunk_ids=[f"chunk_{qid}_0"],
            retrieval_metrics={"recall_at_k": 0.5, "hit_rate": 1.0},
            citation_metrics={"citation_correctness": 0.8},
            ragas_scores={"faithfulness": 0.9},
            duration_seconds=1.5,
            error=None,
        )
        assert result.id == qid
        assert result.example.id == qid
        assert result.retrieved_chunk_ids == [f"chunk_{qid}_0"]
        assert result.ragas_scores == {"faithfulness": 0.9}
        assert result.error is None

    def test_question_result_with_error(self) -> None:
        ex = _make_example("q002")
        result = QuestionResult(
            id="q002",
            example=ex,
            retrieved_chunk_ids=[],
            retrieval_metrics={},
            citation_metrics={},
            ragas_scores=None,
            duration_seconds=0.1,
            error="Retrieval failed: connection timeout",
        )
        assert result.error is not None
        assert "Retrieval failed" in result.error

    def test_question_result_to_dict(self) -> None:
        ex = _make_example("q003")
        result = QuestionResult(
            id="q003",
            example=ex,
            retrieved_chunk_ids=["c1", "c2"],
            retrieval_metrics={"recall_at_k": 1.0},
            citation_metrics={},
            ragas_scores=None,
            duration_seconds=2.0,
            error=None,
        )
        d = result.to_dict()
        assert d["id"] == "q003"
        assert d["retrieved_chunk_ids"] == ["c1", "c2"]
        assert d["retrieval_metrics"] == {"recall_at_k": 1.0}

# ---------------------------------------------------------------------------
# Tests for EvaluationRunner.run (with mocked internals)
# ---------------------------------------------------------------------------

class TestEvaluationRunnerRun:
    def test_run_returns_experiment_results(self, tmp_path: Path) -> None:
        examples = [_make_example("q001"), _make_example("q002")]
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(ex.to_dict()) for ex in examples) + "\n",
            encoding="utf-8",
        )
        dataset = GoldenDataset.load(dataset_path, validate_chunk_ids=False)

        runner = EvaluationRunner(experiment="final", mock=True)

        # Patch the internal _evaluate_question to return deterministic results
        def mock_evaluate(example):
            return QuestionResult(
                id=example.id,
                example=example,
                retrieved_chunk_ids=["chunk_q001_0"],
                retrieval_metrics={"recall_at_k": 1.0},
                citation_metrics={},
                ragas_scores=None,
                duration_seconds=0.1,
                error=None,
            )

        with patch.object(runner, "_evaluate_question", side_effect=mock_evaluate):
            results = runner.run(dataset)

        assert isinstance(results, ExperimentResults)
        assert results.question_count == 2
        assert results.answerable_count == 2
        assert len(results.question_results) == 2

    def test_run_limits_questions(self, tmp_path: Path) -> None:
        examples = [_make_example(f"q{i:03d}") for i in range(5)]
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(ex.to_dict()) for ex in examples) + "\n",
            encoding="utf-8",
        )
        dataset = GoldenDataset.load(dataset_path)

        runner = EvaluationRunner(experiment="final", mock=True)

        def mock_evaluate(example):
            return QuestionResult(
                id=example.id,
                example=example,
                retrieved_chunk_ids=[],
                retrieval_metrics={},
                citation_metrics={},
                ragas_scores=None,
                duration_seconds=0.1,
                error=None,
            )

        with patch.object(runner, "_evaluate_question", side_effect=mock_evaluate):
            results = runner.run(dataset, question_limit=3)

        assert results.question_count == 3

    def test_run_catches_evaluation_error(self, tmp_path: Path) -> None:
        examples = [_make_example("qerr")]
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(ex.to_dict()) for ex in examples) + "\n",
            encoding="utf-8",
        )
        dataset = GoldenDataset.load(dataset_path, validate_chunk_ids=False)

        runner = EvaluationRunner(experiment="final", mock=True)

        def mock_evaluate_with_error(example):
            return QuestionResult(
                id=example.id,
                example=example,
                retrieved_chunk_ids=[],
                retrieval_metrics={},
                citation_metrics={},
                ragas_scores=None,
                duration_seconds=0.1,
                error="Generation timeout",
            )

        with patch.object(
            runner, "_evaluate_question", side_effect=mock_evaluate_with_error
        ):
            results = runner.run(dataset)

        assert any(qr.error is not None for qr in results.question_results)

    def test_unanswerable_question_handling(self, tmp_path: Path) -> None:
        examples = [
            _make_example("q001", answerable=True),
            _make_example("q002", answerable=False, difficulty=Difficulty.UNANSWERABLE),
        ]
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(ex.to_dict()) for ex in examples) + "\n",
            encoding="utf-8",
        )
        dataset = GoldenDataset.load(dataset_path, validate_chunk_ids=False)

        runner = EvaluationRunner(experiment="final", mock=True)

        answerable_result = QuestionResult(
            id="q001",
            example=examples[0],
            retrieved_chunk_ids=["c1"],
            retrieval_metrics={"recall_at_k": 1.0},
            citation_metrics={"citation_correctness": 1.0},
            ragas_scores=None,
            duration_seconds=0.1,
            error=None,
        )
        unanswerable_result = QuestionResult(
            id="q002",
            example=examples[1],
            retrieved_chunk_ids=[],
            retrieval_metrics={},
            citation_metrics={},
            ragas_scores=None,
            duration_seconds=0.1,
            error=None,
        )

        with patch.object(
            runner,
            "_evaluate_question",
            side_effect=[answerable_result, unanswerable_result],
        ):
            results = runner.run(dataset)

        assert results.question_count == 2
        assert results.answerable_count == 1
        assert results.unanswerable_count == 1

    def test_experiment_results_per_difficulty(self, tmp_path: Path) -> None:
        examples = [
            _make_example("q001", difficulty=Difficulty.EASY),
            _make_example("q002", difficulty=Difficulty.HARD),
        ]
        dataset_path = tmp_path / "golden.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(ex.to_dict()) for ex in examples) + "\n",
            encoding="utf-8",
        )
        dataset = GoldenDataset.load(dataset_path)

        runner = EvaluationRunner(experiment="final", mock=True)

        def mock_evaluate(example):
            return QuestionResult(
                id=example.id,
                example=example,
                retrieved_chunk_ids=["c1"],
                retrieval_metrics={"recall_at_k": 1.0},
                citation_metrics={},
                ragas_scores=None,
                duration_seconds=0.1,
                error=None,
            )

        with patch.object(runner, "_evaluate_question", side_effect=mock_evaluate):
            results = runner.run(dataset)

        assert "easy" in results.per_difficulty
        assert "hard" in results.per_difficulty
