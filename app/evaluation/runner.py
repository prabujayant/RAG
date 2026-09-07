"""
Evaluation runner: orchestrates retrieval → generation → grounding for experiments.

Provides four experiment configurations:
  A — Vector (dense) retrieval only
  B — BM25 (sparse) retrieval only
  C — Hybrid (dense + sparse) retrieval
  D — Full pipeline (hybrid + reranking + evidence selection + generation + grounding)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.embeddings import Embedder
from app.generation.client import LLMClient, MockLLM, OpenRouterClient
from app.generation.schemas import AnswerResponse, Citation, GroundingStatus
from app.generation.service import GenerationService
from app.grounding import GroundingValidator
from app.retrieval.bm25 import BM25Indexer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector import VectorStore

from .citation_metrics import compute_citation_metrics
from .datasets import GoldenDataset, GoldenExample, load_golden_dataset
from .ragas_eval import RagasEvaluator
from .retrieval_metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuration dataclasses
# ----------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for a single evaluation experiment."""

    experiment_name: str = "final"
    use_vector: bool = True
    use_bm25: bool = False
    use_reranker: bool = False
    use_evidence_selector: bool = False
    use_grounding: bool = False
    use_ragas: bool = False
    top_k: int = 20
    rerank_top_k: int = 10
    citation_top_k: int = 5


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "vector": ExperimentConfig(experiment_name="A — Vector Only", use_vector=True, use_bm25=False),
    "bm25": ExperimentConfig(experiment_name="B — BM25 Only", use_vector=False, use_bm25=True),
    "hybrid": ExperimentConfig(experiment_name="C — Hybrid", use_vector=True, use_bm25=True),
    "final": ExperimentConfig(
        experiment_name="D — Full Pipeline",
        use_vector=True,
        use_bm25=True,
        use_reranker=True,
        use_evidence_selector=True,
        use_grounding=True,
        use_ragas=True,
        top_k=20,
        rerank_top_k=10,
        citation_top_k=5,
    ),
}


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Result for a single question in an experiment."""

    id: str
    example: GoldenExample
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    citation_metrics: dict[str, float] = field(default_factory=dict)
    ragas_scores: dict[str, float] | None = None
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Aggregated results for an entire experiment run."""

    experiment_name: str
    experiment_label: str
    dataset_path: str
    question_count: int
    answerable_count: int
    unanswerable_count: int
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    question_results: list[QuestionResult] = field(default_factory=list)
    aggregate_retrieval: dict[str, float] = field(default_factory=dict)
    aggregate_citation: dict[str, float] = field(default_factory=dict)
    aggregate_ragas: dict[str, float] = field(default_factory=dict)
    per_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["question_results"] = [qr.to_dict() for qr in self.question_results]
        return d


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


class EvaluationRunner:
    """
    Orchestrates the full evaluation pipeline for a given experiment config.

    Usage::

        runner = EvaluationRunner(experiment="final", mock=True)
        results = runner.run(dataset, question_limit=10)
    """

    def __init__(
        self,
        experiment: ExperimentConfig | str = "final",
        mock: bool = False,
        settings: Settings | None = None,
        _retriever: HybridRetriever | None = None,
        _reranker: Reranker | None = None,
        _generation_client: LLMClient | None = None,
    ):
        if isinstance(experiment, str):
            if experiment not in EXPERIMENTS:
                raise ValueError(f"Unknown experiment {experiment!r}. Choices: {list(EXPERIMENTS)}")
            self.cfg = EXPERIMENTS[experiment]
        else:
            self.cfg = experiment
        self.mock = mock
        self.settings = settings or get_settings()
        self._embedder: Embedder | None = None
        self._vector_store: VectorStore | None = None
        self._bm25: BM25Indexer | None = None
        self._reranker: Reranker | None = None
        self._retriever: HybridRetriever | None = None
        self._gen_client: LLMClient | None = None
        self._gen_service: GenerationService | None = None
        self._grounding: GroundingValidator | None = None
        self._ragas: RagasEvaluator | None = None
        # Allow injected mock components for testing
        if _retriever is not None:
            self._retriever = _retriever
        if _reranker is not None:
            self._reranker = _reranker
        if _generation_client is not None:
            self._gen_client = _generation_client

    # ------------------------------------------------------------------
    # Lazy initialisation of components
    # ------------------------------------------------------------------

    @property
    def _init_done(self) -> bool:
        """Return True once at least one component has been lazily initialized."""
        return any(
            getattr(self, attr, None) is not None
            for attr in (
                "_embedder",
                "_vector_store",
                "_bm25",
                "_reranker",
                "_retriever",
                "_gen_service",
                "_grounding",
                "_ragas",
            )
        )

    def _init_embedder(self) -> None:
        if self._embedder is None:
            self._embedder = Embedder()

    def _init_vector_store(self) -> None:
        if self._vector_store is None:
            self._init_embedder()
            self._vector_store = VectorStore(settings=self.settings)

    def _init_bm25(self) -> None:
        if self._bm25 is None:
            self._bm25 = BM25Indexer()

    def _init_reranker(self) -> None:
        if self._reranker is None and self.cfg.use_reranker:
            self._reranker = Reranker()

    def _init_retriever(self) -> None:
        if self._retriever is None:
            self._init_vector_store()
            self._init_bm25()
            self._retriever = HybridRetriever(
                settings=self.settings,
                embedder=self._embedder,
                vector_store=self._vector_store,
                bm25_indexer=self._bm25,
            )

    def _init_generation(self) -> None:
        if self._gen_service is None:
            if self.mock:
                self._gen_client = MockLLM()
            else:
                self._gen_client = OpenRouterClient(settings=self.settings)
            self._gen_service = GenerationService(client=self._gen_client)

    def _init_grounding(self) -> None:
        if self._grounding is None and self.cfg.use_grounding:
            self._init_generation()
            self._grounding = GroundingValidator(llm_client=self._gen_client)

    def _init_ragas(self) -> None:
        if self._ragas is None and self.cfg.use_ragas:
            self._ragas = RagasEvaluator(disabled=not self.settings.openrouter_api_key, mock=self.mock)

    # ------------------------------------------------------------------
    # Per-question evaluation
    # ------------------------------------------------------------------

    def _evaluate_question(
        self, example: GoldenExample
    ) -> QuestionResult:
        """Evaluate a single question through the configured pipeline."""
        qid = example.id
        question = example.question
        expected_chunks = set(example.relevant_chunk_ids)
        answerable = example.answerable
        start_time = time.perf_counter()

        result = QuestionResult(
            id=qid,
            example=example,
        )

        try:
            # ---- Retrieval ----
            self._init_retriever()
            assert self._retriever is not None
            retrieved = self._retriever.retrieve(question, top_k=self.cfg.top_k)
            result.retrieved_chunk_ids = [r.chunk_id for r in retrieved]

            # ---- Reranking ----
            reranked_chunk_ids: list[str] = []
            if self.cfg.use_reranker and self._reranker:
                reranked = self._reranker.rerank(question, retrieved, top_k=self.cfg.rerank_top_k)
                reranked_chunk_ids = [r.chunk_id for r in reranked]

            # ---- Generation + Grounding ----
            generated_answer = ""
            citations: list[dict[str, Any]] = []
            if self.cfg.use_grounding:
                self._init_grounding()
                # Build a minimal AnswerResponse from retrieved chunks for validation
                chunks_to_use = reranked_chunk_ids or result.retrieved_chunk_ids
                chunks_for_grounding = [r for r in retrieved if r.chunk_id in chunks_to_use]
                response_for_validation = AnswerResponse(
                    answer="",
                    citations=[
                        Citation(citation_id=f"[C{i}]", chunk_id=r.chunk_id, text=r.text)
                        for i, r in enumerate(chunks_for_grounding, start=1)
                    ],
                    grounding_status=GroundingStatus.PARTIALLY_GROUNDED,
                )
                assert self._grounding is not None
                grounded_response = self._grounding.validate(response_for_validation)
                generated_answer = grounded_response.answer
                citations = [
                    {"citation_id": c.citation_id, "chunk_id": c.chunk_id, "text": c.text}
                    for c in grounded_response.citations
                ]
            else:
                self._init_generation()
                assert self._gen_service is not None
                candidates_for_gen = [r for r in retrieved if r.chunk_id in result.retrieved_chunk_ids]
                gen_output = self._gen_service.generate(question, candidates=candidates_for_gen)
                generated_answer = gen_output.answer

            # ---- Citation / Grounding metrics ----
            citation_metrics = compute_citation_metrics(
                question=question,
                answer=generated_answer,
                citations=citations,
                expected_chunk_ids=list(expected_chunks),
                answerable=answerable,
            )
            result.citation_metrics = citation_metrics

            # ---- Retrieval metrics ----
            retrieved_ids = reranked_chunk_ids or result.retrieved_chunk_ids
            result.retrieval_metrics = compute_retrieval_metrics(
                retrieved_ids=retrieved_ids,
                expected_ids=list(expected_chunks),
            )

            # ---- Ragas ----
            ragas_scores: dict[str, Any] | None = None
            if self.cfg.use_ragas:
                self._init_ragas()
                assert self._ragas is not None
                ragas_result = self._ragas.evaluate(
                    question=question,
                    answer=generated_answer,
                    contexts=[r.text for r in retrieved],
                )
                if ragas_result:
                    ragas_scores = ragas_result.to_dict()
            result.ragas_scores = ragas_scores

            result.duration_seconds = time.perf_counter() - start_time

        except Exception as e:
            logger.warning(f"Error evaluating question {qid}: {e}")
            result.error = str(e)
            result.duration_seconds = time.perf_counter() - start_time

        return result

    # ------------------------------------------------------------------
    # Main run entry point
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: GoldenDataset,
        question_limit: int | None = None,
        output_dir: Path | None = None,
    ) -> ExperimentResults:
        """
        Run the full evaluation pipeline.

        Parameters
        ----------
        dataset : GoldenDataset
            The dataset to evaluate on.
        question_limit : int | None
            Limit the number of questions to evaluate (for quick smoke tests).
        output_dir : Path | None
            Directory to write results JSON (defaults to evals/results/).

        Returns
        -------
        ExperimentResults with per-question and aggregate metrics.
        """
        experiment_id = uuid.uuid4().hex[:8]
        logger.info(
            f"Starting evaluation experiment={self.cfg.experiment_name} "
            f"mock={self.mock} limit={question_limit}"
        )

        start_dt = datetime.now(UTC)
        start_iso = start_dt.isoformat()

        examples = dataset.examples
        if question_limit:
            examples = examples[:question_limit]

        question_results: list[QuestionResult] = []
        all_retrieval: list[dict[str, float]] = []
        all_citation: list[dict[str, float]] = []
        all_ragas: list[dict[str, float]] = []
        per_difficulty: dict[str, Any] = {}

        for example in examples:
            qr = self._evaluate_question(example)
            question_results.append(qr)
            all_retrieval.append(qr.retrieval_metrics)
            all_citation.append(qr.citation_metrics)
            if qr.ragas_scores is not None:
                # RagasResult is a dataclass; extract numeric values
                for k in ("faithfulness", "answer_correctness", "answer_relevance", "context_precision"):
                    v = qr.ragas_scores.get(k)
                    if v is not None:
                        all_ragas.append({k: float(v)})
            difficulty = str(qr.example.difficulty.value)
            if difficulty not in per_difficulty:
                per_difficulty[difficulty] = []
            per_difficulty[difficulty].append({
                "retrieval": qr.retrieval_metrics,
                "citation": qr.citation_metrics,
            })

        end_dt = datetime.now(UTC)
        duration = (end_dt - start_dt).total_seconds()

        # Aggregate
        agg_retrieval = _aggregate_metric_list(all_retrieval)
        agg_citation = _aggregate_metric_list(all_citation)
        agg_ragas = _aggregate_metric_list(all_ragas)

        difficulty_agg: dict[str, Any] = {}
        for diff, items in per_difficulty.items():
            diff_retrieval = [item["retrieval"] for item in items]
            diff_citation = [item["citation"] for item in items]
            difficulty_agg[diff] = {
                "count": float(len(items)),
                **_aggregate_metric_list(diff_retrieval),
                **_aggregate_metric_list(diff_citation),
            }

        experiment_results = ExperimentResults(
            experiment_name=self.cfg.experiment_name,
            experiment_label=self.cfg.experiment_name,
            dataset_path=str(dataset.path),
            question_count=len(question_results),
            answerable_count=sum(1 for q in question_results if q.example.answerable),
            unanswerable_count=sum(1 for q in question_results if not q.example.answerable),
            start_time=start_iso,
            end_time=end_dt.isoformat(),
            duration_seconds=duration,
            question_results=question_results,
            aggregate_retrieval=agg_retrieval,
            aggregate_citation=agg_citation,
            aggregate_ragas=agg_ragas,
            per_difficulty=difficulty_agg,
        )

        # Persist
        if output_dir is None:
            output_dir = Path("evals/results/")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"experiment_{self.cfg.experiment_name}_{experiment_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(experiment_results.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to {out_file}")

        return experiment_results


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _aggregate_metric_list(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean of each key across a list of metric dicts."""
    if not metrics_list:
        return {}
    import numpy as np

    keys = set(k for m in metrics_list for k in m)
    result = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if vals:
            result[key] = float(np.mean(vals))
    return result


# ----------------------------------------------------------------------
# Public convenience functions (used by run.py CLI)
# ----------------------------------------------------------------------


def run_evaluation(
    experiment: str = "final",
    mock: bool = False,
    question_limit: int | None = None,
    output_dir: str | None = None,
) -> ExperimentResults:
    """
    Convenience wrapper around EvaluationRunner.run().

    Loads the default dataset and runs the requested experiment.
    """
    settings = get_settings()
    dataset = load_golden_dataset(settings.eval_dataset_path)
    runner = EvaluationRunner(experiment=experiment, mock=mock, settings=settings)
    out_path = Path(output_dir) if output_dir else None
    return runner.run(dataset, question_limit=question_limit, output_dir=out_path)
