"""
Ragas evaluation wrapper with disabled/mock mode.

Ragas (Retrieval-Augmented Generation Assessment) provides LLM-based
evaluation metrics. This module wraps ragas with graceful fallback when
no API key is available or when evaluation is disabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RagasScore:
    """Container for a single Ragas metric score."""

    name: str
    value: float | None  # None when disabled / errored
    error: str | None = None


@dataclass
class RagasResult:
    """Container for all Ragas metric results from a single evaluation."""

    question: str
    answer: str
    contexts: list[str]
    metrics: dict[str, RagasScore] = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "metrics": {
                name: {"value": s.value, "error": s.error} for name, s in self.metrics.items()
            },
            "success": self.success,
        }


class RagasEvaluator:
    """
    Wrapper around ragas evaluation that gracefully handles missing API keys.

    When `disabled=True`, all evaluate() calls return None without making
    any LLM API calls. This is useful for mock mode or when LLM evaluation
    is not yet configured.
    """

    def __init__(self, disabled: bool = False, mock: bool = False):
        self.disabled = disabled
        self.mock = mock
        self._client: Any = None
        self._evaluator: Any = None
        self._initialized = False

    def _initialize(self) -> bool:
        """Lazily initialise the ragas evaluator. Returns True on success."""
        if self._initialized:
            return self._client is not None
        self._initialized = True

        if self.disabled or self.mock:
            logger.info("RagasEvaluator: disabled/mock mode — skipping ragas initialisation")
            return False

        settings = get_settings()
        if not settings.openrouter_api_key:
            logger.warning("RagasEvaluator: no openrouter_api_key — ragas disabled")
            return False

        try:
            from ragas.evaluation import Evaluator
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            self._evaluator = Evaluator(
                metrics=[
                    faithfulness,
                    answer_correctness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
            )
            logger.info("RagasEvaluator: initialised successfully")
            return True
        except ImportError as e:
            logger.warning(f"RagasEvaluator: ragas not installed — {e}")
            return False
        except Exception as e:
            logger.warning(f"RagasEvaluator: initialisation failed — {e}")
            return False

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RagasResult | None:
        """Run Ragas evaluation for a single question."""
        result = RagasResult(question=question, answer=answer, contexts=contexts)

        if self._initialize() is False:
            # Return a result with all None values so the caller can
            # still record that evaluation was skipped
            return result

        try:
            from ragas import EvaluationDataset

            # Build a ragas-compatible dataset with a single row
            row = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": "",  # we don't have ground-truth answers here
            }
            dataset = EvaluationDataset([row])

            # Run evaluation
            scores = self._evaluator.evaluate(dataset)

            # Extract per-metric scores
            score_dict = scores.to_pandas().iloc[0].to_dict()
            metric_names = [
                "faithfulness",
                "answer_correctness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            for name in metric_names:
                val = score_dict.get(name)
                result.metrics[name] = RagasScore(
                    name=name,
                    value=float(val) if val is not None and str(val) != "nan" else None,
                    error=None,
                )

            result.success = True
            logger.debug(f"RagasEvaluator: evaluated question={question[:50]!r} — success")

        except Exception as e:
            logger.warning(f"RagasEvaluator: evaluation error for question={question[:50]!r} — {e}")
            for name in ["faithfulness", "answer_correctness", "answer_relevancy",
                         "context_precision", "context_recall"]:
                result.metrics[name] = RagasScore(name=name, value=None, error=str(e))

        return result

    def evaluate_batch(
        self, items: list[dict[str, str | list[str]]]
    ) -> list[RagasResult | None]:
        """
        Run Ragas evaluation for a batch of items.

        Each item must have keys: ``question``, ``answer``, ``contexts``.
        Returns a list of results (same length as input). Items that
        could not be evaluated produce None entries.
        """
        return [self.evaluate(i["question"], i["answer"], i["contexts"]) for i in items]  # type: ignore[arg-type]
