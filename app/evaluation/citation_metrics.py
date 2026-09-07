"""Citation and grounding metrics.

These metrics operate on a single :class:`AnswerResponse` together with
the relevant chunk ids from the golden example. They are:

- :func:`citation_correctness`     — fraction of cited chunk ids that are
  actually present in the evidence.
- :func:`citation_completeness`    — fraction of relevant chunk ids that
  appear in the citations.
- :func:`citation_precision`       — ``|cited ∩ relevant| / |cited|``
- :func:`grounded_answer_rate`     — fraction of answerable answers that
  produced a ``GroundingStatus.GROUNDED`` response.
- :func:`refusal_correctness`      — fraction of unanswerable questions
  where the model refused (and vice versa for answerable).

All functions are pure; they do not call LLMs, databases, or Ragas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.generation.schemas import AnswerResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-answer metrics
# ---------------------------------------------------------------------------


def cited_chunk_ids(answer: AnswerResponse) -> set[str]:
    """Return the set of chunk_ids that the answer cited."""
    return {c.chunk_id for c in answer.citations}


def citation_correctness(
    answer: AnswerResponse,
    relevant_ids: set[str],
    *,
    candidate_chunk_ids: set[str] | None = None,
) -> float:
    """Fraction of cited chunk_ids that are valid (exist in evidence).

    ``candidate_chunk_ids`` is the set of chunk_ids the LLM was given to
    cite from. If ``None`` (default), correctness only checks that the
    cited chunk is in the relevant set.
    """
    cited = cited_chunk_ids(answer)
    if not cited:
        return 0.0
    valid = candidate_chunk_ids if candidate_chunk_ids is not None else relevant_ids
    if not valid:
        return 0.0
    return len(cited & valid) / len(cited)


def citation_completeness(
    answer: AnswerResponse,
    relevant_ids: set[str],
) -> float:
    """Fraction of relevant chunk_ids that appear in the citations."""
    if not relevant_ids:
        return 0.0
    cited = cited_chunk_ids(answer)
    return len(cited & relevant_ids) / len(relevant_ids)


def citation_precision(
    answer: AnswerResponse,
    relevant_ids: set[str],
) -> float:
    """``|cited ∩ relevant| / |cited|`` (alias of correctness vs relevant set)."""
    cited = cited_chunk_ids(answer)
    if not cited:
        return 0.0
    return len(cited & relevant_ids) / len(cited)


def unsupported_claim_rate(answer: AnswerResponse) -> float:
    """Fraction of validated claims that are UNSUPPORTED.

    A claim with no citations counts as UNSUPPORTED.
    """
    from app.generation.schemas import CitationStatus

    if not answer.claims:
        return 0.0
    unsupported = sum(1 for c in answer.claims if c.status == CitationStatus.UNSUPPORTED)
    return unsupported / len(answer.claims)


# ---------------------------------------------------------------------------
# Aggregate metric set
# ---------------------------------------------------------------------------


@dataclass
class CitationMetricSet:
    """Aggregate citation/grounding metrics for an evaluation slice."""

    grounded_answer_rate: float = 0.0
    refusal_rate: float = 0.0
    refusal_correctness: float = 0.0
    citation_correctness: float = 0.0
    citation_completeness: float = 0.0
    citation_precision: float = 0.0
    unsupported_claim_rate: float = 0.0
    count: int = 0
    per_difficulty: dict[str, CitationMetricSet] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "grounded_answer_rate": self.grounded_answer_rate,
            "refusal_rate": self.refusal_rate,
            "refusal_correctness": self.refusal_correctness,
            "citation_correctness": self.citation_correctness,
            "citation_completeness": self.citation_completeness,
            "citation_precision": self.citation_precision,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "count": self.count,
            "per_difficulty": {
                k: v.to_dict() for k, v in self.per_difficulty.items()
            },
        }


def compute_citation_metrics(
    *,
    question: str,
    answer: str,
    citations: list[dict],
    expected_chunk_ids: list[str],
    answerable: bool,
) -> dict[str, float]:
    """
    Compute citation / grounding metrics for a single answer.

    Parameters
    ----------
    question : the original question (for logging)
    answer : the generated answer string
    citations : list of Citation dicts with ``chunk_id`` keys
    expected_chunk_ids : relevant chunk ids from the golden example
    answerable : True if this is an answerable question

    Returns
    -------
    dict with keys: citation_correctness, citation_completeness,
    citation_precision, grounded_answer_rate, unsupported_claim_rate,
    refusal_correctness
    """
    cited = {str(c["chunk_id"]) for c in citations if c.get("chunk_id")}
    relevant = {str(cid) for cid in expected_chunk_ids}

    if not relevant and not cited:
        return {
            "citation_correctness": 0.0,
            "citation_completeness": 0.0,
            "citation_precision": 0.0,
            "grounded_answer_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "refusal_correctness": 0.0,
        }

    # Citation correctness: fraction of cited chunks that are relevant
    correctness = len(cited & relevant) / len(cited) if cited else 0.0

    # Citation completeness: fraction of relevant chunks that were cited
    completeness = len(cited & relevant) / len(relevant) if relevant else 0.0

    # Citation precision: same as correctness here
    precision = correctness

    # Refusal correctness: correct refusal on unanswerable, correct answer on answerable
    # We can't fully determine this from citations alone, so we infer from whether
    # the answer is non-empty and cites relevant chunks
    refused_correct = (1.0 if not cited and not answer else 0.0) if not answerable else 1.0 if cited else 0.0

    return {
        "citation_correctness": correctness,
        "citation_completeness": completeness,
        "citation_precision": precision,
        "grounded_answer_rate": correctness,  # used as proxy when no claim-level data
        "unsupported_claim_rate": 1.0 - correctness if cited else 0.0,
        "refusal_correctness": refused_correct,
    }


__all__ = [
    "cited_chunk_ids",
    "citation_correctness",
    "citation_completeness",
    "citation_precision",
    "unsupported_claim_rate",
    "CitationMetricSet",
    "compute_citation_metrics",
]
