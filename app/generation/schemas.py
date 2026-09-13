"""Generation domain schemas — LLM input/output models.

These are pure domain types. The OpenAPI layer uses its own request/response
models defined in ``app.api.schemas``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CitationStatus(StrEnum):
    """Whether a citation actually supports its associated claim."""

    SUPPORTED = "supported"  # The chunk fully supports the claim.
    PARTIALLY_SUPPORTED = "partially_supported"  # Partial overlap.
    UNSUPPORTED = "unsupported"  # The chunk does not support the claim.


class GroundingStatus(StrEnum):
    """Aggregate grounding result for an entire answer."""

    GROUNDED = "grounded"  # Every claim is supported.
    PARTIALLY_GROUNDED = "partially_grounded"  # At least one claim unsupported.
    UNGROUNDED = "ungrounded"  # No claims are supported / answer is wrong.
    REFUSED = "refused"  # Evidence was insufficient; answer refused.


@dataclass
class Citation:
    """A single citation embedded in the generated answer.

    Attributes
    ----------
    citation_id:
        Unique citation marker, e.g. ``[C1]``, ``[C2]``.
    chunk_id:
        The source chunk this citation points to.
    text:
        The exact text span cited from the chunk.
    page_number:
        Page number in source, if known.
    section:
        Section heading in source, if known.
    source:
        Human-readable name of the document the citation came from. Surfaced so
        the UI can show which file an answer was drawn from — without it, an
        answer grounded in a *different* document is indistinguishable from one
        grounded in the expected file.
    """

    citation_id: str
    chunk_id: str
    text: str
    page_number: int | None = None
    section: str | None = None
    source: str | None = None

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "section": self.section,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Citation:
        return cls(
            citation_id=data["citation_id"],
            chunk_id=data["chunk_id"],
            text=data["text"],
            page_number=data.get("page_number"),
            section=data.get("section"),
            source=data.get("source"),
        )


@dataclass
class AnswerClaim:
    """A single atomic claim extracted from a generated answer.

    Attributes
    ----------
    claim:
        The textual claim, e.g. ``"Access tokens expire after 60 minutes."``.
    citation_ids:
        List of ``citation_id`` values this claim was derived from.
    status:
        Whether the evidence actually supports the claim.
    reason:
        Short human-readable explanation when status is not SUPPORTED.
    """

    claim: str
    citation_ids: list[str] = field(default_factory=list)
    status: CitationStatus = CitationStatus.UNSUPPORTED
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "citation_ids": list(self.citation_ids),
            "status": self.status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnswerClaim:
        return cls(
            claim=data["claim"],
            citation_ids=list(data.get("citation_ids", [])),
            status=CitationStatus(data.get("status", CitationStatus.UNSUPPORTED.value)),
            reason=data.get("reason"),
        )


@dataclass
class TokenUsage:
    """Tokens spent and estimated cost for the LLM calls behind an answer.

    Attributes
    ----------
    prompt_tokens / completion_tokens:
        Totals across all calls (generation; agent turns accumulate).
    total_tokens:
        Convenience sum (prompt + completion).
    cost_usd:
        Estimated USD cost, or None when the model is unpriced.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> TokenUsage | None:
        if not data:
            return None
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
            cost_usd=data.get("cost_usd"),
        )


@dataclass
class AnswerResponse:
    """The full answer produced by the LLM, with grounding metadata.

    Attributes
    ----------
    answer:
        The generated answer text. May contain citation markers like ``[C1]``.
    citations:
        All citations extracted/identified in the answer.
    claims:
        Each atomic claim paired with its validation status.
    grounded:
        Shortcut: ``True`` iff ``grounding_status`` is GROUNDED.
    grounding_status:
        Aggregate validation result.
    confidence:
        LLM-reported confidence score in [0, 1]; may be None if model doesn't
        produce one.
    refused:
        Whether the LLM refused to answer due to insufficient evidence.
    refused_reason:
        The LLM's stated reason for refusal, if applicable.
    generic:
        Whether the answer may contain general knowledge beyond the cited
        evidence (generic mode). Generic answers are never grounded.
    total_latency_ms:
        End-to-end latency for the generation call in milliseconds.
    model:
        Which model was used to generate this answer.
    usage:
        Tokens spent and estimated cost behind this answer (None when the
        LLM was never reached, e.g. retrieval found nothing).
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    claims: list[AnswerClaim] = field(default_factory=list)
    grounded: bool = False
    grounding_status: GroundingStatus = GroundingStatus.UNGROUNDED
    confidence: float | None = None
    refused: bool = False
    refused_reason: str | None = None
    generic: bool = False
    total_latency_ms: float | None = None
    model: str | None = None
    usage: TokenUsage | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "claims": [c.to_dict() for c in self.claims],
            "grounded": self.grounded,
            "grounding_status": self.grounding_status.value,
            "confidence": self.confidence,
            "refused": self.refused,
            "refused_reason": self.refused_reason,
            "generic": self.generic,
            "total_latency_ms": self.total_latency_ms,
            "model": self.model,
            "usage": self.usage.to_dict() if self.usage else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnswerResponse:
        return cls(
            answer=data["answer"],
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            claims=[AnswerClaim.from_dict(c) for c in data.get("claims", [])],
            grounded=data.get("grounded", False),
            grounding_status=GroundingStatus(data.get("grounding_status", GroundingStatus.UNGROUNDED.value)),
            confidence=data.get("confidence"),
            refused=data.get("refused", False),
            refused_reason=data.get("refused_reason"),
            generic=data.get("generic", False),
            total_latency_ms=data.get("total_latency_ms"),
            model=data.get("model"),
            usage=TokenUsage.from_dict(data.get("usage")),
        )
