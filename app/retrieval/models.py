"""Retrieval domain models.

These types are shared across all retrievers (vector, BM25, hybrid) so the
rest of the application deals in a single, stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RetrieverType(StrEnum):
    """Which retriever produced a result."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """A single chunk returned by any retriever.

    Attributes
    ----------
    chunk_id:
        Deterministic id of the form ``{document_id}:{chunk_index}``.
    document_id:
        Stable document identifier.
    text:
        Raw text content of the chunk.
    score:
        Relevance score from the retriever. Interpretation is retriever-specific
        (cosine similarity for vector, BM25 score for BM25, RRF score for hybrid).
    source:
        File path or URL the chunk originated from.
    page_number:
        Page number in the source document, if available.
    section:
        Heading/section name this chunk belongs to, if detected.
    retriever:
        Which retriever produced this result.
    rank:
        0-based rank within this retriever's result list (useful for fusion).
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    page_number: int | None = None
    section: str | None = None
    retriever: RetrieverType = RetrieverType.VECTOR
    rank: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "page_number": self.page_number,
            "section": self.section,
            "retriever": self.retriever.value,
            "rank": self.rank,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RetrievalResult:
        """Reconstruct a result from :meth:`to_dict` output (round-trip)."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            text=data["text"],
            score=data["score"],
            source=data["source"],
            page_number=data.get("page_number"),
            section=data.get("section"),
            retriever=RetrieverType(data.get("retriever", RetrieverType.VECTOR.value)),
            rank=data.get("rank", 0),
            metadata=data.get("metadata", {}),
        )
