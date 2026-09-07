"""API schemas for the query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.generation.schemas import (
    CitationStatus,
    GroundingStatus,
)


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        min_length=5,
        max_length=2000,
        description="The user's question",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Override the number of evidence chunks to retrieve",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (None uses the configured default)",
    )


class QueryResponseCitation(BaseModel):
    """Serialized citation for the API response."""

    citation_id: str = Field(description="Citation marker, e.g. [C1]")
    chunk_id: str = Field(description="Internal chunk identifier")
    text: str = Field(description="The cited evidence text")
    page_number: int | None = Field(default=None, description="Page number in source")
    section: str | None = Field(default=None, description="Section heading in source")


class QueryResponseClaim(BaseModel):
    """Serialized claim for the API response."""

    claim: str = Field(description="The extracted atomic claim text")
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Citations attached to this claim",
    )
    status: CitationStatus = Field(description="Groundedness of the claim")
    reason: str | None = Field(default=None, description="Validation reason")


class QueryResponse(BaseModel):
    """Response for POST /query.

    Exposes the full answer structure including citations, claims,
    grounding status, confidence, refusal information, latency, and model.
    """

    answer: str = Field(description="The generated answer text")
    citations: list[QueryResponseCitation] = Field(
        default_factory=list,
        description="Evidence citations in the answer",
    )
    claims: list[QueryResponseClaim] = Field(
        default_factory=list,
        description="Atomic claims with per-claim grounding status",
    )
    grounded: bool = Field(
        description="True when all claims are fully supported by evidence",
    )
    grounding_status: GroundingStatus = Field(
        description="Aggregate grounding status",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Groundedness confidence score",
    )
    refused: bool = Field(
        default=False,
        description="True when the model refused to answer",
    )
    refused_reason: str | None = Field(
        default=None,
        description="Reason for refusal, if refused",
    )
    latency_ms: float = Field(
        description="Total processing time in milliseconds",
    )
    model: str = Field(description="LLM model used for generation")
