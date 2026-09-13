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
    document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Restrict retrieval to these document IDs. Use this to ask "
            "questions about a specific uploaded document."
        ),
    )
    anchor_document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Anchor documents for the question (e.g. the user's upload). "
            "Other *uploaded* documents are excluded from retrieval so an "
            "answer anchored on one upload never draws specifics from a "
            "different upload; the shared corpus stays searchable."
        ),
    )
    allow_generic: bool = Field(
        default=False,
        description=(
            "When true, the model may answer from general knowledge where "
            "the evidence is thin instead of refusing. Generic answers are "
            "returned with generic=true and grounded=false."
        ),
    )


class AgentQueryRequest(BaseModel):
    """Request body for POST /query/agent (tool-use path)."""

    question: str = Field(
        min_length=5,
        max_length=2000,
        description="The user's question",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description=(
            "Restrict the agent's search_documents calls to these document IDs. "
            "Enforced server-side on every tool call."
        ),
    )
    top_k: int | None = Field(
        default=5,
        ge=1,
        le=10,
        description="Default fan-out suggested to the agent per search",
    )
    max_steps: int | None = Field(
        default=6,
        ge=1,
        le=10,
        description="Maximum LLM turns in the tool-use loop",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (None uses the configured default)",
    )


class AgentUsage(BaseModel):
    """Tokens spent and estimated cost for one agent run."""

    prompt_tokens: int = Field(default=0, description="Total prompt tokens across turns")
    completion_tokens: int = Field(default=0, description="Total completion tokens across turns")
    total_tokens: int = Field(default=0, description="prompt + completion tokens")
    cost_usd: float | None = Field(
        default=None, description="Estimated USD cost (None when model is unpriced)"
    )


class AgentToolStep(BaseModel):
    """One executed tool call in the agent trace."""

    tool_name: str = Field(description="Tool that was called")
    arguments: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    ok: bool = Field(description="Whether the tool call succeeded")
    output_preview: str = Field(default="", description="First 300 chars of the tool output")
    latency_ms: float = Field(default=0.0, description="Tool execution time")


class AgentQueryResponse(BaseModel):
    """Response for POST /query/agent.

    Same grounding contract as QueryResponse (citations + per-claim
    validation), plus the tool-use trace so callers can see which searches
    and chunk fetches produced the answer.
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
        description="Mean deterministic validation score across claims",
    )
    refused: bool = Field(
        default=False,
        description="True when no relevant evidence was found",
    )
    refused_reason: str | None = Field(default=None, description="Reason for refusal, if refused")
    steps: list[AgentToolStep] = Field(
        default_factory=list,
        description="Tool calls executed, in order",
    )
    evidence_chunks_used: int = Field(default=0, description="Distinct chunks retrieved during the run")
    usage: AgentUsage = Field(default_factory=AgentUsage, description="Tokens spent and estimated cost")
    latency_ms: float = Field(description="Total processing time in milliseconds")
    model: str = Field(description="LLM model used for generation")


class QueryResponseCitation(BaseModel):
    """Serialized citation for the API response."""

    citation_id: str = Field(description="Citation marker, e.g. [C1]")
    chunk_id: str = Field(description="Internal chunk identifier")
    text: str = Field(description="The cited evidence text")
    page_number: int | None = Field(default=None, description="Page number in source")
    section: str | None = Field(default=None, description="Section heading in source")
    source: str | None = Field(
        default=None,
        description="Name of the document this citation came from",
    )


class QueryResponseClaim(BaseModel):
    """Serialized claim for the API response."""

    claim: str = Field(description="The extracted atomic claim text")
    citation_ids: list[str] = Field(
        default_factory=list,
        description="Citations attached to this claim",
    )
    status: CitationStatus = Field(description="Groundedness of the claim")
    reason: str | None = Field(default=None, description="Validation reason")


class QueryResponseUsage(BaseModel):
    """Tokens spent and estimated cost behind an answer."""

    prompt_tokens: int = Field(default=0, description="Prompt tokens spent")
    completion_tokens: int = Field(default=0, description="Completion tokens spent")
    total_tokens: int = Field(default=0, description="prompt + completion tokens")
    cost_usd: float | None = Field(default=None, description="Estimated USD cost (None when unpriced)")


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
    usage: QueryResponseUsage | None = Field(
        default=None,
        description="Tokens spent and estimated cost (None when the LLM was never reached)",
    )
    generic: bool = Field(
        default=False,
        description=(
            "True when the answer may contain general knowledge beyond the "
            "cited evidence (allow_generic mode). Generic answers are never "
            "reported as grounded."
        ),
    )
