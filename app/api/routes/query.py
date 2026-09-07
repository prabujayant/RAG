"""Query route: retrieval → generation → grounding pipeline."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.common import ErrorResponse
from app.api.schemas.query import QueryRequest, QueryResponse, QueryResponseCitation, QueryResponseClaim
from app.config import get_settings
from app.db.models import QueryRecord
from app.db.session import SessionLocal
from app.generation.schemas import AnswerResponse, GroundingStatus
from app.generation.service import GenerationService
from app.grounding import GroundingValidator
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Generation or retrieval error"},
    },
)
async def query(
    request: QueryRequest,
) -> QueryResponse:
    """Answer a question using hybrid retrieval + grounded generation.

    Pipeline:
    1. Hybrid search (vector + BM25) → top-k evidence chunks
    2. LLM generation → structured AnswerResponse
    3. Claim extraction + grounding validation
    4. Persist QueryRecord to PostgreSQL
    """
    settings = get_settings()
    request_id = str(uuid.uuid4())
    query_id = request_id  # same as request_id for API calls

    # Step 1: retrieval
    try:
        retriever = HybridRetriever(settings=settings)
        candidates = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.warning("Retrieval failed for request %s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "RETRIEVAL_FAILED",
                "message": "Failed to retrieve evidence for your question.",
                "request_id": request_id,
            },
        ) from exc

    if not candidates:
        # No evidence found — return an honest empty answer
        return QueryResponse(
            answer="No relevant documents were found for your question.",
            citations=[],
            claims=[],
            grounded=False,
            grounding_status=GroundingStatus.UNGROUNDED,
            confidence=None,
            refused=False,
            refused_reason=None,
            latency_ms=0.0,
            model=getattr(settings, "openrouter_model", "unknown"),
        )

    # Step 2: generation
    try:
        generation_service = GenerationService(settings=settings)
        response: AnswerResponse = generation_service.generate(
            question=request.question,
            candidates=candidates,
            temperature=request.temperature,
        )
    except Exception as exc:
        logger.warning("Generation failed for request %s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "GENERATION_FAILED",
                "message": "Failed to generate an answer.",
                "request_id": request_id,
            },
        ) from exc

    # Step 3: claim extraction + grounding validation (Phase 7)
    try:
        validator = GroundingValidator()
        validated: AnswerResponse = validator.validate(response)
    except Exception as exc:
        logger.warning(
            "Grounding validation failed for request %s, skipping: %s",
            request_id,
            exc,
        )
        validated = response  # fall back to unvalidated response

    # Step 4: persist query record
    try:
        _persist_query_record(
            query_id=query_id,
            question=request.question,
            answer=validated.answer,
            grounded=validated.grounded,
            confidence=validated.confidence,
            latency_ms=validated.total_latency_ms,
            retrieved_chunk_ids=[c.chunk_id for c in candidates],
            citation_ids=[c.citation_id for c in validated.citations],
        )
    except Exception as exc:
        logger.warning("Failed to persist QueryRecord for %s: %s", request_id, exc)
        # Non-fatal — do not fail the request

    # Step 5: build response
    return QueryResponse(
        answer=validated.answer,
        citations=[
            QueryResponseCitation(
                citation_id=c.citation_id,
                chunk_id=c.chunk_id,
                text=c.text,
                page_number=c.page_number,
                section=c.section,
            )
            for c in validated.citations
        ],
        claims=[
            QueryResponseClaim(
                claim=cl.claim,
                citation_ids=list(cl.citation_ids),
                status=cl.status,
                reason=cl.reason,
            )
            for cl in validated.claims
        ],
        grounded=validated.grounded,
        grounding_status=validated.grounding_status,
        confidence=validated.confidence,
        refused=validated.refused,
        refused_reason=validated.refused_reason,
        latency_ms=validated.total_latency_ms or 0.0,
        model=validated.model or "unknown",
    )


def _persist_query_record(
    query_id: str,
    question: str,
    answer: str | None,
    grounded: bool | None,
    confidence: float | None,
    latency_ms: float | None,
    retrieved_chunk_ids: list[str],
    citation_ids: list[str],
) -> None:
    """Persist a QueryRecord to PostgreSQL (non-fatal on failure)."""
    session = SessionLocal()
    try:
        record = QueryRecord(
            id=query_id,
            query_id=query_id,
            question=question,
            answer=answer,
            grounded=grounded,
            confidence=confidence,
            total_latency_ms=latency_ms,
            retrieved_chunk_ids=retrieved_chunk_ids,
            citation_ids=citation_ids,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()
