"""Query route: retrieval → generation → grounding pipeline."""

from __future__ import annotations

import logging
import os
import re
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.schemas.common import ErrorResponse
from app.api.schemas.query import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentToolStep,
    AgentUsage,
    QueryRequest,
    QueryResponse,
    QueryResponseCitation,
    QueryResponseClaim,
    QueryResponseUsage,
)
from app.config import get_settings
from app.db.models import Document, QueryRecord
from app.db.session import SessionLocal
from app.generation.agent import run_agent
from app.generation.schemas import AnswerResponse, CitationStatus, GroundingStatus
from app.generation.service import GenerationService
from app.grounding import GroundingValidator
from app.grounding.citation_validator import CitationValidator
from app.grounding.claims import ClaimExtractor
from app.observability.metrics import count as count_metric
from app.observability.metrics import record as record_metric
from app.observability.timing import time_operation
from app.retrieval.evidence import EvidenceSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.safety import QuestionVerdict, drop_tainted_chunks, redact_pii, screen_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# Inline evidence markers the agent emits: [chunk_id:<id>] (instructed) or the
# bare [<id>] form models often emit instead. Only markers matching a chunk ID
# actually retrieved in this run are rewritten to [C1]… citations below.
_CHUNK_MARK_RE = re.compile(r"\[(?:chunk_id:)?([^\]\s]+)\]")


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
# NOTE: deliberately a *sync* endpoint. The pipeline below is blocking
# (CPU-bound embedding + synchronous calls to Qdrant/Postgres/OpenRouter).
# Declaring it `async def` while never awaiting anything would run all of that
# on the event loop, freezing the entire server — health checks, page loads and
# concurrent queries would all queue behind a single query. FastAPI runs sync
# endpoints in a worker thread instead.
def query(
    request: QueryRequest,
) -> QueryResponse:
    """Answer a question using hybrid retrieval + grounded generation.

    Pipeline:
    1. Hybrid search (vector + BM25) → top-k evidence chunks, excluding
       other uploads when the question is anchored on specific upload(s)
    2. Cross-encoder reranking → relevance-ordered candidates
    3. Evidence selection → FINAL_CONTEXT_K chunks within MAX_CONTEXT_TOKENS
    4. LLM generation → structured AnswerResponse (generic mode optional)
    5. Claim extraction + grounding validation
    6. Persist QueryRecord to PostgreSQL
    """
    settings = get_settings()
    request_id = str(uuid.uuid4())
    query_id = request_id  # same as request_id for API calls
    stage_ms: dict[str, float] = {}

    # Step 0: safety — screen the question before any retrieval or LLM
    # spend. Refusals return HTTP 200 with refused=True (a policy outcome,
    # not a server error) so the UI renders them like grounding refusals.
    verdict = screen_question(request.question)
    if not verdict.allowed:
        count_metric(f"safety.question_blocked.{verdict.category}")
        logger.warning(
            "Safety refusal for request %s (%s)", request_id, verdict.category
        )
        return _safety_refusal(
            request_id, settings, verdict, "No LLM call was made."
        )

    # Step 1: retrieval
    try:
        retriever = HybridRetriever(settings=settings)
        # Anchored questions (e.g. about a user upload) must not draw
        # specifics from *other* uploads; the shared corpus stays searchable.
        # An explicit document_ids filter is already strict, so exclusion
        # only applies when no such filter is given.
        exclude_ids = (
            _other_upload_ids(request.anchor_document_ids, settings=settings)
            if request.anchor_document_ids and not request.document_ids
            else None
        )
        with time_operation("query.retrieval", log_on_exit=False) as t:
            candidates = retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                filter_document_ids=request.document_ids,
                exclude_document_ids=exclude_ids,
            )
        stage_ms["retrieval"] = t.duration_ms
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

    # Step 2: cross-encoder reranking (non-fatal — falls back to hybrid order).
    # Honors ENABLE_RERANKER / RERANK_TOP_K; the Reranker itself degrades
    # gracefully when disabled or when the model is unavailable.
    try:
        reranker = Reranker(settings=settings)
        with time_operation("query.rerank", log_on_exit=False) as t:
            candidates = reranker.rerank(
                query=request.question,
                candidates=candidates,
            )
        stage_ms["rerank"] = t.duration_ms
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Reranking failed for request %s, using hybrid order: %s",
            request_id,
            exc,
        )

    # Step 3: evidence selection (non-fatal — falls back to reranked list).
    # Enforces FINAL_CONTEXT_K / MAX_CONTEXT_TOKENS so generation always sees
    # a bounded, deduplicated, document-diverse context.
    try:
        selector = EvidenceSelector(settings=settings)
        with time_operation("query.evidence", log_on_exit=False) as t:
            selected = selector.select(candidates)
        stage_ms["evidence"] = t.duration_ms
        if selected:
            candidates = selected
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Evidence selection failed for request %s, using reranked list: %s",
            request_id,
            exc,
        )

    if not candidates:
        # Selection produced nothing (e.g. over-budget) — honest empty answer
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

    # Step 4: generation
    try:
        generation_service = GenerationService(settings=settings)
        with time_operation("query.generation", log_on_exit=False) as t:
            response: AnswerResponse = generation_service.generate(
                question=request.question,
                candidates=candidates,
                temperature=request.temperature,
                allow_generic=request.allow_generic,
            )
        stage_ms["generation"] = t.duration_ms
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

    # Step 5: claim extraction + grounding validation (Phase 7)
    try:
        validator = GroundingValidator(llm_client=generation_service.client)
        with time_operation("query.grounding", log_on_exit=False) as t:
            validated: AnswerResponse = validator.validate(response)
        stage_ms["grounding"] = t.duration_ms
    except Exception as exc:
        logger.warning(
            "Grounding validation failed for request %s, skipping: %s",
            request_id,
            exc,
        )
        # Fail closed: never return an unvalidated response as grounded.
        validated = response
        validated.grounded = False
        validated.grounding_status = GroundingStatus.PARTIALLY_GROUNDED

    # Step 6: persist query record
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

    for stage, ms in stage_ms.items():
        record_metric(f"query.{stage}", ms)
    logger.info(
        "Query complete %s: %d candidates, stages ms: %s",
        request_id,
        len(candidates),
        {k: round(v, 1) for k, v in stage_ms.items()},
    )

    # Step 7: build response
    usage = validated.usage
    return QueryResponse(
        answer=validated.answer,
        citations=[
            QueryResponseCitation(
                citation_id=c.citation_id,
                chunk_id=c.chunk_id,
                text=c.text,
                page_number=c.page_number,
                section=c.section,
                source=c.source,
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
        usage=QueryResponseUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=usage.cost_usd if usage else None,
        )
        if usage
        else None,
        generic=validated.generic,
    )


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------


def _safety_refusal(
    request_id: str,
    settings,
    verdict: QuestionVerdict,
    note: str,
) -> QueryResponse:
    """Build the policy-refusal response shared by the safety gate."""
    try:
        _persist_query_record(
            query_id=request_id,
            question="",
            answer=None,
            grounded=False,
            confidence=None,
            latency_ms=0.0,
            retrieved_chunk_ids=[],
            citation_ids=[],
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal
        logger.warning("Failed to persist safety record for %s: %s", request_id, exc)
    return QueryResponse(
        answer="I can't answer that question.",
        citations=[],
        claims=[],
        grounded=False,
        grounding_status=GroundingStatus.REFUSED,
        confidence=0.0,
        refused=True,
        refused_reason=f"{verdict.reason} {note}",
        latency_ms=0.0,
        model=getattr(settings, "openrouter_model", "unknown"),
        generic=False,
    )


# ---------------------------------------------------------------------------
# POST /query/agent — tool-use (agentic) path
# ---------------------------------------------------------------------------


@router.post(
    "/agent",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Agent or retrieval error"},
    },
)
# Sync endpoint like POST /query: the agent loop is blocking (embedding +
# Qdrant/Postgres/OpenRouter round-trips), so it runs in FastAPI's worker
# threadpool instead of the event loop.
def query_agent(request: AgentQueryRequest) -> AgentQueryResponse:
    """Answer a question with function calling.

    The LLM gathers evidence itself via ``search_documents`` / ``fetch_chunk``
    tools (several turns), then answers. Inline ``[chunk_id:<id>]`` markers
    are rewritten to ``[C1]…`` citations and every extracted claim is
    validated with the existing deterministic citation validator — the same
    grounding contract as POST /query, plus the full tool trace.
    """
    settings = get_settings()
    request_id = str(uuid.uuid4())

    # Safety gate first: no retrieval, no tool loop, no LLM spend on refusal.
    verdict = screen_question(request.question)
    if not verdict.allowed:
        count_metric(f"safety.question_blocked.{verdict.category}")
        logger.warning(
            "Safety refusal for agent request %s (%s)", request_id, verdict.category
        )
        return AgentQueryResponse(
            answer="I can't answer that question.",
            citations=[],
            claims=[],
            grounded=False,
            grounding_status=GroundingStatus.REFUSED,
            confidence=0.0,
            refused=True,
            refused_reason=f"{verdict.reason} No LLM call was made.",
            steps=[],
            evidence_chunks_used=0,
            usage=AgentUsage(),
            latency_ms=0.0,
            model=str(getattr(settings, "openrouter_model", "unknown")),
        )

    try:
        with time_operation("query.agent", log_on_exit=False) as t:
            result = run_agent(
                question=request.question,
                document_ids=request.document_ids,
                top_k=request.top_k or 5,
                max_steps=request.max_steps or 6,
                temperature=request.temperature,
                settings=settings,
            )
        agent_ms = t.duration_ms
    except Exception as exc:
        logger.warning("Agent run failed for request %s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "AGENT_FAILED",
                "message": "The agent failed to run.",
                "request_id": request_id,
            },
        ) from exc

    if result.finish_reason == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "AGENT_FAILED",
                "message": f"The agent failed: {result.error}",
                "request_id": request_id,
            },
        )

    steps = [
        AgentToolStep(
            tool_name=s.tool_name,
            arguments=s.arguments,
            ok=s.ok,
            output_preview=s.output_preview,
            latency_ms=s.latency_ms,
        )
        for s in result.steps
    ]
    usage = AgentUsage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
        cost_usd=result.cost_usd,
    )

    if not result.evidence:
        # Honest refusal: the tools found nothing relevant.
        try:
            _persist_query_record(
                query_id=request_id,
                question=request.question,
                answer=result.answer,
                grounded=False,
                confidence=None,
                latency_ms=result.latency_ms,
                retrieved_chunk_ids=[],
                citation_ids=[],
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            logger.warning("Failed to persist agent record for %s: %s", request_id, exc)
        return AgentQueryResponse(
            answer=result.answer or "I don't have enough information in the provided documents to answer.",
            citations=[],
            claims=[],
            grounded=False,
            grounding_status=GroundingStatus.UNGROUNDED,
            confidence=None,
            refused=True,
            refused_reason="No relevant evidence was retrieved.",
            steps=steps,
            evidence_chunks_used=0,
            usage=usage,
            latency_ms=result.latency_ms,
            model=str(result.model or getattr(settings, "openrouter_model", "unknown")),
        )

    # Safety: drop tool-retrieved evidence carrying injection payloads. The
    # agent loop has no service layer, so this endpoint filters directly.
    clean_evidence, dropped = drop_tainted_chunks(result.evidence)
    if dropped:
        count_metric("safety.chunks_dropped", dropped)
    if not clean_evidence:
        return AgentQueryResponse(
            answer="The retrieved documents contained untrusted instructional "
            "content, so I can't answer from them.",
            citations=[],
            claims=[],
            grounded=False,
            grounding_status=GroundingStatus.REFUSED,
            confidence=0.0,
            refused=True,
            refused_reason="All retrieved evidence was filtered by the safety screen.",
            steps=steps,
            evidence_chunks_used=0,
            usage=usage,
            latency_ms=result.latency_ms,
            model=str(result.model or getattr(settings, "openrouter_model", "unknown")),
        )
    result.evidence = clean_evidence

    # Rewrite [chunk_id:X] markers to [C1]… in first-seen order.
    by_id = {e.chunk_id: e for e in result.evidence}
    ordered_ids: list[str] = []
    for match in _CHUNK_MARK_RE.finditer(result.answer):
        cid = match.group(1)
        if cid in by_id and cid not in ordered_ids:
            ordered_ids.append(cid)
    if not ordered_ids:
        # Model answered without markers — attribute to all retrieved evidence.
        ordered_ids = [e.chunk_id for e in result.evidence]
    canonical = {cid: f"[C{i + 1}]" for i, cid in enumerate(ordered_ids)}
    answer_out = _CHUNK_MARK_RE.sub(lambda m: canonical.get(m.group(1), m.group(0)), result.answer)

    # Safety: mask PII in the final answer. Citation excerpts above are
    # masked the same way, so claim validation below scores the exact text
    # the user sees — consistent with POST /query.
    answer_out, redacted = redact_pii(answer_out)
    if redacted:
        count_metric("safety.pii_redacted", redacted)

    citations = []
    for cid in ordered_ids:
        text, n = redact_pii((by_id[cid].text or "")[:2000])
        if n:
            count_metric("safety.pii_redacted", n)
        citations.append(
            QueryResponseCitation(
                citation_id=canonical[cid],
                chunk_id=cid,
                text=text,
                page_number=by_id[cid].page_number,
                section=by_id[cid].section,
                source=by_id[cid].source,
            )
        )
    citation_text = {c.citation_id: c.text for c in citations}

    # Claim-split + deterministic validation against the cited chunks.
    validator = CitationValidator()
    claims: list[QueryResponseClaim] = []
    scores: list[float] = []
    for claim in ClaimExtractor().extract(answer_out):
        linked = [c for c in claim.citation_ids if c in citation_text] or list(citation_text)
        evidence_text = "\n\n".join(citation_text[c] for c in linked)
        claim_verdict = validator.validate(claim.text, evidence_text, linked[0] if linked else None)
        scores.append(claim_verdict.score)
        claims.append(
            QueryResponseClaim(
                claim=claim.text,
                citation_ids=linked,
                status=claim_verdict.status,
                reason=claim_verdict.reason,
            )
        )

    if not claims:
        grounded, status_ = False, GroundingStatus.UNGROUNDED
    elif all(c.status == CitationStatus.SUPPORTED for c in claims):
        grounded, status_ = True, GroundingStatus.GROUNDED
    elif any(c.status == CitationStatus.SUPPORTED for c in claims):
        grounded, status_ = False, GroundingStatus.PARTIALLY_GROUNDED
    else:
        grounded, status_ = False, GroundingStatus.UNGROUNDED
    confidence = round(sum(scores) / len(scores), 3) if scores else None

    try:
        _persist_query_record(
            query_id=request_id,
            question=request.question,
            answer=answer_out,
            grounded=grounded,
            confidence=confidence,
            latency_ms=result.latency_ms,
            retrieved_chunk_ids=[e.chunk_id for e in result.evidence],
            citation_ids=[c.citation_id for c in citations],
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal
        logger.warning("Failed to persist agent record for %s: %s", request_id, exc)

    record_metric("query.agent", agent_ms)
    logger.info(
        "Agent query complete %s: %d steps, %d evidence, grounded=%s, tokens=%d+%d, cost_usd=%s",
        request_id,
        len(steps),
        len(result.evidence),
        grounded,
        result.prompt_tokens,
        result.completion_tokens,
        result.cost_usd,
    )
    return AgentQueryResponse(
        answer=answer_out,
        citations=citations,
        claims=claims,
        grounded=grounded,
        grounding_status=status_,
        confidence=confidence,
        refused=False,
        refused_reason=None,
        steps=steps,
        evidence_chunks_used=len(result.evidence),
        usage=usage,
        latency_ms=result.latency_ms,
        model=str(result.model or getattr(settings, "openrouter_model", "unknown")),
    )


# ---------------------------------------------------------------------------
# POST /query/stream — SSE progress streaming
# ---------------------------------------------------------------------------


@router.post("/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Stream RAG progress as Server-Sent Events.

    Emits ``retrieval → reranking → evidence → generation → grounding → done``
    events so the UI renders progress instantly instead of blocking on the
    LLM round-trips. The ``done`` event carries the full QueryResponse JSON.
    Reuses the exact pipeline as POST /query (no logic fork).
    """
    import asyncio
    import contextlib
    import json

    async def _events_inner():
        settings = get_settings()
        request_id = str(uuid.uuid4())
        yield f'event: started\ndata: {{"request_id": "{request_id}"}}\n\n'

        # Safety gate first (mirrors POST /query): refuse without any
        # retrieval or LLM spend, emitting the full response shape.
        verdict = screen_question(request.question)
        if not verdict.allowed:
            count_metric(f"safety.question_blocked.{verdict.category}")
            logger.warning(
                "Safety refusal for stream request %s (%s)", request_id, verdict.category
            )
            payload = json.dumps(
                {
                    "answer": "I can't answer that question.",
                    "citations": [],
                    "claims": [],
                    "grounded": False,
                    "grounding_status": GroundingStatus.REFUSED.value,
                    "confidence": 0.0,
                    "refused": True,
                    "refused_reason": f"{verdict.reason} No LLM call was made.",
                    "generic": False,
                    "latency_ms": 0.0,
                    "model": getattr(settings, "openrouter_model", "unknown"),
                    "usage": None,
                }
            )
            yield f"event: done\ndata: {payload}\n\n"
            return

        # Retrieval (blocking — offload to thread)
        yield 'event: retrieval\ndata: {"stage": "hybrid_search"}\n\n'
        try:
            retriever = HybridRetriever(settings=settings)
            exclude_ids = (
                _other_upload_ids(request.anchor_document_ids, settings=settings)
                if request.anchor_document_ids and not request.document_ids
                else None
            )
            candidates = await asyncio.to_thread(
                retriever.retrieve,
                request.question,
                request.top_k,
                request.document_ids,
                exclude_ids,
            )
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'stage': 'retrieval', 'error': str(exc)})}\n\n"
            return
        yield f"event: retrieved\ndata: {json.dumps({'candidates': len(candidates)})}\n\n"
        if not candidates:
            # Emit the full response shape: the UI resolves status/badges from
            # these fields, and a partial payload renders as a broken answer.
            payload = json.dumps(
                {
                    "answer": "No relevant documents were found for your question.",
                    "citations": [],
                    "claims": [],
                    "grounded": False,
                    "grounding_status": GroundingStatus.UNGROUNDED.value,
                    "confidence": None,
                    "refused": False,
                    "refused_reason": None,
                    "generic": False,
                    "latency_ms": 0.0,
                    "model": getattr(settings, "openrouter_model", "unknown"),
                    "usage": None,
                }
            )
            yield f"event: done\ndata: {payload}\n\n"
            return

        # Rerank + evidence (non-fatal, same as /query)
        yield "event: reranking\ndata: {}\n\n"
        with contextlib.suppress(Exception):
            candidates = await asyncio.to_thread(
                Reranker(settings=settings).rerank, request.question, candidates
            )
        with contextlib.suppress(Exception):
            selected = await asyncio.to_thread(EvidenceSelector(settings=settings).select, candidates)
            if selected:
                candidates = selected
        yield f"event: evidence\ndata: {json.dumps({'candidates': len(candidates)})}\n\n"

        # Generation
        yield "event: generation\ndata: {}\n\n"
        try:
            service = GenerationService(settings=settings)
            response: AnswerResponse = await asyncio.to_thread(
                service.generate,
                request.question,
                candidates,
                temperature=request.temperature,
                allow_generic=request.allow_generic,
            )
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'stage': 'generation', 'error': str(exc)})}\n\n"
            return

        # Grounding
        yield "event: grounding\ndata: {}\n\n"
        try:
            validated = await asyncio.to_thread(
                GroundingValidator(llm_client=service.client).validate, response
            )
        except Exception:
            validated = response
            validated.grounded = False
            validated.grounding_status = GroundingStatus.PARTIALLY_GROUNDED

        final = validated.to_dict()
        # The API/UI contract calls this `latency_ms`; the domain dataclass
        # names it `total_latency_ms`. `to_dict()` already renders enums as
        # their string values (e.g. GroundingStatus.UNGROUNDED -> "ungrounded"),
        # which `str(enum)` would NOT do.
        final["latency_ms"] = final.pop("total_latency_ms", None) or 0.0
        final["model"] = final.get("model") or "unknown"
        # Persist like POST /query (non-fatal) so analytics don't diverge.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                _persist_query_record,
                request_id,
                request.question,
                validated.answer,
                validated.grounded,
                validated.confidence,
                validated.total_latency_ms,
                [c.chunk_id for c in candidates],
                [c.citation_id for c in validated.citations],
            )
        yield f"event: done\ndata: {json.dumps(final)}\n\n"

    async def _events():
        """Guarantee the client always receives either ``done`` or ``error``.

        An exception raised after the last ``yield`` (e.g. in response
        serialization) otherwise terminates the SSE response silently: the
        browser sees the stream close with no ``done`` event and surfaces a
        generic "stream ended without a result", hiding the real cause.
        """
        try:
            async for event in _events_inner():
                yield event
        except Exception as exc:  # noqa: BLE001 — surface, never die silently
            logger.exception("Query stream failed")
            yield f"event: error\ndata: {json.dumps({'stage': 'stream', 'error': str(exc)})}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


def _other_upload_ids(anchor_ids: list[str] | None, settings=None) -> list[str] | None:
    """Return uploaded document IDs other than *anchor_ids*.

    Uploads are persisted under the configured upload directory, so any
    document sourced from there (except the anchors) is another upload whose
    specifics must not leak into an anchored answer. Returns None when there
    is nothing to exclude so retrievers skip the exclusion clause entirely.
    """
    if not anchor_ids:
        return None
    settings = settings or get_settings()
    upload_root = os.path.abspath(settings.upload_dir)
    prefix = upload_root + os.sep
    anchors = set(anchor_ids)
    with SessionLocal() as session:
        rows = session.execute(select(Document.document_id, Document.source)).all()
    excluded = [
        doc_id
        for doc_id, source in rows
        if doc_id not in anchors and isinstance(source, str) and os.path.abspath(source).startswith(prefix)
    ]
    return excluded or None


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
    with SessionLocal() as session:
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
