"""Regression tests for the SSE ``/query/stream`` payload.

Guards a bug where the stream built its ``done`` payload with Pydantic's
``model_dump()`` against *dataclass* schemas (``AttributeError``) and rendered
enums with ``str()`` (``"GroundingStatus.UNGROUNDED"`` instead of
``"ungrounded"``). The exception killed the generator after the ``grounding``
event, so the browser never received ``done`` and surfaced the confusing
"Stream ended without a result" error.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from app.generation.schemas import (
    AnswerClaim,
    AnswerResponse,
    Citation,
    CitationStatus,
    GroundingStatus,
)
from app.retrieval.models import RetrievalResult, RetrieverType


def _candidate() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="API keys are single-secret; rotating invalidates the previous key.",
        score=0.9,
        source="api-keys.md",
        retriever=RetrieverType.VECTOR,
        rank=0,
    )


def _answer() -> AnswerResponse:
    return AnswerResponse(
        answer="Rotating an API key invalidates the previous key. [C1]",
        citations=[
            Citation(
                citation_id="[C1]",
                chunk_id="chunk-1",
                text="Rotating an API key invalidates the previous key.",
                source="api-keys.md",
            )
        ],
        claims=[
            AnswerClaim(
                claim="Rotating invalidates the previous key.",
                citation_ids=["[C1]"],
                status=CitationStatus.SUPPORTED,
            )
        ],
        grounded=False,
        grounding_status=GroundingStatus.UNGROUNDED,
        confidence=0.42,
        total_latency_ms=12.5,
        model="test/model",
    )


async def _collect(response) -> list[str]:
    """Drain the streaming response, normalizing chunks to text."""
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return chunks


def _parse_events(chunks: list[str]) -> list[tuple[str, str | None]]:
    """Parse SSE frames into (event, data) pairs."""
    events: list[tuple[str, str | None]] = []
    for raw in "".join(chunks).split("\n\n"):
        if not raw.strip():
            continue
        name: str | None = None
        data: str | None = None
        for line in raw.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if name:
            events.append((name, data))
    return events


@pytest.mark.unit
def test_answer_response_to_dict_is_json_serializable() -> None:
    """The domain schemas must serialize to JSON with string enum values."""
    payload = _answer().to_dict()

    json.dumps(payload)  # must not raise (no model_dump / enum objects)

    assert payload["grounding_status"] == "ungrounded"
    assert payload["claims"][0]["status"] == "supported"
    assert payload["citations"][0]["citation_id"] == "[C1]"


@pytest.mark.unit
def test_stream_emits_done_with_full_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stream must always terminate with a complete ``done`` payload."""
    from app.api.routes import query as q
    from app.api.schemas.query import QueryRequest

    candidate = _candidate()
    answer = _answer()

    class _Retriever:
        def __init__(self, *args, **kwargs) -> None: ...
        def retrieve(self, *args, **kwargs) -> list[RetrievalResult]:
            return [candidate]

    class _Reranker:
        def __init__(self, *args, **kwargs) -> None: ...
        def rerank(self, *args, **kwargs) -> list[RetrievalResult]:
            return [candidate]

    class _Selector:
        def __init__(self, *args, **kwargs) -> None: ...
        def select(self, *args, **kwargs) -> list[RetrievalResult]:
            return [candidate]

    class _Service:
        def __init__(self, *args, **kwargs) -> None:
            self.client = object()

        def generate(self, *args, **kwargs) -> AnswerResponse:
            return answer

    class _Validator:
        def __init__(self, *args, **kwargs) -> None: ...
        def validate(self, *args, **kwargs) -> AnswerResponse:
            return answer

    monkeypatch.setattr(q, "HybridRetriever", _Retriever)
    monkeypatch.setattr(q, "Reranker", _Reranker)
    monkeypatch.setattr(q, "EvidenceSelector", _Selector)
    monkeypatch.setattr(q, "GenerationService", _Service)
    monkeypatch.setattr(q, "GroundingValidator", _Validator)
    # Persistence is positionally invoked and best-effort; keep the test hermetic.
    monkeypatch.setattr(q, "_persist_query_record", lambda *a, **k: None)

    response = q.query_stream(QueryRequest(question="how do i rotate api keys?", top_k=5))
    events = _parse_events(asyncio.run(_collect(response)))

    names = [name for name, _ in events]
    assert "error" not in names, f"unexpected error event: {events}"
    assert names[-1] == "done"
    assert "grounding" in names

    done = json.loads(dict(events)["done"])
    assert done["answer"] == answer.answer
    assert done["grounding_status"] == "ungrounded"
    assert done["refused"] is False
    assert done["generic"] is False
    assert done["latency_ms"] == 12.5
    assert done["model"] == "test/model"
    assert done["citations"][0]["citation_id"] == "[C1]"
    assert done["claims"][0]["status"] == "supported"


@pytest.mark.unit
def test_stream_reports_error_instead_of_dying_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline failure must yield an ``error`` event, not end the stream."""
    from app.api.routes import query as q
    from app.api.schemas.query import QueryRequest

    class _Boom:
        def __init__(self, *args, **kwargs) -> None: ...
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("backend exploded")

    monkeypatch.setattr(q, "HybridRetriever", _Boom)

    response = q.query_stream(QueryRequest(question="how do i rotate api keys?", top_k=5))
    events = _parse_events(asyncio.run(_collect(response)))

    names = [name for name, _ in events]
    assert "error" in names
    assert "done" not in names
    assert "backend exploded" in (dict(events)["error"] or "")
