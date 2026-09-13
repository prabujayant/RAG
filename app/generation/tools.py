"""Retrieval tools for the agentic query path (function calling).

Exposes the existing hybrid retrieval + chunk store as OpenAI-compatible
function tools, so an LLM with tool support (e.g. nex-n2.5-mini via
OpenRouter, which advertises ``tools``/``tool_choice``) can gather evidence
itself instead of receiving a fixed evidence block.

Available tools:

- ``search_documents`` — hybrid (vector + BM25) search over indexed chunks,
  optionally restricted to specific document IDs.
- ``fetch_chunk`` — full text of one chunk by ID (search returns snippets so
  the model can pull the complete passage when a claim needs it).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.config.settings import Settings

logger = logging.getLogger(__name__)

# Search returns snippets (not full chunks) so a top-k result set stays
# within a bounded context budget. The model pulls full text via fetch_chunk.
MAX_SNIPPET_CHARS = 600
# Hard cap on a single serialized tool result; the agent loop additionally
# truncates what is appended to the message history.
MAX_TOOL_OUTPUT_CHARS = 6000
# Search fan-out bounds for the tool argument.
MIN_TOP_K = 1
MAX_TOP_K = 10


@dataclass
class ToolContext:
    """Injectable dependencies for tool executors (real by default)."""

    settings: Settings | None = None
    retriever: Any | None = None  # HybridRetriever (lazy so tests can fake it)
    session_factory: Callable[[], Any] | None = None

    def resolved_settings(self) -> Settings:
        return self.settings or get_settings()

    def resolved_retriever(self) -> Any:
        if self.retriever is not None:
            return self.retriever
        from app.retrieval.hybrid import HybridRetriever

        return HybridRetriever(settings=self.resolved_settings())

    def session_scope(self) -> Any:
        """Context manager yielding a SQLAlchemy session."""
        if self.session_factory is not None:
            return self.session_factory()
        from app.db.session import session_scope

        return session_scope()


@dataclass
class ToolCallResult:
    """Outcome of one tool execution."""

    ok: bool
    output: str  # JSON string on success, error message on failure.


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…[truncated]"


def search_documents(
    query: str,
    top_k: int = 5,
    document_ids: list[str] | None = None,
    ctx: ToolContext | None = None,
) -> ToolCallResult:
    """Run hybrid search and return snippet results as a JSON string."""
    context = ctx or ToolContext()
    cleaned = (query or "").strip()
    if not cleaned:
        return ToolCallResult(ok=False, output="search_documents: 'query' must be non-empty.")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return ToolCallResult(ok=False, output="search_documents: 'top_k' must be an integer.")
    top_k = max(MIN_TOP_K, min(MAX_TOP_K, top_k))

    try:
        retriever = context.resolved_retriever()
        hits = retriever.retrieve(
            query=cleaned,
            top_k=top_k,
            filter_document_ids=document_ids,
        )
    except Exception as exc:  # noqa: BLE001 — surface to the model, don't crash the loop
        logger.warning("search_documents failed for %r: %s", cleaned, exc)
        return ToolCallResult(ok=False, output=f"search_documents failed: {exc}")

    results = [
        {
            "chunk_id": h.chunk_id,
            "document_id": h.document_id,
            "score": round(float(h.score), 4),
            "source": h.source,
            "page_number": h.page_number,
            "section": h.section,
            "text": _truncate(h.text or "", MAX_SNIPPET_CHARS),
        }
        for h in (hits or [])
    ]
    payload = json.dumps({"results": results})
    return ToolCallResult(ok=True, output=_truncate(payload, MAX_TOOL_OUTPUT_CHARS))


def fetch_chunk(chunk_id: str, ctx: ToolContext | None = None) -> ToolCallResult:
    """Return the full text of one chunk as a JSON string."""
    context = ctx or ToolContext()
    cleaned = (chunk_id or "").strip()
    if not cleaned:
        return ToolCallResult(ok=False, output="fetch_chunk: 'chunk_id' must be non-empty.")
    try:
        from app.db.models import Chunk as ChunkModel

        with context.session_scope() as session:
            row = session.get(ChunkModel, cleaned)
            if row is None:
                return ToolCallResult(ok=False, output=f"fetch_chunk: no chunk with id {cleaned!r}.")
            payload = json.dumps(
                {
                    "chunk_id": row.id,
                    "document_id": row.document_id,
                    "page_number": row.page_number,
                    "section": row.section,
                    "source": row.source,
                    "text": row.text,
                }
            )
    except Exception as exc:  # noqa: BLE001 — surface to the model, don't crash the loop
        logger.warning("fetch_chunk failed for %r: %s", cleaned, exc)
        return ToolCallResult(ok=False, output=f"fetch_chunk failed: {exc}")
    return ToolCallResult(ok=True, output=_truncate(payload, MAX_TOOL_OUTPUT_CHARS))


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions + dispatcher
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the indexed documents for passages relevant to a query. "
                "Returns up to top_k chunks with snippet text. When the question "
                "is about a specific document, pass its ID in document_ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or a question).",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": MIN_TOP_K,
                        "maximum": MAX_TOP_K,
                        "default": 5,
                        "description": "How many chunks to return.",
                    },
                    "document_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Restrict search to these document IDs.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunk",
            "description": (
                "Fetch the complete text of one chunk by its chunk_id "
                "(chunk_ids come from search_documents results). Use this when "
                "a snippet is cut off and a claim needs the full passage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id from a search result.",
                    },
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_NAMES = frozenset(d["function"]["name"] for d in TOOL_DEFINITIONS)


def execute_tool(
    name: str,
    arguments: dict | None,
    ctx: ToolContext | None = None,
) -> ToolCallResult:
    """Dispatch one tool call by name; never raises (errors become output)."""
    args = arguments or {}
    if name == "search_documents":
        return search_documents(
            query=args.get("query", ""),
            top_k=args.get("top_k", 5),
            document_ids=args.get("document_ids"),
            ctx=ctx,
        )
    if name == "fetch_chunk":
        return fetch_chunk(chunk_id=args.get("chunk_id", ""), ctx=ctx)
    return ToolCallResult(ok=False, output=f"Unknown tool {name!r}. Use one of: {sorted(TOOL_NAMES)}.")


__all__ = [
    "MAX_SNIPPET_CHARS",
    "MAX_TOOL_OUTPUT_CHARS",
    "TOOL_DEFINITIONS",
    "TOOL_NAMES",
    "ToolCallResult",
    "ToolContext",
    "execute_tool",
    "fetch_chunk",
    "search_documents",
]
