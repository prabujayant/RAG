"""Unit tests for the agentic tool-use path (no network, no model weights)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from app.generation.agent import (
    _evidence_from_search_output,
    run_agent,
)
from app.generation.client import LLMUsage, ToolChatTurn
from app.generation.tools import (
    MAX_SNIPPET_CHARS,
    TOOL_DEFINITIONS,
    TOOL_NAMES,
    ToolContext,
    execute_tool,
    fetch_chunk,
    search_documents,
)


def _hit(chunk_id="chunk-1", text="Evidence text about propellant.", **kwargs):
    params = {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "text": text,
        "score": 0.9,
        "source": "manual.pdf",
        "page_number": 3,
        "section": "Propulsion",
    }
    params.update(kwargs)
    return SimpleNamespace(**params)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, top_k=None, filter_document_ids=None, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, "filter_document_ids": filter_document_ids})
        return self.hits


class FakeChatClient:
    """Scripted chat client: pops one assistant message per turn."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat_with_tools(self, *, messages, tools, tool_choice="auto", max_output_tokens, temperature=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.script:
            return ToolChatTurn(message={"content": "done", "tool_calls": []}, usage=LLMUsage())
        return ToolChatTurn(
            message=self.script.pop(0),
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.01),
        )


def _tool_call(name, arguments, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tool_definitions_are_valid_function_schemas():
    assert {"search_documents", "fetch_chunk"} == TOOL_NAMES
    for tool in TOOL_DEFINITIONS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] in TOOL_NAMES
        assert fn["parameters"]["type"] == "object"
        assert "query" in fn["parameters"]["properties"] or fn["name"] == "fetch_chunk"


@pytest.mark.unit
def test_execute_unknown_tool_returns_error():
    result = execute_tool("fly_rocket", {})
    assert result.ok is False
    assert "Unknown tool" in result.output


@pytest.mark.unit
def test_search_documents_returns_snippets_and_records_filter():
    long_text = "x" * (MAX_SNIPPET_CHARS + 100)
    retriever = FakeRetriever([_hit(text=long_text)])
    ctx = ToolContext(retriever=retriever)
    result = search_documents("propellant", top_k=3, document_ids=["doc-1"], ctx=ctx)
    assert result.ok is True
    payload = json.loads(result.output)
    assert len(payload["results"]) == 1
    assert len(payload["results"][0]["text"]) <= MAX_SNIPPET_CHARS + len("…[truncated]")
    assert retriever.calls[0]["filter_document_ids"] == ["doc-1"]
    assert retriever.calls[0]["top_k"] == 3


@pytest.mark.unit
def test_search_documents_rejects_empty_query_and_clamps_top_k():
    ctx = ToolContext(retriever=FakeRetriever([]))
    assert search_documents("  ", ctx=ctx).ok is False
    result = search_documents("q", top_k=999, ctx=ctx)
    assert result.ok is True
    assert ctx.retriever.calls[0]["top_k"] == 10


@pytest.mark.unit
def test_search_documents_failure_surfaces_as_error_output():
    class Boom:
        def retrieve(self, *a, **k):
            raise RuntimeError("qdrant down")

    result = search_documents("q", ctx=ToolContext(retriever=Boom()))
    assert result.ok is False
    assert "qdrant down" in result.output


@pytest.mark.unit
def test_fetch_chunk_found_and_missing():
    row = SimpleNamespace(
        id="chunk-9",
        document_id="doc-2",
        page_number=1,
        section="S",
        source="f.pdf",
        text="full passage",
    )

    @contextmanager
    def factory(hit=True):
        class Session:
            def get(self, model, chunk_id):
                assert chunk_id == "chunk-9"
                return row if hit else None

        yield Session()

    ok_result = fetch_chunk("chunk-9", ctx=ToolContext(session_factory=lambda: factory(True)))
    assert ok_result.ok is True
    assert json.loads(ok_result.output)["text"] == "full passage"

    missing = fetch_chunk("chunk-9", ctx=ToolContext(session_factory=lambda: factory(False)))
    assert missing.ok is False
    assert "no chunk" in missing.output

    assert fetch_chunk("  ").ok is False


@pytest.mark.unit
def test_evidence_parsing_helpers():
    assert _evidence_from_search_output("not json") == []
    good = json.dumps({"results": [{"chunk_id": "c1", "text": "t"}]})
    parsed = _evidence_from_search_output(good)
    assert [e.chunk_id for e in parsed] == ["c1"]


# ---------------------------------------------------------------------------
# Agent loop (scripted client, fake retriever — no network)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_agent_loop_calls_tool_then_answers():
    retriever = FakeRetriever([_hit(chunk_id="chunk-1", text="Propellant load is 1200t.")])
    client = FakeChatClient(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("search_documents", {"query": "propellant load"})],
            },
            {"content": "Load is 1200t [chunk_id:chunk-1]", "tool_calls": []},
        ]
    )
    result = run_agent(
        "What is the propellant load?",
        client=client,
        tools_ctx=ToolContext(retriever=retriever),
        max_steps=4,
    )
    assert result.finish_reason == "answered"
    assert result.answer == "Load is 1200t [chunk_id:chunk-1]"
    assert [e.chunk_id for e in result.evidence] == ["chunk-1"]
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "search_documents"
    assert result.steps[0].ok is True
    # Usage accumulates across turns (2 turns × 100/50/0.01).
    assert result.prompt_tokens == 200
    assert result.completion_tokens == 100
    assert result.cost_usd == pytest.approx(0.02)
    # Scope enforced server-side even when the model omits document_ids.
    assert retriever.calls[0]["filter_document_ids"] is None


@pytest.mark.unit
def test_agent_enforces_document_scope():
    retriever = FakeRetriever([_hit()])
    client = FakeChatClient(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("search_documents", {"query": "q"})],
            },
            {"content": "answer [chunk_id:chunk-1]", "tool_calls": []},
        ]
    )
    run_agent(
        "q?",
        document_ids=["doc-9"],
        client=client,
        tools_ctx=ToolContext(retriever=retriever),
    )
    assert retriever.calls[0]["filter_document_ids"] == ["doc-9"]


@pytest.mark.unit
def test_agent_max_steps_guard():
    retriever = FakeRetriever([_hit()])
    script = [
        {
            "content": "",
            "tool_calls": [_tool_call("search_documents", {"query": f"q{i}"}, call_id=f"c{i}")],
        }
        for i in range(10)
    ]
    client = FakeChatClient(script)
    result = run_agent(
        "q?",
        client=client,
        tools_ctx=ToolContext(retriever=retriever),
        max_steps=2,
    )
    assert result.finish_reason == "max_steps"
    assert len(result.steps) == 2


@pytest.mark.unit
def test_agent_no_evidence_refusal():
    client = FakeChatClient([{"content": "", "tool_calls": []}])
    result = run_agent("q?", client=client, tools_ctx=ToolContext(retriever=FakeRetriever([])))
    assert result.finish_reason == "no_evidence"
    assert result.evidence == []


@pytest.mark.unit
def test_agent_tool_error_is_fed_back_not_raised():
    class Boom:
        def retrieve(self, *a, **k):
            raise RuntimeError("nope")

    client = FakeChatClient(
        [
            {
                "content": "",
                "tool_calls": [_tool_call("search_documents", {"query": "q"})],
            },
            {"content": "I don't have enough information.", "tool_calls": []},
        ]
    )
    result = run_agent("q?", client=client, tools_ctx=ToolContext(retriever=Boom()))
    assert result.finish_reason in {"answered", "no_evidence"}
    assert result.steps[0].ok is False
    # The error text goes back into the conversation for the next turn.
    tool_messages = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages and "nope" in tool_messages[0]["content"]
