"""Unit tests for LLM cost tracking (no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from app.generation.client import OpenRouterClient
from app.observability.cost import (
    estimate_cost_usd,
    normalize_model_id,
    record_llm_usage,
    reset_usage,
    usage_summary,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_usage()
    yield
    reset_usage()


@pytest.mark.unit
def test_normalize_model_id():
    assert normalize_model_id("nex-agi/nex-n2.5-mini:free") == "nex-agi/nex-n2.5-mini"
    assert normalize_model_id("OpenAI/GPT-4o-Mini ") == "openai/gpt-4o-mini"
    assert normalize_model_id(None) == ""
    assert normalize_model_id("") == ""


@pytest.mark.unit
def test_estimate_cost_math():
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out → 1M+1M = $0.75.
    assert estimate_cost_usd("openai/gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    # Free tier costs nothing but is still priced (not unknown).
    assert estimate_cost_usd("nex-agi/nex-n2.5-mini:free", 5000, 2000) == 0.0
    # Unknown model or missing counts → None (unpriced, never invented).
    assert estimate_cost_usd("someone/mystery-model", 100, 100) is None
    assert estimate_cost_usd("openai/gpt-4o-mini", None, 100) is None


@pytest.mark.unit
def test_record_aggregation_and_summary():
    record_llm_usage("generation", "openai/gpt-4o-mini", 1_000_000, 0)
    record_llm_usage("generation", "openai/gpt-4o-mini", 0, 1_000_000)
    record_llm_usage("agent", "unknown/model", 10, 20)
    summary = usage_summary()
    gen = summary["by_usage"]["generation::openai/gpt-4o-mini"]
    assert gen["calls"] == 2
    assert gen["prompt_tokens"] == 1_000_000
    assert gen["completion_tokens"] == 1_000_000
    assert gen["total_tokens"] == 2_000_000
    assert gen["cost_usd"] == 0.75
    assert summary["totals"]["total_tokens"] == 2_000_030
    assert summary["totals"]["cost_usd"] == 0.75


def _mocked_client(response_data: dict) -> OpenRouterClient:
    http_client = MagicMock(spec=httpx.Client)
    fake_request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    http_client.post.return_value = httpx.Response(200, json=response_data, request=fake_request)
    return OpenRouterClient(
        settings=MagicMock(
            openrouter_api_key="test-key",
            openrouter_model="openai/gpt-4o-mini",
            openrouter_base_url="https://api.test/v1",
            llm_timeout_seconds=30.0,
            llm_json_mode=False,
            llm_max_retries=0,
        ),
        http_client=http_client,
    )


@pytest.mark.unit
def test_generate_records_cost_and_tokens():
    client = _mocked_client(
        {
            "model": "openai/gpt-4o-mini",
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        }
    )
    resp = client.generate(
        system_prompt="s",
        user_prompt="u",
        evidence_context="",
        max_output_tokens=10,
        usage_label="generation",
    )
    assert resp.usage.cost_usd == 0.75
    totals = usage_summary()["totals"]
    assert totals["prompt_tokens"] == 1_000_000
    assert totals["cost_usd"] == 0.75


@pytest.mark.unit
def test_chat_with_tools_records_usage_per_turn():
    client = _mocked_client(
        {
            "model": "nex-agi/nex-n2.5-mini:free",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "search_documents", "arguments": '{"query": "q"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 500, "completion_tokens": 60, "total_tokens": 560},
        }
    )
    turn = client.chat_with_tools(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        max_output_tokens=100,
    )
    assert turn.message["tool_calls"][0]["id"] == "c1"
    assert turn.usage.prompt_tokens == 500
    assert turn.usage.completion_tokens == 60
    assert turn.usage.cost_usd == 0.0  # free tier: priced, zero cost
    totals = usage_summary()["totals"]
    assert totals["calls"] == 1
    assert totals["total_tokens"] == 560
    assert "agent::nex-agi/nex-n2.5-mini" in usage_summary()["by_usage"]
