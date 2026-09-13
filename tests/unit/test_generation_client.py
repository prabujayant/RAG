"""Unit tests for the LLM client interface and implementations."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from app.generation.client import (
    LLMUsage,
    MockLLM,
    OpenRouterClient,
)


class TestMockLLM:
    """Tests for the deterministic MockLLM."""

    def test_returns_configured_response(self) -> None:
        """Mock returns the configured text."""
        mock = MockLLM(response_text="hello world")
        resp = mock.generate(
            system_prompt="system",
            user_prompt="user",
            evidence_context="ctx",
            max_output_tokens=100,
        )
        assert resp.text == "hello world"
        assert resp.model == "mock/test"
        assert isinstance(resp.usage, LLMUsage)

    def test_tracks_calls(self) -> None:
        """Mock records every call for inspection."""
        mock = MockLLM(response_text="ok")
        mock.generate(
            system_prompt="sys1",
            user_prompt="usr1",
            evidence_context="ctx1",
            max_output_tokens=50,
        )
        mock.generate(
            system_prompt="sys2",
            user_prompt="usr2",
            evidence_context="ctx2",
            max_output_tokens=60,
        )
        assert len(mock.calls) == 2
        assert mock.calls[0]["system_prompt"] == "sys1"
        assert mock.calls[1]["max_output_tokens"] == 60

    def test_configure_mid_test(self) -> None:
        """configure() updates behaviour without resetting call history."""
        mock = MockLLM(response_text="v1")
        mock.generate(system_prompt="s", user_prompt="u", evidence_context="c", max_output_tokens=10)
        assert mock.calls[0]["system_prompt"] == "s"

        mock.configure(response_text="v2")
        resp = mock.generate(
            system_prompt="s2",
            user_prompt="u2",
            evidence_context="c2",
            max_output_tokens=20,
        )
        assert resp.text == "v2"
        assert len(mock.calls) == 2  # history preserved

    def test_simulated_latency(self) -> None:
        """latency_ms reflects the configured delay."""
        mock = MockLLM(response_text="slow", latency_ms=50)
        resp = mock.generate(system_prompt="s", user_prompt="u", evidence_context="c", max_output_tokens=10)
        assert resp.latency_ms == 50.0

    def test_raise_on_call(self) -> None:
        """raise_on_call raises an HTTP error."""
        mock = MockLLM(response_text="fail", raise_on_call=True)
        with pytest.raises(httpx.HTTPStatusError):
            mock.generate(system_prompt="s", user_prompt="u", evidence_context="c", max_output_tokens=10)

class TestOpenRouterClient:
    """Tests for OpenRouterClient using a fake HTTP client."""

    def _make_client(self, response_data: dict, status: int = 200) -> OpenRouterClient:
        """Build a client with a mocked HTTP response."""
        http_client = MagicMock(spec=httpx.Client)
        fake_request = httpx.Request("POST", "https://api.test/v1/chat/completions")
        fake_response = httpx.Response(status, json=response_data, request=fake_request)
        http_client.post.return_value = fake_response
        return OpenRouterClient(
            settings=MagicMock(
                openrouter_api_key="test-key",
                openrouter_model="test/model",
                openrouter_base_url="https://api.test/v1",
                llm_timeout_seconds=30.0,
                llm_json_mode=True,
                # No retries: keeps unit tests fast and deterministic.
                llm_max_retries=0,
            ),
            http_client=http_client,
        )

    def test_generate_returns_normalized_response(self) -> None:
        """generate() returns a properly structured LLMResponse."""
        client = self._make_client({
            "model": "test/model",
            "choices": [{"message": {"content": "test output"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        resp = client.generate(
            system_prompt="system",
            user_prompt="user",
            evidence_context="ctx",
            max_output_tokens=100,
        )
        assert resp.text == "test output"
        assert resp.model == "test/model"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5

    def test_generate_raises_on_missing_api_key(self) -> None:
        """Missing API key raises a clear error."""
        client = OpenRouterClient(
            settings=MagicMock(
                openrouter_api_key="",
                openrouter_base_url="https://api.test/v1",
                llm_timeout_seconds=30.0,
            ),
        )
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            client.generate(
                system_prompt="s",
                user_prompt="u",
                evidence_context="c",
                max_output_tokens=10,
            )

    def test_generate_passes_temperature(self) -> None:
        """Temperature is included in the request payload when set."""
        http_client = MagicMock(spec=httpx.Client)
        fake_request = httpx.Request("POST", "https://x/v1/chat/completions")
        http_client.post.return_value = httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": "x"}}],
        }, request=fake_request)
        client = OpenRouterClient(
            settings=MagicMock(
                openrouter_api_key="k",
                openrouter_model="m",
                openrouter_base_url="https://x/v1",
                llm_timeout_seconds=30.0,
                llm_json_mode=True,
                llm_max_retries=0,
            ),
            http_client=http_client,
        )
        client.generate(
            system_prompt="s",
            user_prompt="u",
            evidence_context="c",
            max_output_tokens=10,
            temperature=0.7,
        )
        payload = http_client.post.call_args[1]["json"]
        assert payload.get("temperature") == 0.7

    def test_reasoning_param_absent_on_high(self) -> None:
        """Default effort sends no reasoning override (provider default)."""
        http_client = MagicMock(spec=httpx.Client)
        fake_request = httpx.Request("POST", "https://x/v1/chat/completions")
        http_client.post.return_value = httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": "x"}}],
        }, request=fake_request)
        client = OpenRouterClient(
            settings=MagicMock(
                openrouter_api_key="k",
                openrouter_model="m",
                openrouter_base_url="https://x/v1",
                llm_timeout_seconds=30.0,
                llm_json_mode=False,
                llm_max_retries=0,
                llm_reasoning_effort="high",
            ),
            http_client=http_client,
        )
        client.generate(
            system_prompt="s", user_prompt="u", evidence_context="c",
            max_output_tokens=10,
        )
        assert "reasoning" not in http_client.post.call_args[1]["json"]

    def test_reasoning_param_sent_on_medium(self) -> None:
        """Lowered effort is forwarded as OpenRouter's reasoning.effort."""
        http_client = MagicMock(spec=httpx.Client)
        fake_request = httpx.Request("POST", "https://x/v1/chat/completions")
        http_client.post.return_value = httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": "x"}}],
        }, request=fake_request)
        client = OpenRouterClient(
            settings=MagicMock(
                openrouter_api_key="k",
                openrouter_model="m",
                openrouter_base_url="https://x/v1",
                llm_timeout_seconds=30.0,
                llm_json_mode=False,
                llm_max_retries=0,
                llm_reasoning_effort="medium",
            ),
            http_client=http_client,
        )
        client.generate(
            system_prompt="s", user_prompt="u", evidence_context="c",
            max_output_tokens=10,
        )
        payload = http_client.post.call_args[1]["json"]
        assert payload.get("reasoning") == {"effort": "medium"}

    def test_generate_raises_on_empty_choices(self) -> None:
        """Unexpected response shape (no choices) raises ValueError."""
        client = self._make_client({"model": "m", "choices": []})
        with pytest.raises(ValueError, match="no choices"):
            client.generate(
                system_prompt="s",
                user_prompt="u",
                evidence_context="c",
                max_output_tokens=10,
            )

class TestLLMClientProtocol:
    """Verify MockLLM and OpenRouterClient satisfy LLMClient at type level."""

    def test_mock_is_llm_client(self) -> None:
        """MockLLM is an LLMClient (structural check via ABC registration)."""
        mock = MockLLM()
        # At runtime just verify it has the right method
        assert hasattr(mock, "generate")

    def test_openrouter_is_llm_client(self) -> None:
        """OpenRouterClient is an LLMClient."""
        client = OpenRouterClient(
            settings=MagicMock(
                openrouter_api_key="k",
                openrouter_model="m",
                openrouter_base_url="https://x/v1",
                llm_timeout_seconds=30.0,
            ),
            http_client=MagicMock(spec=httpx.Client),
        )
        assert hasattr(client, "generate")
