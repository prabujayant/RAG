"""LLM client interface and implementations.

Provides:
- ``LLMClient`` protocol defining the client interface.
- ``OpenRouterClient`` — production client backed by OpenAI-compatible API.
- ``MockLLM`` — deterministic fake for testing without API keys.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from app.config import get_settings
from app.config.settings import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    """Token usage metadata from an LLM response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResponse:
    """Normalized response from an LLM client."""

    text: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: dict | None = None
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract protocol for LLM clients."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        evidence_context: str,
        max_output_tokens: int,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a response given system prompt, user prompt, and evidence."""
        ...


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


class OpenRouterClient(LLMClient):
    """Production LLM client using the OpenAI-compatible OpenRouter API."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._http_client = http_client

    def _build_retry_client(self) -> httpx.Client:
        """Build an HTTP client with retry configuration."""
        return httpx.Client(
            base_url=self._settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._settings.llm_timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )

    def _make_request(self, payload: dict) -> httpx.Response:
        """Make an HTTP request to the OpenRouter API with retries."""
        client = self._http_client or self._build_retry_client()
        try:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            return response
        finally:
            if self._http_client is None:
                client.close()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        evidence_context: str,
        max_output_tokens: int,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Call the OpenRouter API and return a normalized response."""
        if not self._settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        payload: dict[str, object] = {
            "model": self._settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\n{evidence_context}",
                },
            ],
            "max_tokens": max_output_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        start = time.perf_counter()
        response = self._make_request(payload)
        latency_ms = (time.perf_counter() - start) * 1000

        raw = response.json()
        choices = raw.get("choices", [])
        if not choices:
            raise ValueError(f"Unexpected OpenRouter response: no choices: {raw}")

        message = choices[0].get("message", {})
        text = message.get("content", "")
        model = raw.get("model", self._settings.openrouter_model)

        usage_data = raw.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )

        return LLMResponse(
            text=text,
            model=model,
            usage=usage,
            raw=raw,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Mock LLM for testing
# ---------------------------------------------------------------------------


class MockLLM(LLMClient):
    """Deterministic fake LLM client for testing the generation pipeline.

    Parameters
    ----------
    response_text:
        Fixed text to return on every call.
    latency_ms:
        Simulated request latency.  Pass 0 for instant responses.
    raise_on_call:
        If True, raise an exception on every call.
    """

    def __init__(
        self,
        response_text: str = "This is a mock response.",
        latency_ms: float = 0.0,
        raise_on_call: bool = False,
    ) -> None:
        self._response_text = response_text
        self._latency_ms = latency_ms
        self._raise_on_call = raise_on_call
        self._calls: list[dict] = []

    @property
    def calls(self) -> list[dict]:
        """Record of every call made to this mock (read-only copy)."""
        return list(self._calls)

    def configure(
        self,
        response_text: str | None = None,
        latency_ms: float | None = None,
        raise_on_call: bool | None = None,
    ) -> None:
        """Update the mock's behaviour mid-test."""
        if response_text is not None:
            self._response_text = response_text
        if latency_ms is not None:
            self._latency_ms = latency_ms
        if raise_on_call is not None:
            self._raise_on_call = raise_on_call

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        evidence_context: str,
        max_output_tokens: int,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return the configured mock response."""
        if self._raise_on_call:
            raise httpx.HTTPStatusError(
                "Mock HTTP error",
                request=httpx.Request("POST", "http://mock"),
                response=httpx.Response(500),
            )

        self._calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "evidence_context": evidence_context,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )

        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000)

        return LLMResponse(
            text=self._response_text,
            model="mock/test",
            usage=LLMUsage(10, 20, 30),
            raw=None,
            latency_ms=self._latency_ms,
        )
