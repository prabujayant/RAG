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
from app.observability.cost import record_llm_usage

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
    cost_usd: float | None = None


@dataclass
class LLMResponse:
    """Normalized response from an LLM client."""

    text: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: dict | None = None
    latency_ms: float | None = None


@dataclass
class ToolChatTurn:
    """One assistant turn from a tool-enabled chat call."""

    message: dict
    usage: LLMUsage = field(default_factory=LLMUsage)


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
        max_retries: int | None = None,
        usage_label: str = "generation",
    ) -> LLMResponse:
        """Generate a response given system prompt, user prompt, and evidence.

        ``max_retries`` overrides the configured retry count for this call
        (0 disables retrying, which is useful for latency-sensitive callers).
        ``usage_label`` attributes token/cost recording (e.g. "grounding").
        """
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
        # Reuse a shared HTTP client to pool connections across calls.
        self._http_client: httpx.Client = http_client or self._build_client()

    def _build_client(self) -> httpx.Client:
        """Build a keep-alive HTTP client with connection pooling."""
        return httpx.Client(
            base_url=self._settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._settings.llm_timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    def _make_request(self, payload: dict) -> httpx.Response:
        """Make an HTTP request to the OpenRouter API."""
        response = self._http_client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response

    def _post_chat(self, payload: dict) -> dict:
        """POST a chat payload, falling back once without JSON mode on HTTP 400."""
        try:
            response = self._make_request(payload)
        except httpx.HTTPStatusError as exc:
            # Not every provider/model accepts `response_format`.
            if "response_format" in payload and exc.response.status_code == 400:
                logger.warning("Provider rejected response_format (HTTP 400); retrying without JSON mode.")
                payload.pop("response_format", None)
                response = self._make_request(payload)
            else:
                raise
        return response.json()

    def _reasoning_param(self) -> dict[str, object]:
        """Return the reasoning-effort payload override, or {} for default.

        Lower-than-default effort short-circuits the model's thinking and is
        the single biggest latency lever on reasoning models. "high" sends
        nothing (provider default) so the default path is byte-identical.
        """
        effort = getattr(self._settings, "llm_reasoning_effort", "high")
        if not isinstance(effort, str):
            return {}
        if effort.lower() == "high":
            return {}
        return {"reasoning": {"effort": effort.lower()}}

    def chat_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str | dict = "auto",
        max_output_tokens: int,
        temperature: float | None = None,
        max_retries: int | None = None,
        usage_label: str = "agent",
    ) -> ToolChatTurn:
        """Call the chat API with function tools; return the turn + usage.

        Unlike :meth:`generate` this passes no ``response_format`` (JSON mode
        conflicts with tool calling on several providers). Token usage from
        the final response is aggregated into the cost registry under
        ``usage_label`` so multi-turn agent runs stay attributable.
        """
        if not self._settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        payload: dict[str, object] = {
            "model": self._settings.openrouter_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_output_tokens,
        }
        payload.update(self._reasoning_param())
        if temperature is not None:
            payload["temperature"] = temperature

        retries = self._settings.llm_max_retries if max_retries is None else max_retries
        attempts = max(1, retries + 1)
        raw: dict = {}
        message: dict = {}

        for attempt in range(attempts):
            try:
                raw = self._post_chat(payload)
            except httpx.HTTPStatusError as exc:
                if attempt == attempts - 1 or not self._is_retryable(exc):
                    raise
                logger.warning(
                    "Tool chat failed with HTTP %s; retrying (%d/%d)",
                    exc.response.status_code,
                    attempt + 1,
                    attempts,
                )
                self._backoff(attempt)
                continue

            choices = raw.get("choices") or []
            if not choices:
                raise ValueError(f"Unexpected OpenRouter response: no choices: {raw}")
            message = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
            # Reasoning models may return content=null; tool_calls still count.
            has_content = bool((message.get("content") or "").strip())
            has_tools = bool(message.get("tool_calls"))
            if has_content or has_tools or attempt == attempts - 1:
                break
            logger.warning(
                "Tool chat returned neither content nor tool calls; retrying (%d/%d)",
                attempt + 1,
                attempts,
            )
            self._backoff(attempt)

        model = raw.get("model", self._settings.openrouter_model)
        usage_data = raw.get("usage", {}) if isinstance(raw, dict) else {}
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )
        usage.cost_usd = record_llm_usage(usage_label, model, usage.prompt_tokens, usage.completion_tokens)
        return ToolChatTurn(message=message, usage=usage)

    @staticmethod
    def _is_retryable(exc: httpx.HTTPStatusError) -> bool:
        """429 (rate limit) and 5xx (transient upstream) are worth retrying."""
        code = exc.response.status_code
        return code == 429 or code >= 500

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Brief backoff between retries (kept short to bound added latency)."""
        time.sleep(min(0.5 * (2**attempt), 2.0))

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        evidence_context: str,
        max_output_tokens: int,
        temperature: float | None = None,
        max_retries: int | None = None,
        usage_label: str = "generation",
    ) -> LLMResponse:
        """Call the OpenRouter API and return a normalized response.

        Retries when the upstream returns a rate-limit/5xx error or an *empty*
        completion. Free-tier providers do both intermittently, and an empty
        completion would otherwise surface as an unparseable answer. Pass
        ``max_retries=0`` to fail fast instead. Token usage is aggregated
        into the cost registry under ``usage_label``.
        """
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
        payload.update(self._reasoning_param())
        if temperature is not None:
            payload["temperature"] = temperature
        if self._settings.llm_json_mode:
            # Reasoning models otherwise stream chain-of-thought into `content`
            # and never emit the JSON payload we need.
            payload["response_format"] = {"type": "json_object"}

        retries = self._settings.llm_max_retries if max_retries is None else max_retries
        attempts = max(1, retries + 1)
        start = time.perf_counter()
        raw: dict = {}
        text = ""

        for attempt in range(attempts):
            try:
                raw = self._post_chat(payload)
            except httpx.HTTPStatusError as exc:
                if attempt == attempts - 1 or not self._is_retryable(exc):
                    raise
                logger.warning(
                    "LLM request failed with HTTP %s; retrying (%d/%d)",
                    exc.response.status_code,
                    attempt + 1,
                    attempts,
                )
                self._backoff(attempt)
                continue

            choices = raw.get("choices") or []
            if not choices:
                raise ValueError(f"Unexpected OpenRouter response: no choices: {raw}")

            # Reasoning models may return `content: null`; never let None reach
            # the JSON parser.
            text = (choices[0].get("message") or {}).get("content") or ""
            if text.strip() or attempt == attempts - 1:
                break

            logger.warning("LLM returned empty content; retrying (%d/%d)", attempt + 1, attempts)
            self._backoff(attempt)

        latency_ms = (time.perf_counter() - start) * 1000
        model = raw.get("model", self._settings.openrouter_model)

        choices = raw.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        if finish_reason == "length":
            logger.warning(
                "LLM response truncated at max_tokens=%s (finish_reason=length); "
                "the JSON payload may be incomplete. Consider raising "
                "MAX_ANSWER_TOKENS.",
                max_output_tokens,
            )

        usage_data = raw.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )
        usage.cost_usd = record_llm_usage(usage_label, model, usage.prompt_tokens, usage.completion_tokens)

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
        max_retries: int | None = None,
        usage_label: str = "generation",
    ) -> LLMResponse:
        """Return the configured mock response (no cost recording — no real tokens)."""
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
