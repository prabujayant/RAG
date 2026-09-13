"""Generation service: orchestrates evidence + LLM → structured answer."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import get_settings
from app.config.settings import Settings
from app.generation.client import LLMClient, OpenRouterClient
from app.generation.prompts import build_prompts
from app.generation.schemas import (
    AnswerResponse,
    Citation,
    GroundingStatus,
    TokenUsage,
)
from app.observability.metrics import count as count_metric
from app.safety import drop_tainted_chunks, redact_pii

if TYPE_CHECKING:
    from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(raw_text: str | None) -> str | None:
    """Extract the JSON object from *raw_text*.

    Handles:
    - Bare JSON (the whole response, as produced in JSON mode)
    - Markdown code fences
    - Extra text before/after the JSON block
    - Malformed JSON (returns None so caller can fall back to refusal)
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Fast path: the whole response is a JSON object (the JSON-mode case).
    if text.startswith("{") and safe_parse_json(text) is not None:
        return text

    # Try candidates back-to-front: reasoning models emit a chain-of-thought
    # preamble (which may itself contain braces) *before* the real payload.
    for candidate in reversed(_iter_json_candidates(text)):
        if safe_parse_json(candidate) is not None:
            return candidate

    # Final fallback: the response may have been cut off mid-JSON because the
    # model hit max_tokens (finish_reason="length"). A truncated object never
    # closes its braces, so it yields no candidate above; repair it instead of
    # discarding an otherwise usable answer.
    start = text.find("{")
    if start != -1:
        repaired = _repair_truncated_json(text[start:])
        if repaired is not None and safe_parse_json(repaired) is not None:
            return repaired

    return None


def _repair_truncated_json(text: str) -> str | None:
    """Best-effort repair of a JSON object truncated mid-value.

    Closes an unterminated string and appends the missing container closers so
    a response cut off by ``max_tokens`` can still be parsed. Returns ``None``
    when the fragment cannot be repaired. The recovered text is intentionally
    partial — the caller treats it as the model's answer.
    """
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return None  # structurally broken, not merely truncated

    if not stack and not in_string:
        return text

    repaired = text
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    repaired += "".join(reversed(stack))

    try:
        json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        return None
    return repaired


def _iter_json_candidates(text: str) -> list[str]:
    """Return brace/fence-delimited JSON candidates from *text*, in order.

    Braces inside string literals are ignored so quoted text such as
    ``"answer": "use {braces}"`` cannot corrupt the depth tracking.
    """
    candidates: list[str] = []

    # 1. Markdown code fences.
    candidates.extend(m.group(1).strip() for m in _JSON_CODE_FENCE_RE.finditer(text))

    # 2. Top-level brace-balanced spans.
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1].strip())
                start = None

    return candidates


def safe_parse_json(raw: str) -> dict | None:
    """Parse *raw* as JSON, returning None on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Citation validation helpers
# ---------------------------------------------------------------------------

_CITATION_MARKER_RE = re.compile(r"\[C(\d+)\]", re.IGNORECASE)


def _coerce_citations(raw: object) -> list[dict]:
    """Normalize the model's ``citations`` field into a list of dicts.

    Models sometimes emit a bare string (``"[C1]"`` or ``"[C1], [C2]"``) or a
    single object instead of the documented list of citation objects. Iterating
    over a string would yield individual characters, so anything that is not a
    list of objects is normalized here.
    """
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        return [{"citation_id": match.group(0)} for match in _CITATION_MARKER_RE.finditer(raw)]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _normalize_citation_id(cid: str) -> str:
    """Strip whitespace and normalize a citation ID to [C{n}] form."""
    cid = cid.strip()
    if not cid.startswith("["):
        cid = f"[{cid}"
    if not cid.endswith("]"):
        cid = f"{cid}]"
    return cid


def _build_chunk_id_set(candidates: list[RetrievalResult]) -> set[str]:
    """Return the set of valid chunk_ids in the candidate list."""
    return {r.chunk_id for r in candidates}


def _build_citation_id_set(candidates: list[RetrievalResult]) -> set[str]:
    """Return valid citation IDs (e.g. [C1], [C2]) for the candidates."""
    return {f"[C{i}]" for i in range(1, len(candidates) + 1)}


def _citation_index(citation_id: str) -> int | None:
    """Return the 0-based candidate index for a citation_id like ``[C1]``.

    Returns ``None`` if the ID cannot be parsed or is out of range (checked
    by the caller against the candidate length).
    """
    stripped = citation_id.strip("[]C")
    try:
        idx = int(stripped) - 1
    except ValueError:
        return None
    if idx < 0:
        return None
    return idx


def _candidate_for_citation_id(
    citation_id: str,
    candidates: list[RetrievalResult],
) -> RetrievalResult | None:
    """Map a citation_id like ``[C1]`` to its corresponding candidate chunk."""
    idx = _citation_index(citation_id)
    if idx is None or idx >= len(candidates):
        return None
    return candidates[idx]


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class GenerationService:
    """Service that converts a question + evidence into a grounded answer.

    Parameters
    ----------
    client:
        An :class:`LLMClient` instance (e.g. :class:`OpenRouterClient` or
        :class:`MockLLM`).  If ``None`` a client is built from *settings*.
    settings:
        Application settings.  If ``None`` the global settings singleton is used.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @property
    def client(self) -> LLMClient:
        """Lazily build the LLM client on first use."""
        if self._client is None:
            self._client = OpenRouterClient(settings=self._settings)
        return self._client

    def generate(
        self,
        question: str,
        candidates: list[RetrievalResult],
        *,
        temperature: float | None = None,
        allow_generic: bool = False,
    ) -> AnswerResponse:
        """Generate a grounded answer from the question and retrieved evidence.

        Parameters
        ----------
        question:
            The user's question.
        candidates:
            Evidence chunks from the retrieval pipeline.
        temperature:
            Override the model's sampling temperature.
        allow_generic:
            When True, the model may answer from general knowledge where the
            evidence is thin instead of refusing. The response is flagged
            ``generic=True`` and is never reported as grounded.

        Returns
        -------
        AnswerResponse
            A structured answer with citations, grounding metadata, and latency.
        """
        start = time.perf_counter()

        # Step 0: safety — drop evidence carrying instruction-takeover
        # payloads so tainted text never reaches the model prompt. An empty
        # remainder flows into the normal no-evidence refusal below.
        candidates, dropped = drop_tainted_chunks(candidates)
        if dropped:
            count_metric("safety.chunks_dropped", dropped)

        # Step 1: build prompts
        system_prompt, user_prompt = build_prompts(question, candidates, allow_generic=allow_generic)

        # Step 2: call LLM
        try:
            response = self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                evidence_context="",  # evidence is already in user_prompt
                max_output_tokens=self._settings.max_answer_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed: %s", exc)
            return self._build_error_response(
                f"Generation failed: {exc}",
                start=start,
                model=getattr(self._client, "__class__", type(self._client)).__name__,
            )

        # Step 3: parse JSON
        raw_text = response.text
        json_str = extract_json(raw_text)
        data = safe_parse_json(json_str) if json_str else None

        if not isinstance(data, dict):
            logger.warning(
                "Failed to parse JSON object from LLM response: %r",
                (raw_text or "")[:200],
            )
            return self._build_error_response(
                "Failed to parse model output as JSON.",
                start=start,
                model=response.model,
            )

        # Step 4: validate and sanitize citations
        valid_chunk_ids = _build_chunk_id_set(candidates)
        valid_citation_ids = _build_citation_id_set(candidates)

        citations = self._parse_and_validate_citations(
            data,
            candidates,
            valid_chunk_ids,
            valid_citation_ids,
        )

        # Step 5: extract answer fields
        answer = str(data.get("answer", ""))
        refused = bool(data.get("refused", False))
        refused_reason = data.get("refused_reason")
        confidence = self._normalize_confidence(data.get("confidence"))

        # Step 5b: safety — mask PII in the answer and citation excerpts
        # before anything is returned. Grounding validation (a later stage)
        # runs on the same masked text, so scores stay consistent.
        answer, redacted = redact_pii(answer)
        if refused_reason:
            refused_reason, n = redact_pii(refused_reason)
            redacted += n
        for citation in citations:
            citation.text, n = redact_pii(citation.text)
            redacted += n
        if redacted:
            count_metric("safety.pii_redacted", redacted)

        # Step 6: compute grounding status
        grounding_status = self._compute_grounding_status(refused, citations)

        latency_ms = (time.perf_counter() - start) * 1000

        call_usage = response.usage
        usage = TokenUsage(
            prompt_tokens=call_usage.prompt_tokens or 0,
            completion_tokens=call_usage.completion_tokens or 0,
            total_tokens=call_usage.total_tokens
            or (call_usage.prompt_tokens or 0) + (call_usage.completion_tokens or 0),
            cost_usd=call_usage.cost_usd,
        )

        return AnswerResponse(
            answer=answer,
            citations=citations,
            claims=[],  # Claim-level validation deferred to Phase 7
            grounded=grounding_status == GroundingStatus.GROUNDED,
            grounding_status=grounding_status,
            confidence=confidence,
            refused=refused,
            refused_reason=refused_reason,
            generic=allow_generic and not refused,
            total_latency_ms=latency_ms,
            model=response.model,
            usage=usage,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_and_validate_citations(
        self,
        data: dict,
        candidates: list[RetrievalResult],
        valid_chunk_ids: set[str],
        valid_citation_ids: set[str],
    ) -> list[Citation]:
        """Parse citation objects, dropping those with unknown IDs.

        Citation markers like ``[C1]`` map 1:1 to the retrieved candidate list
        (``[C{i}]`` ↔ ``candidates[i - 1]``). Evidence text and chunk metadata
        are therefore resolved authoritatively from the *candidate* that was
        shown to the model, rather than trusting the model to echo back the
        ``chunk_id`` and ``text`` byte-for-byte. This makes grounding robust to
        models that truncate, rephrase, or omit the text/chunk fields.
        """
        raw_citations = _coerce_citations(data.get("citations"))
        citations: list[Citation] = []
        # A citation id maps 1:1 to a candidate, so a repeated marker (the model
        # often echoes the same [C1] several times) is always redundant. Keeping
        # duplicates produced two entries with the same citation_id, which the
        # UI renders as duplicate rows and React rejects as duplicate keys.
        seen_citation_ids: set[str] = set()

        for c in raw_citations:
            try:
                citation_id = _normalize_citation_id(c.get("citation_id") or "")
            except (TypeError, ValueError):
                continue  # skip malformed entries

            # Reject unknown citation IDs (e.g. [C99] when only 5 chunks shown)
            if citation_id not in valid_citation_ids:
                logger.debug(
                    "Dropping citation with unknown ID %r (valid: %s)",
                    citation_id,
                    valid_citation_ids,
                )
                continue

            if citation_id in seen_citation_ids:
                logger.debug("Dropping duplicate citation %r", citation_id)
                continue
            seen_citation_ids.add(citation_id)

            # Map [C{i}] -> candidates[i - 1] to get the authoritative chunk.
            candidate = _candidate_for_citation_id(citation_id, candidates)
            if candidate is None:
                continue

            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    page_number=candidate.page_number,
                    section=candidate.section,
                    # Bare file name so the UI can show which document the
                    # answer actually came from. The 8-hex prefix is added when
                    # an upload collides with an existing file name; strip it
                    # for display so the user sees the original name.
                    source=(
                        re.sub(r"^[0-9a-f]{8}_", "", Path(candidate.source).name)
                        if candidate.source
                        else None
                    ),
                )
            )

        return citations

    def _normalize_confidence(self, value: object) -> float | None:
        """Coerce *value* to a float in [0, 1], or return None."""
        if value is None:
            return None
        if not isinstance(value, (int, float, str)):
            return None
        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return None

    def _compute_grounding_status(
        self,
        refused: bool,
        citations: list[Citation],
    ) -> GroundingStatus:
        """Determine the aggregate grounding status."""
        if refused:
            return GroundingStatus.REFUSED
        if not citations:
            return GroundingStatus.UNGROUNDED
        return GroundingStatus.GROUNDED

    def _build_error_response(
        self,
        message: str,
        *,
        start: float,
        model: str,
    ) -> AnswerResponse:
        """Return a safe error response without fabricating an answer."""
        return AnswerResponse(
            answer=message,
            citations=[],
            claims=[],
            grounded=False,
            grounding_status=GroundingStatus.REFUSED,
            confidence=None,
            refused=True,
            refused_reason=message,
            total_latency_ms=(time.perf_counter() - start) * 1000,
            model=model,
        )
