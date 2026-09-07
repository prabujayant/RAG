"""Generation service: orchestrates evidence + LLM → structured answer."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from app.config import get_settings
from app.config.settings import Settings
from app.generation.client import LLMClient, OpenRouterClient
from app.generation.prompts import build_prompts
from app.generation.schemas import (
    AnswerResponse,
    Citation,
    GroundingStatus,
)

if TYPE_CHECKING:
    from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)
_JSON_BRACES_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw_text: str) -> str | None:
    """Extract the first JSON object from *raw_text*.

    Handles:
    - Bare JSON: ``{...}``
    - Markdown code fences: `````json\n{...}\n``` ``
    - Extra text before/after the JSON block
    - Malformed JSON (returns None so caller can fall back to refusal)
    """
    # Try fenced JSON first
    match = _JSON_CODE_FENCE_RE.search(raw_text)
    if match:
        return match.group(1).strip()

    # Try bare JSON object
    match = _JSON_BRACES_RE.search(raw_text)
    if match:
        return match.group(0).strip()

    return None


def safe_parse_json(raw: str) -> dict | None:
    """Parse *raw* as JSON, returning None on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Citation validation helpers
# ---------------------------------------------------------------------------

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


def _find_chunk_id_for_citation(
    citation_id: str, candidates: list[RetrievalResult]
) -> str | None:
    """Map a citation_id like [C1] to the corresponding chunk_id."""
    # citation_id is like "[C1]" — map to index
    stripped = citation_id.strip("[]C")
    try:
        idx = int(stripped) - 1
    except ValueError:
        return None

    if idx < 0 or idx >= len(candidates):
        return None

    return candidates[idx].chunk_id


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

        Returns
        -------
        AnswerResponse
            A structured answer with citations, grounding metadata, and latency.
        """
        start = time.perf_counter()

        # Step 1: build prompts
        system_prompt, user_prompt = build_prompts(question, candidates)

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

        if data is None:
            logger.warning("Failed to parse JSON from LLM response: %s", raw_text[:200])
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

        # Step 6: compute grounding status
        grounding_status = self._compute_grounding_status(refused, citations)

        latency_ms = (time.perf_counter() - start) * 1000

        return AnswerResponse(
            answer=answer,
            citations=citations,
            claims=[],  # Claim-level validation deferred to Phase 7
            grounded=grounding_status == GroundingStatus.GROUNDED,
            grounding_status=grounding_status,
            confidence=confidence,
            refused=refused,
            refused_reason=refused_reason,
            total_latency_ms=latency_ms,
            model=response.model,
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
        raw_citations: list[dict] = data.get("citations") or []
        citations: list[Citation] = []

        for c in raw_citations:
            try:
                citation_id = _normalize_citation_id(c.get("citation_id", ""))
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
