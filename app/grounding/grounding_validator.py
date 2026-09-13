"""Aggregate grounding validation.

Takes an AnswerResponse, extracts claims, validates citations,
and produces a grounded response with status and confidence.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.generation.schemas import (
    AnswerClaim,
    AnswerResponse,
    Citation,
    CitationStatus,
    GroundingStatus,
)
from app.generation.service import extract_json, safe_parse_json
from app.grounding.citation_validator import CitationValidator, ValidationResult
from app.grounding.claims import Claim, ClaimExtractor

if TYPE_CHECKING:
    from app.generation.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Judge prompt
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM = (
    "You are a strict factual accuracy judge. "
    "Given a CLAIM and cited EVIDENCE, respond with ONLY a JSON object "
    "and nothing else:\n"
    '{"status": "supported" | "partially_supported" | "unsupported", '
    '"score": <float 0.0-1.0>, '
    '"reason": "<brief explanation>"}\n'
    "Be strict: invented or unreferenced facts must be marked unsupported."
)

_LLM_JUDGE_USER = "CLAIM: {claim}\n\nEVIDENCE (source: {source}, section: {section}):\n{evidence}"


def _build_llm_judge_messages(
    claim: str,
    evidence: str,
    section: str | None,
    source: str | None,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM judge."""
    system_prompt = _LLM_JUDGE_SYSTEM
    user_prompt = _LLM_JUDGE_USER.format(
        claim=claim,
        evidence=evidence,
        section=section or "unknown",
        source=source or "unknown document",
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# ValidationResult helpers
# ---------------------------------------------------------------------------


def _parse_llm_judge_response(raw: str | None) -> dict | None:
    """Extract the judge's JSON verdict, or ``None`` when unavailable.

    Delegates to the shared extraction helpers so reasoning-model output
    (chain-of-thought preamble followed by the JSON payload) is handled the
    same way as answer generation. Never raises on ``None`` input.
    """
    parsed = safe_parse_json(extract_json(raw))
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# GroundingValidator
# ---------------------------------------------------------------------------


@dataclass
class GroundingConfig:
    """Configuration for the grounding validator.

    Attributes
    ----------
    use_llm_judge:
        Whether to use the LLM judge for validation. Default True (falls back
        to deterministic checks when the judge is unavailable or fails).
    supported_threshold:
        Token overlap ratio for SUPPORTED. Default 0.55.
    partial_threshold:
        Token overlap ratio for PARTIALLY_SUPPORTED. Default 0.20.
    confidence_scale_with_support:
        If True, confidence is multiplied by average support score. Default True.
    judge_max_tokens:
        Max completion tokens for the LLM judge's JSON verdict.
    """

    use_llm_judge: bool = True
    supported_threshold: float = 0.55
    partial_threshold: float = 0.20
    confidence_scale_with_support: bool = True
    judge_max_tokens: int = 512


class GroundingValidator:
    """Validates an AnswerResponse and produces a grounded result.

    Parameters
    ----------
    config:
        GroundingConfig controlling behavior. Default is deterministic-only.
    citation_validator:
        CitationValidator instance. If None, created with defaults.
    claim_extractor:
        ClaimExtractor instance. If None, created with defaults.
    llm_client:
        Optional LLMClient for LLM-assisted validation.
    """

    def __init__(
        self,
        config: GroundingConfig | None = None,
        citation_validator: CitationValidator | None = None,
        claim_extractor: ClaimExtractor | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config or GroundingConfig()
        self.citation_validator = citation_validator or CitationValidator(
            supported_threshold=self.config.supported_threshold,
            partial_threshold=self.config.partial_threshold,
        )
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self._llm_client = llm_client

    def validate(self, response: AnswerResponse) -> AnswerResponse:
        """Validate an AnswerResponse and return an updated copy.

        This method:
        1. Extracts claims from the answer text
        2. Validates each citation against the evidence
        3. Calculates grounding status and confidence
        4. Returns a new AnswerResponse with updated fields
        """
        from app.generation.schemas import GroundingStatus

        # If the answer was refused, propagate that
        if response.refused:
            return self._build_refused_response(response)

        # Extract claims from answer text
        raw_claims = self.claim_extractor.extract(response.answer)

        # Build a map from citation_id -> Citation for lookup
        citation_map: dict[str, Citation] = {c.citation_id: c for c in response.citations}

        # Validate all cited claims in parallel so LLM judge calls are concurrent.
        # Each call is I/O-bound (HTTP request to the LLM API), so ThreadPoolExecutor
        # gives near-linear speedup on multi-core machines.
        validated_claims: list[AnswerClaim] = []
        if raw_claims:
            with ThreadPoolExecutor(max_workers=min(len(raw_claims), 8)) as pool:
                futures = {
                    pool.submit(self._validate_claim, raw, citation_map, response.citations): idx
                    for idx, raw in enumerate(raw_claims)
                }
                # Collect results in submission order to preserve claim sequence
                results: list[AnswerClaim | None] = [None] * len(raw_claims)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                validated_claims = [r for r in results if r is not None]

        # Calculate aggregate grounding status
        grounding_status = self._compute_grounding_status(validated_claims, response.refused)

        # A general-mode answer may include context beyond the document, so it
        # can never be reported as *fully* grounded even when every cited claim
        # checks out. Called-out here so the API contract ("generic answers are
        # never grounded") holds regardless of the claim verdicts.
        if response.generic and grounding_status == GroundingStatus.GROUNDED:
            grounding_status = GroundingStatus.PARTIALLY_GROUNDED

        # Calculate grounding confidence
        confidence = self._compute_confidence(
            validated_claims,
            response.confidence,
            grounding_status,
        )

        grounded = grounding_status == GroundingStatus.GROUNDED

        return AnswerResponse(
            answer=response.answer,
            citations=response.citations,
            claims=validated_claims,
            grounded=grounded,
            grounding_status=grounding_status,
            confidence=confidence,
            refused=response.refused,
            refused_reason=response.refused_reason,
            generic=response.generic,
            total_latency_ms=response.total_latency_ms,
            model=response.model,
            usage=response.usage,
        )

    def _validate_claim(
        self,
        raw: Claim,
        citation_map: dict[str, Citation],
        all_citations: list[Citation],
    ) -> AnswerClaim:
        """Validate a single claim against its cited evidence.

        Returns an AnswerClaim with status and reason.
        """
        from app.generation.schemas import CitationStatus

        # If no citations, mark as unsupported with no reason
        if not raw.citation_ids:
            return AnswerClaim(
                claim=raw.text,
                citation_ids=[],
                status=CitationStatus.UNSUPPORTED,
                reason="Claim has no citations.",
            )

        validated_statuses: list[CitationStatus] = []
        reasons: list[str] = []

        for cid in raw.citation_ids:
            citation = citation_map.get(cid)
            if citation is None:
                # Unknown citation ID
                validated_statuses.append(CitationStatus.UNSUPPORTED)
                reasons.append(f"Unknown citation ID: {cid}.")
                continue

            evidence_text = citation.text
            validation_result = self._do_validate(
                raw.text,
                evidence_text,
                cid,
                citation.section,
                None,
            )
            validated_statuses.append(validation_result.status)
            reasons.append(f"[{cid}] {validation_result.reason}")

        # Aggregate: best status wins
        overall_status = self._aggregate_statuses(validated_statuses)
        # Combine reasons
        combined_reason = " ".join(reasons) if reasons else None

        return AnswerClaim(
            claim=raw.text,
            citation_ids=raw.citation_ids,
            status=overall_status,
            reason=combined_reason,
        )

    def _do_validate(
        self,
        claim: str,
        evidence: str,
        citation_id: str,
        section: str | None,
        source: str | None,
    ) -> ValidationResult:
        """Perform validation, using the LLM judge as the semantic arbiter.

        This mirrors OpenAI-style groundedness checks: rather than relying on
        brittle token-overlap heuristics, the judge reads the claim together
        with the cited evidence and decides whether the claim is supported.
        The deterministic validator remains as a fast fallback if the LLM
        judge is unavailable or fails.
        """
        # Fast deterministic path (used as a fallback / sanity baseline)
        result = self.citation_validator.validate(claim, evidence, citation_id)

        # When the LLM judge is enabled, its semantic verdict is authoritative.
        # This rescues answers the model paraphrases rather than copying verbatim,
        # which deterministic token-overlap would otherwise (incorrectly) flag.
        if self.config.use_llm_judge:
            llm_result = self._llm_judge(claim, evidence, citation_id, section, source)
            if llm_result is not None:
                raw_status = llm_result.get("status", "unsupported")
                try:
                    status = CitationStatus(raw_status)
                except ValueError:
                    status = CitationStatus.UNSUPPORTED
                try:
                    score = float(llm_result.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                reason = llm_result.get("reason", "LLM judge returned no reason.")
                return ValidationResult(status=status, score=score, reason=reason)

        return result

    def _llm_judge(
        self,
        claim: str,
        evidence: str,
        citation_id: str,
        section: str | None,
        source: str | None,
    ) -> dict | None:
        """Call the LLM judge, returning parsed JSON or None on failure."""
        if self._llm_client is None:
            return None

        system_prompt, user_prompt = _build_llm_judge_messages(claim, evidence, section, source)

        try:
            raw_response = self._llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                # The evidence is already interpolated into `user_prompt`; passing
                # it again here duplicates it in the message body and makes the
                # verdict less reliable.
                evidence_context="",
                max_output_tokens=self.config.judge_max_tokens,
                temperature=0.0,
                # Fail fast: an empty/failed verdict falls back to the
                # deterministic validator, and each retry costs a full LLM
                # round trip that would inflate query latency.
                max_retries=0,
                usage_label="grounding",
            )
            return _parse_llm_judge_response(raw_response.text)
        except Exception:
            # LLM judge failure → fall back to deterministic
            return None

    def _aggregate_statuses(self, statuses: list[CitationStatus]) -> CitationStatus:
        """Return the best (highest-priority) status from a list."""
        if not statuses:
            return CitationStatus.UNSUPPORTED
        if CitationStatus.SUPPORTED in statuses:
            return CitationStatus.SUPPORTED
        if CitationStatus.PARTIALLY_SUPPORTED in statuses:
            return CitationStatus.PARTIALLY_SUPPORTED
        return CitationStatus.UNSUPPORTED

    def _compute_grounding_status(
        self,
        claims: list[AnswerClaim],
        refused: bool,
    ) -> GroundingStatus:
        """Compute aggregate grounding status from validated claims."""
        if refused or not claims:
            return GroundingStatus.REFUSED

        statuses = [c.status for c in claims]
        if not statuses:
            return GroundingStatus.UNGROUNDED

        if all(s == CitationStatus.SUPPORTED for s in statuses):
            return GroundingStatus.GROUNDED
        if any(s == CitationStatus.SUPPORTED for s in statuses):
            return GroundingStatus.PARTIALLY_GROUNDED
        return GroundingStatus.UNGROUNDED

    def _compute_confidence(
        self,
        claims: list[AnswerClaim],
        llm_confidence: float | None,
        grounding_status: GroundingStatus,
    ) -> float | None:
        """Compute grounding confidence.

        confidence = base × claim_support_avg × citation_completeness
        """
        if grounding_status == GroundingStatus.REFUSED:
            return 0.0

        if not claims:
            return llm_confidence

        # Average claim support score from the validator
        claim_scores = [
            1.0
            if c.status == CitationStatus.SUPPORTED
            else 0.5
            if c.status == CitationStatus.PARTIALLY_SUPPORTED
            else 0.0
            for c in claims
        ]
        avg_support = sum(claim_scores) / len(claim_scores)

        # Citation completeness: fraction of claims that have at least one citation
        cited = sum(1 for c in claims if c.citation_ids)
        citation_completeness = cited / len(claims)

        # Base confidence from LLM, defaulting to 0.5
        base = llm_confidence if llm_confidence is not None else 0.5

        if self.config.confidence_scale_with_support:
            confidence = base * avg_support * citation_completeness
        else:
            confidence = base * avg_support

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _build_refused_response(self, response: AnswerResponse) -> AnswerResponse:
        """Build a grounded response for an already-refused answer."""
        return AnswerResponse(
            answer=response.answer,
            citations=response.citations,
            claims=[],
            grounded=False,
            grounding_status=GroundingStatus.REFUSED,
            confidence=0.0,
            refused=True,
            refused_reason=response.refused_reason,
            generic=False,
            total_latency_ms=response.total_latency_ms,
            model=response.model,
            usage=response.usage,
        )
