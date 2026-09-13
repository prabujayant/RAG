"""Deterministic citation validation based on text overlap.

Validates that a claim is supported by the cited evidence using
normalized token overlap, key phrase matching, and number/negation checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.generation.schemas import CitationStatus


@dataclass
class ValidationResult:
    """Result of validating a claim against a cited chunk of evidence.

    Attributes
    ----------
    status:
        SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED.
    score:
        A float in [0.0, 1.0] indicating how strongly the evidence supports the claim.
    reason:
        Human-readable explanation of the validation decision.
    """

    status: CitationStatus
    score: float
    reason: str


# Minimum token-overlap ratio to consider a claim "supported"
_OVERLAP_SUPPORTED_THRESHOLD = 0.65
# Minimum ratio to consider "partially supported"
_OVERLAP_PARTIAL_THRESHOLD = 0.30


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase word-like tokens from *text*."""
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    # Remove common stopwords
    stopwords = frozenset([
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "up", "down", "out", "off", "over", "under", "again",
        "this", "that", "these", "those", "it", "its",
    ])
    return {t for t in tokens if t not in stopwords and len(t) > 1}


def _extract_numbers_and_codes(text: str) -> set[str]:
    """Extract numeric and code-like identifiers (e.g. 60, AES-256, v2, amsk_)."""
    # Numbers
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    # Alphanumeric codes like AES-256, v2.0, TLS-1.2 (case-insensitive)
    codes = set(
        re.findall(r"\b[A-Z][A-Z0-9]*(?:-\d+(?:\.\d+)?)+\b", text, flags=re.IGNORECASE)
    )
    # Underscore identifiers like amsk_, err_429, api_key_v2
    underscore_codes = set(re.findall(r"\b[a-z]+_[a-z0-9_]+\b", text.lower()))
    return numbers | codes | underscore_codes


def _has_negation(text: str) -> bool:
    """Return True if *text* contains a negation phrase."""
    negation_patterns = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnone\b",
        r"\bneither\b", r"\bnobody\b", r"\bnothing\b",
        r"\bdoesn't\b", r"\bdon't\b",
        r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
        r"\bhasn't\b", r"\bhaven't\b", r"\bhadn't\b",
        r"\bwon't\b", r"\bwouldn't\b", r"\bcan't\b", r"\bcannot\b",
        r"\bshouldn't\b", r"\bmustn't\b",
    ]
    return any(re.search(p, text.lower()) for p in negation_patterns)


def _compute_overlap(claim_tokens: set[str], evidence_tokens: set[str]) -> float:
    """Return token overlap ratio between claim and evidence.

    Ratio = |intersection| / |claim_tokens|.
    Returns 0.0 if claim_tokens is empty.
    """
    if not claim_tokens:
        return 0.0
    intersection = claim_tokens & evidence_tokens
    return len(intersection) / len(claim_tokens)


def _key_phrase_overlap(claim: str, evidence: str) -> float:
    """Return overlap score based on shared 2-word bigrams.

    Uses bigram (sliding window of 2) matching to handle phrase variations
    like "tokens expire in" vs "tokens expire" — both share the "tokens expire"
    bigram even though the phrases aren't identical.
    """
    # Split into lowercase words
    words = re.findall(r"\b[\w']+\b", claim.lower())
    if len(words) < 2:
        return 0.0

    # Build bigrams from claim words
    claim_bigrams = set()
    for i in range(len(words) - 1):
        claim_bigrams.add((words[i], words[i + 1]))

    if not claim_bigrams:
        return 0.0

    # Build bigrams from evidence words
    evidence_words = re.findall(r"\b[\w']+\b", evidence.lower())
    evidence_bigrams = set()
    for i in range(len(evidence_words) - 1):
        evidence_bigrams.add((evidence_words[i], evidence_words[i + 1]))

    # Count how many claim bigrams appear in evidence
    matches = sum(1 for bg in claim_bigrams if bg in evidence_bigrams)
    return matches / len(claim_bigrams)


def _check_number_mismatch(claim: str, evidence: str) -> bool:
    """Return True if claim contains numbers that don't appear in evidence.

    This catches cases like claiming "60 minutes" when evidence says "30 minutes".
    """
    claim_numbers = _extract_numbers_and_codes(claim)
    evidence_numbers = _extract_numbers_and_codes(evidence)
    # If claim has numbers/codes, at least one must appear in evidence
    return bool(claim_numbers) and not (claim_numbers & evidence_numbers)


def _check_negation_mismatch(claim: str, evidence: str) -> bool:
    """Return True if claim is affirmative but evidence is negated (or vice versa)."""
    claim_neg = _has_negation(claim)
    evidence_neg = _has_negation(evidence)
    # Mismatch: one has negation, the other doesn't. Only a problem if the
    # claim makes a specific assertion (affirmative claim + negative evidence).
    return claim_neg != evidence_neg


class CitationValidator:
    """Validates a claim against a cited evidence chunk.

    Parameters
    ----------
    supported_threshold:
        Token overlap ratio above which a claim is considered SUPPORTED.
        Default 0.65.
    partial_threshold:
        Token overlap ratio above which a claim is considered PARTIALLY_SUPPORTED.
        Default 0.30.
    min_claim_tokens:
        Minimum number of tokens in the claim for validation. Default 3.
    """

    def __init__(
        self,
        supported_threshold: float = _OVERLAP_SUPPORTED_THRESHOLD,
        partial_threshold: float = _OVERLAP_PARTIAL_THRESHOLD,
        min_claim_tokens: int = 2,
    ) -> None:
        if supported_threshold < partial_threshold:
            raise ValueError("supported_threshold must be >= partial_threshold")
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold
        self.min_claim_tokens = min_claim_tokens

    def validate(self, claim: str, evidence: str, citation_id: str | None = None) -> ValidationResult:
        """Validate a single claim against a single piece of evidence.

        Parameters
        ----------
        claim:
            The claim text to validate.
        evidence:
            The cited evidence text.
        citation_id:
            Optional citation ID for error messages.

        Returns
        -------
        ValidationResult
            The validation outcome with status, score, and reason.
        """
        from app.generation.schemas import CitationStatus

        claim = claim.strip()
        evidence = evidence.strip()

        if not claim or not evidence:
            return ValidationResult(
                status=CitationStatus.UNSUPPORTED,
                score=0.0,
                reason="Empty claim or evidence.",
            )

        # Check for negation mismatch first — semantic incompatibility overrides
        # token overlap, so check this before number/token checks
        if _check_negation_mismatch(claim, evidence):
            claim_neg = _has_negation(claim)
            evidence_neg = _has_negation(evidence)
            return ValidationResult(
                status=CitationStatus.UNSUPPORTED,
                score=0.0,
                reason=(
                    f"Negation mismatch: claim is "
                    f"{'negative' if claim_neg else 'affirmative'}, "
                    f"evidence is {'negative' if evidence_neg else 'affirmative'}."
                ),
            )

        # Check for number/code mismatch — strong negative signal
        if _check_number_mismatch(claim, evidence):
            return ValidationResult(
                status=CitationStatus.UNSUPPORTED,
                score=0.0,
                reason="The claim contains specific values not found in evidence.",
            )

        # Token-based overlap
        claim_tokens = _tokenize(claim)
        evidence_tokens = _tokenize(evidence)

        if len(claim_tokens) < self.min_claim_tokens:
            return ValidationResult(
                status=CitationStatus.PARTIALLY_SUPPORTED,
                score=0.5,
                reason="Claim too short for reliable validation.",
            )

        overlap = _compute_overlap(claim_tokens, evidence_tokens)
        key_phrase_score = _key_phrase_overlap(claim, evidence)
        # Weighted combination: 60% token overlap, 40% key phrases
        score = 0.6 * overlap + 0.4 * key_phrase_score

        if score >= self.supported_threshold:
            reason = "Evidence strongly supports the claim via token and phrase overlap."
            status = CitationStatus.SUPPORTED
        elif score >= self.partial_threshold:
            reason = "Evidence partially supports the claim."
            status = CitationStatus.PARTIALLY_SUPPORTED
        else:
            reason = "Insufficient evidence overlap to support the claim."
            status = CitationStatus.UNSUPPORTED

        return ValidationResult(status=status, score=score, reason=reason)
