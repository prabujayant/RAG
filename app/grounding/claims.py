"""Claim extraction from generated answer text.

Splits an answer into atomic claims and attaches citation IDs to each claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Pattern that matches citation markers like [C1], [C2], [C10]
_CITATION_RE = re.compile(r"\[[Cc](\d+)\]")

# Fragments to skip (headings, greetings, empty strings)
_SKIP_FRAGMENTS = frozenset([
    "",
    ".",
    "?",
    "!",
    "...",
])


@dataclass
class Claim:
    """A single atomic claim extracted from an answer.

    Attributes
    ----------
    text:
        The claim text without the citation markers.
    citation_ids:
        List of citation IDs found in the original text, e.g. ``["[C1]", "[C2]"]``.
    """

    text: str
    citation_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.citation_ids is None:  # pragma: no cover - defensive
            self.citation_ids = []


def _strip_citations(text: str) -> str:
    """Remove all citation markers from *text*."""
    return _CITATION_RE.sub("", text).strip()


def _normalize_text(text: str) -> str:
    """Collapse whitespace and strip punctuation from both ends."""
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing punctuation that commonly appears after citation markers
    text = re.sub(r"^[^\w\u00c0-\u024f]+|[^\w\u00c0-\u024f]+$", "", text)
    return text


class ClaimExtractor:
    """Extracts atomic claims from answer text.

    Parameters
    ----------
    min_claim_length:
        Minimum character length for a claim to be kept. Default 10.
    skip_fragments:
        Fragments to skip. Can be extended or replaced.
    """

    def __init__(
        self,
        min_claim_length: int = 10,
        skip_fragments: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.min_claim_length = min_claim_length
        self.skip_fragments = skip_fragments if skip_fragments is not None else _SKIP_FRAGMENTS

    def extract(self, text: str) -> list[Claim]:
        """Split *text* into atomic claims and return a list of Claim objects.

        Each returned claim has:
        - ``text``: the original text with citation markers stripped
        - ``citation_ids``: list of citation IDs found in the text
        """
        raw_claims = self._split_on_markers(text)
        claims: list[Claim] = []
        for raw in raw_claims:
            # Strip citations before normalization so markers like [C1] don't
            # interfere with punctuation stripping (the [ is stripped but C1]
            # would be left behind otherwise)
            stripped = _strip_citations(raw)
            normalized = _normalize_text(stripped)
            if self._should_skip(normalized):
                continue
            citation_ids = self._extract_citation_ids(raw)
            claims.append(Claim(text=normalized, citation_ids=citation_ids))
        return claims

    def _should_skip(self, text: str) -> bool:
        """Return True if *text* is a fragment that should be skipped."""
        if not text:
            return True
        if len(text) < self.min_claim_length:
            return True
        return text.lower() in self.skip_fragments

    def _split_on_markers(self, text: str) -> list[str]:
        """Split text on sentence boundaries and '...' while preserving context."""
        # First, normalize internal whitespace
        text = re.sub(r"\s+", " ", text)
        # Split on common sentence terminators followed by a space or end
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u00c0-\u024f(])", text)
        return [p.strip() for p in parts if p.strip()]

    def _extract_citation_ids(self, segment: str) -> list[str]:
        """Extract all unique citation IDs from a segment, preserving original form."""
        # Preserve original bracket style — find the exact string in the original
        seen: set[str] = set()
        result: list[str] = []
        for match in _CITATION_RE.finditer(segment):
            cid = match.group(0)  # e.g. "[C1]"
            if cid.lower() not in seen:
                seen.add(cid.lower())
                result.append(cid)
        return result


def extract_claims(text: str, **kwargs: object) -> list[Claim]:
    """Convenience wrapper around ``ClaimExtractor().extract()``.

    Accepts the same keyword arguments as ``ClaimExtractor``.
    """
    return ClaimExtractor(**kwargs).extract(text)  # type: ignore[arg-type]
