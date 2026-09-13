"""Deterministic safety screening: questions, evidence, and answers.

No LLM calls anywhere in this module — screening must be free, fast, and
un-influenceable by the text being screened. Three gates:

1. :func:`screen_question` — refuse disallowed or injection-carrying questions
   *before* any retrieval or LLM spend.
2. :func:`drop_tainted_chunks` — remove evidence chunks containing
   instruction-takeover payloads so they never reach a model prompt.
3. :func:`redact_pii` — mask emails/phones/SSNs/keys/card numbers in the
   final answer text (applied after validation so grounding scores are
   computed on the unredacted text, then masked for display).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.safety.patterns import (
    API_KEY_RE,
    DISALLOWED_QUESTION_PATTERNS,
    EMAIL_RE,
    INJECTION_PATTERNS,
    PHONE_RE,
    SSN_RE,
    find_card_numbers,
)

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"


class TaintScannable(Protocol):
    """Anything with the chunk fields the injection scan reads.

    Satisfied by both ``RetrievalResult`` (fixed pipeline) and
    ``AgentEvidence`` (agent path) without coupling safety to either.
    """

    @property
    def chunk_id(self) -> str: ...
    @property
    def document_id(self) -> str | None: ...
    @property
    def text(self) -> str | None: ...


@dataclass
class QuestionVerdict:
    """Outcome of screening one user question."""

    allowed: bool
    category: str | None = None  # "disallowed_content" | "prompt_injection"
    reason: str | None = None


def find_injection_matches(text: str) -> list[str]:
    """Return names of injection patterns found in *text* (empty = clean)."""
    if not text:
        return []
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]


def screen_question(question: str) -> QuestionVerdict:
    """Decide whether a question may be processed at all."""
    text = question or ""
    for name, pattern in DISALLOWED_QUESTION_PATTERNS:
        if pattern.search(text):
            return QuestionVerdict(
                allowed=False,
                category="disallowed_content",
                reason=(
                    "This question asks for disallowed content "
                    f"({name.replace('-', ' ')}), which I can't help with."
                ),
            )
    injections = find_injection_matches(text)
    if injections:
        return QuestionVerdict(
            allowed=False,
            category="prompt_injection",
            reason=(
                "This question contains instruction-takeover phrasing "
                f"({', '.join(injections)}). Please rephrase as a plain question "
                "about your document."
            ),
        )
    return QuestionVerdict(allowed=True)


def drop_tainted_chunks[T: TaintScannable](
    candidates: list[T],
) -> tuple[list[T], int]:
    """Split candidates into (clean, dropped_count) by injection scan."""
    clean: list[T] = []
    dropped = 0
    for candidate in candidates or []:
        if find_injection_matches(candidate.text or ""):
            dropped += 1
            logger.warning(
                "Dropping tainted chunk %s from %s (prompt-injection payload)",
                candidate.chunk_id,
                candidate.document_id,
            )
        else:
            clean.append(candidate)
    return clean, dropped


def redact_pii(text: str | None) -> tuple[str, int]:
    """Mask PII in *text*; return (masked_text, redaction_count)."""
    if not text:
        return "", 0
    count = 0
    masked = text
    for pattern in (EMAIL_RE, PHONE_RE, SSN_RE, API_KEY_RE):
        masked, n = pattern.subn(REDACTED, masked)
        count += n
    for card in find_card_numbers(masked):
        masked = masked.replace(card, REDACTED, 1)
        count += 1
    return masked, count


__all__ = [
    "REDACTED",
    "QuestionVerdict",
    "drop_tainted_chunks",
    "find_injection_matches",
    "redact_pii",
    "screen_question",
]
