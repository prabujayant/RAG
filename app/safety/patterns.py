"""Safety pattern library: prompt-injection, disallowed content, and PII.

All patterns are deterministic regexes — no LLM calls, so screening is free
and cannot be prompt-engineered around from inside the screened text. Kept
deliberately tight (high precision): a false block refuses a legitimate
question, which is worse than the narrow miss of an exotic phrasing.
"""

from __future__ import annotations

import re

# --- Prompt injection -------------------------------------------------------
# Instruction-takeover phrasings smuggled into uploaded documents or questions.
# Each entry is (name, compiled pattern); names surface in metrics/logs.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in [
        (
            "ignore-instructions",
            r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|directives?)",
        ),
        ("disregard-prior", r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\b"),
        (
            "forget-training",
            r"forget\s+(all\s+|your\s+|previous\s+)?(instructions?|training|rules|directives?)",
        ),
        ("role-takeover", r"you\s+are\s+now\s+"),
        ("new-instructions", r"new\s+(system\s+)?instructions?\s*:"),
        (
            "reveal-prompt",
            r"reveal\s+(your\s+)?((system|initial|hidden|secret)\s+){1,3}(prompt|instructions?|directives?)",
        ),
        ("bypass-safety", r"bypass\s+(your\s+)?(safety|filters?|restrictions?|guardrails?|refusal)"),
        ("jailbreak", r"\bjailbreak\b|DAN\s+mode|do\s+anything\s+now"),
        ("override-restrictions", r"override\s+(your\s+)?(instructions?|safety|restrictions?)"),
        ("fake-system-tag", r"\[\s*system\s*\]"),
        ("developer-override", r"developer\s+mode\s+(enabled|on)\b"),
        (
            "instruction-delimiter",
            r"(?m)^\s*(###\s*new\s+instructions?|---\s*end\s+of\s+(context|document))\s*$",
        ),
    ]
)

# --- Disallowed question content --------------------------------------------
# High-precision only: how-tos for wrongdoing where the *request itself* is
# the harm, regardless of what any document says.
DISALLOWED_QUESTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in [
        (
            "explosives",
            r"(make|build|create|construct)\s+(a\s+)?(bomb|explosive|pipe\s*bomb|molotov|bioweapon|chemical\s+weapon)",
        ),
        ("drug-manufacture", r"manufacture\s+(meth|fentanyl|heroin|cocaine|explosives?)"),
        ("malware", r"(write|create|develop)\s+(malware|ransomware|spyware|a\s+virus|keylogger)"),
        ("hacking", r"how\s+to\s+hack\s+(into|a\b)|ddos\s+attack|zero[\s-]?day\s+exploit|rce\s+exploit"),
        ("self-harm", r"how\s+to\s+(commit\s+suicide|kill\s+myself|self[\s-]?harm)"),
        ("csam", r"child\s+(porn|sexual)|csam|\bloli\b"),
        ("violent-wrongdoing", r"how\s+to\s+(rob\s+a\s+bank|shoplift|counterfeit|forge\s+(a\s+)?passport)"),
    ]
)

# --- PII --------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d{1,3}[\s().-]?\d{3}[\s().-]?\d{3}[\s().-]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
API_KEY_RE = re.compile(
    r"sk-[A-Za-z0-9\-_]{8,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]+|gh[pousr]_[A-Za-z0-9_]+"
)
CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum: separates real card numbers from random digit runs."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_card_numbers(text: str) -> list[str]:
    """Return digit runs that pass the Luhn check (likely card numbers)."""
    hits = []
    for match in CARD_CANDIDATE_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            hits.append(match.group(0))
    return hits


__all__ = [
    "API_KEY_RE",
    "CARD_CANDIDATE_RE",
    "DISALLOWED_QUESTION_PATTERNS",
    "EMAIL_RE",
    "INJECTION_PATTERNS",
    "PHONE_RE",
    "SSN_RE",
    "find_card_numbers",
]
