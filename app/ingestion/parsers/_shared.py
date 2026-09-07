"""Shared types and utilities for all parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    """Result returned by every parser.

    Attributes
    ----------
    text:
        Normalised plain-text content ready for chunking.
    metadata:
        Arbitrary key-value metadata collected during parsing (e.g. page count,
        title, author). Contents are format-specific.
    """

    text: str
    metadata: dict


# --------------------------------------------------------------------------  .
# Whitespace normalisation used by all parsers.
# --------------------------------------------------------------------------  .

_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _normalize_text(text: str) -> str:
    """Collapse excessive whitespace; trim trailing newlines."""
    text = text.replace("\r\n", "\n")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
