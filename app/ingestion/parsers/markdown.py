"""Markdown parser — strips front matter, normalises whitespace."""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.parsers._shared import ParsedDocument, _normalize_text

# Matches YAML front matter: lines between --- markers at top of file.
_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_inline_code_re = re.compile(r"`([^`]+)`")


def extract_markdown(file_path: str | Path) -> ParsedDocument:
    """Read a markdown file and return its plain-text content.

    Strips YAML front matter and code fences so the chunker only sees prose.

    Returns
    -------
    ParsedDocument
    """
    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    # Strip front matter.
    raw = _FRONT_MATTER_RE.sub("", raw)
    # Collapse code fences to a single placeholder line (keep structure, drop noise).
    raw = _CODE_FENCE_RE.sub("[code block]", raw)
    text = _normalize_text(raw)
    return ParsedDocument(text=text, metadata={})
