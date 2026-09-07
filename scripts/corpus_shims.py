"""Shared text utilities for the AskMyDocs document corpus generator.

These helpers keep the generated markdown documents consistent: every
heading gets an anchor, every generated document carries front-matter
metadata, and mimicry of realistic technical documentation (tables,
admonitions, code blocks) is uniform across documents.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Heading helpers
# ---------------------------------------------------------------------------

_ANCHOR_STRIP_RE = re.compile(r"[^a-z0-9\- ]+")
_ANCHOR_SPACE_RE = re.compile(r"\s+")


def slugify(text: str) -> str:
    """Convert arbitrary heading text into an HTML-style anchor id.

    Example: ``"Token Expiry & Refresh"`` -> ``"token-expiry-refresh"``
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _ANCHOR_STRIP_RE.sub("", text)
    text = _ANCHOR_SPACE_RE.sub("-", text).strip("-")
    return text or "section"


def anchor(text: str) -> str:
    """Return the anchor link suffix for a heading, e.g. ``(#token-expiry)``."""
    return f"(#{slugify(text)})"


class Heading:
    """Small builder that renders a markdown heading with an anchor id."""

    __slots__ = ("level", "text")

    def __init__(self, level: int, text: str) -> None:
        self.level = level
        self.text = text

    def render(self) -> str:
        return f"{'#' * self.level} {self.text} {anchor(self.text)}\n\n"


def h2(text: str) -> str:
    return Heading(2, text).render()


def h3(text: str) -> str:
    return Heading(3, text).render()


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def para(text: str) -> str:
    """Render a single paragraph.

    Paragraphs are separated by a blank line so markdown renderers and the
    paragraph-aware chunker treat each paragraph as its own block.
    """
    return f"{text}\n\n"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table with a header row and separator."""
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n\n"


def code_block(language: str, code: str) -> str:
    """Render a fenced code block."""
    fence = "```"
    return f"{fence}{language}\n{code}\n{fence}\n\n"


def note_box(kind: str, text: str) -> str:
    """Render an Admonition-style box (Note / Warning / Tip)."""
    marker = {"note": "> **Note**", "tip": "> **Tip**", "warning": "> **Warning**"}.get(
        kind, "> **Note**"
    )
    return f"{marker}: {text}\n\n"


# ---------------------------------------------------------------------------
# Front matter / metadata
# ---------------------------------------------------------------------------


def front_matter(*, doc_id: str, title: str, version: str, module: str) -> str:
    """Render YAML-ish front matter consistent with project metadata keys."""
    today = date.today().isoformat()
    return (
        "---\n"
        f"doc_id: {doc_id}\n"
        f"title: {title}\n"
        f"version: {version}\n"
        f"module: {module}\n"
        f"last_updated: {today}\n"
        "---\n\n"
    )


def write_doc(path: Path, content: str) -> None:
    """Write a text document, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")