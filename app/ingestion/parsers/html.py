"""HTML parser using beautifulsoup4."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.ingestion.parsers._shared import ParsedDocument, _normalize_text


def extract_html(file_path: str | Path) -> ParsedDocument:
    """Extract visible text from an HTML file, stripping scripts and styles.

    Returns
    -------
    ParsedDocument
        With ``text`` as stripped visible text and ``metadata["title"]``.
    """
    with open(str(file_path), encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Remove noise elements.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    # Collect headings + paragraphs.
    parts: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        t = tag.get_text(separator=" ", strip=True)
        if t:
            parts.append(_normalize_text(t))
    return ParsedDocument(
        text="\n\n".join(parts),
        metadata={"title": title},
    )
