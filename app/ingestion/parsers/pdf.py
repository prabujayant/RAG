"""PDF parser using pypdf."""

from __future__ import annotations

from pathlib import Path

import pypdf

from app.ingestion.parsers._shared import ParsedDocument, _normalize_text


def extract_pdf(file_path: str | Path) -> ParsedDocument:
    """Extract text and metadata from a PDF file.

    Returns
    -------
    ParsedDocument
        With ``text`` containing the full document text (pages joined by ``\\n``)
        and ``metadata["page_count"]`` set.
    """
    reader = pypdf.PdfReader(str(file_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i}]\n{_normalize_text(text)}")
    return ParsedDocument(
        text="\n\n".join(pages),
        metadata={
            "page_count": len(reader.pages),
            "author": reader.metadata.author if reader.metadata else None,
        },
    )
