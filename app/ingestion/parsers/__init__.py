"""Ingestion parsers: PDF, DOCX, HTML, Markdown."""

from __future__ import annotations

from app.ingestion.parsers.docx import extract_docx
from app.ingestion.parsers.html import extract_html
from app.ingestion.parsers.markdown import extract_markdown
from app.ingestion.parsers.pdf import extract_pdf

__all__ = ["extract_pdf", "extract_docx", "extract_html", "extract_markdown"]
