"""Tests that the mixed-format corpus tree has the required shape."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_PDFS = [
    "authentication-guide.pdf",
    "api-reference.pdf",
    "security-guide.pdf",
    "deployment-guide.pdf",
    "troubleshooting-guide.pdf",
]

def test_pdf_directory_exists(corpus_dir: Path) -> None:
    assert (corpus_dir / "pdf").is_dir()

def test_required_pdfs_present(corpus_dir: Path) -> None:
    pdfs = {p.name for p in (corpus_dir / "pdf").glob("*.pdf")}
    for required in REQUIRED_PDFS:
        assert required in pdfs, f"missing {required}"

def test_all_format_directories_present(corpus_dir: Path) -> None:
    for fmt in ("markdown", "pdf", "docx", "html"):
        assert (corpus_dir / fmt).is_dir()

def test_markdown_documents_exist(corpus_dir: Path) -> None:
    md_files = list((corpus_dir / "markdown").glob("*.md"))
    assert len(md_files) >= 20

def test_docx_and_html_exist(corpus_dir: Path) -> None:
    assert len(list((corpus_dir / "docx").glob("*.docx"))) >= 3
    assert len(list((corpus_dir / "html").glob("*.html"))) >= 3

def test_manifest_matches_files(corpus_dir: Path) -> None:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    # Every manifest-relative path must exist.
    for rel in manifest.values():
        assert (corpus_dir / rel).exists(), f"manifest points to missing {rel}"

def test_markdown_docs_have_real_structure(corpus_dir: Path) -> None:
    """Documents must contain headings, paragraphs and technical content."""
    sample = corpus_dir / "markdown" / "authentication-guide.md"
    content = sample.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert "## " in content
    assert "amsk_" in content
    assert "OAuth 2.0" in content
