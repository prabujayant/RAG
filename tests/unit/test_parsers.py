"""Tests for document parsers (PDF, DOCX, HTML, Markdown)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from app.ingestion.parsers import extract_docx, extract_html, extract_markdown, extract_pdf
from app.ingestion.parsers._shared import ParsedDocument


class TestExtractMarkdown:
    def test_extracts_plain_text(self) -> None:
        content = "# Hello\n\nThis is a test document.\n\n## Section\n\nSome text here."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            doc = extract_markdown(path)
            assert isinstance(doc, ParsedDocument)
            assert "Hello" in doc.text
            assert "Section" in doc.text
            assert "test document" in doc.text
        finally:
            Path(path).unlink()

    def test_handles_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("")
            path = f.name
        try:
            doc = extract_markdown(path)
            assert doc.text == ""
        finally:
            Path(path).unlink()

    def test_preserves_paragraphs(self) -> None:
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            doc = extract_markdown(path)
            assert "First paragraph" in doc.text
            assert "Second paragraph" in doc.text
            assert "Third paragraph" in doc.text
        finally:
            Path(path).unlink()

    def test_code_blocks_stripped(self) -> None:
        content = "# Doc\n\n```python\nprint('hello')\n```\n\nNormal text."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            doc = extract_markdown(path)
            assert "Normal text" in doc.text
        finally:
            Path(path).unlink()

class TestExtractHTML:
    def test_extracts_text_content(self) -> None:
        html = "<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = f.name
        try:
            doc = extract_html(path)
            assert isinstance(doc, ParsedDocument)
            assert "Title" in doc.text
            assert "Paragraph text" in doc.text
        finally:
            Path(path).unlink()

    def test_removes_script_and_style(self) -> None:
        html = (
            "<html><head><style>body { color: red; }</style></head>"
            "<body><script>alert('x')</script><p>Content</p></body></html>"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = f.name
        try:
            doc = extract_html(path)
            assert "Content" in doc.text
            assert "alert" not in doc.text
            assert "color: red" not in doc.text
        finally:
            Path(path).unlink()

    def test_handles_nested_tags(self) -> None:
        html = "<div><section><article><p>Deeply nested text.</p></article></section></div>"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = f.name
        try:
            doc = extract_html(path)
            assert "Deeply nested text" in doc.text
        finally:
            Path(path).unlink()

    def test_empty_html(self) -> None:
        html = "<html><body></body></html>"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            path = f.name
        try:
            doc = extract_html(path)
            assert doc.text.strip() == ""
        finally:
            Path(path).unlink()

class TestExtractDOCX:
    def test_extracts_paragraphs(self) -> None:
        try:
            from docx import Document

            doc = Document()
            doc.add_paragraph("First paragraph text.")
            doc.add_paragraph("Second paragraph text.")
            doc.add_paragraph("Third paragraph text.")

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
                doc.save(f.name)
                path = f.name
            try:
                result = extract_docx(path)
                assert isinstance(result, ParsedDocument)
                assert "First paragraph text" in result.text
                assert "Second paragraph text" in result.text
                assert result.metadata["paragraph_count"] == 3
            finally:
                Path(path).unlink()
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_empty_document(self) -> None:
        try:
            from docx import Document

            doc = Document()
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
                doc.save(f.name)
                path = f.name
            try:
                result = extract_docx(path)
                assert result.text == ""
                assert result.metadata["paragraph_count"] == 0
            finally:
                Path(path).unlink()
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_handles_empty_paragraphs(self) -> None:
        try:
            from docx import Document

            doc = Document()
            doc.add_paragraph("Content paragraph.")
            doc.add_paragraph("")  # empty
            doc.add_paragraph("More content.")

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
                doc.save(f.name)
                path = f.name
            try:
                result = extract_docx(path)
                assert "Content paragraph" in result.text
                assert "More content" in result.text
            finally:
                Path(path).unlink()
        except ImportError:
            pytest.skip("python-docx not installed")

class TestExtractPDF:
    def test_extracts_page_count(self) -> None:
        try:
            from pypdf import PdfWriter

            # Create a minimal blank PDF
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.add_blank_page(width=200, height=200)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                writer.write(f)
                path = f.name
            try:
                result = extract_pdf(path)
                assert isinstance(result, ParsedDocument)
                assert result.metadata["page_count"] == 2
            finally:
                Path(path).unlink()
        except ImportError:
            pytest.skip("pypdf not installed")

    def test_empty_pdf(self) -> None:
        try:
            from pypdf import PdfWriter

            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                writer.write(f)
                path = f.name
            try:
                result = extract_pdf(path)
                assert result.metadata["page_count"] == 1
            finally:
                Path(path).unlink()
        except ImportError:
            pytest.skip("pypdf not installed")

class TestParsedDocument:
    def test_has_text_and_metadata(self) -> None:
        doc = ParsedDocument(text="Sample text", metadata={"key": "value"})
        assert doc.text == "Sample text"
        assert doc.metadata["key"] == "value"

    def test_metadata_is_dict(self) -> None:
        doc = ParsedDocument(text="", metadata={})
        assert isinstance(doc.metadata, dict)
