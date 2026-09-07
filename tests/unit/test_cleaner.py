"""Tests for text cleaning utilities."""

from __future__ import annotations

from app.ingestion.cleaner import clean_text


class TestCleanText:
    def test_normal_text_unchanged(self) -> None:
        text = "This is a normal sentence with standard ASCII characters."
        result = clean_text(text)
        assert result == text

    def test_strips_control_characters(self) -> None:
        # C0 controls except \n \r \t should be stripped
        text = "Hello\x07World\x1ETest"
        result = clean_text(text)
        assert "\x07" not in result
        assert "\x1E" not in result
        assert "Hello" in result
        assert "World" in result

    def test_preserves_newlines(self) -> None:
        text = "Line1\nLine2\rLine3"
        result = clean_text(text)
        assert "\n" in result
        assert "Line1" in result
        assert "Line2" in result

    def test_normalizes_unicode_quotes(self) -> None:
        # Smart quotes to ASCII
        text = "\u2018single\u2019 and \u201Cdouble\u201D quotes"
        result = clean_text(text)
        assert "'" in result
        assert '"' in result
        assert "\u2018" not in result
        assert "\u2019" not in result

    def test_normalizes_dashes(self) -> None:
        text = "en\u2013dash and em\u2014dash"
        result = clean_text(text)
        assert "\u2013" not in result  # en dash -> --
        assert "\u2014" not in result  # em dash -> --
        assert "--" in result

    def test_removes_non_breaking_space(self) -> None:
        text = "Word\u00a0with\u00a0NBSP"
        result = clean_text(text)
        assert "\u00a0" not in result
        assert "with" in result

    def test_collapses_whitespace(self) -> None:
        text = "Word    multiple   spaces   here"
        result = clean_text(text)
        assert "  " not in result.replace("--", "-")  # -- is valid, but no double spaces
        assert "multiple" in result

    def test_collapse_newlines(self) -> None:
        text = "Para1\n\n\n\n\nPara2"
        result = clean_text(text)
        # No more than 2 consecutive newlines
        assert "\n\n\n" not in result
        assert "Para1" in result
        assert "Para2" in result

    def test_trims_edges(self) -> None:
        text = "   \n\n   Hello World   \n\n   "
        result = clean_text(text)
        assert result.startswith("Hello")
        assert result.endswith("World")

    def test_removes_page_number_lines(self) -> None:
        text = "Content\n123\nMore content"
        result = clean_text(text)
        # Lines that are just numbers should be stripped
        assert "123" not in result

    def test_removes_page_break_lines(self) -> None:
        text = "Content\n---\nMore content"
        result = clean_text(text)
        assert "---" not in result

    def test_removes_asterisk_lines(self) -> None:
        text = "Content\n***\nMore content"
        result = clean_text(text)
        assert "***" not in result

    def test_removes_underscore_lines(self) -> None:
        text = "Content\n___\nMore content"
        result = clean_text(text)
        assert "___" not in result

    def test_removes_header_boilerplate(self) -> None:
        text = "AskMyDocs | Some Page\nActual content\nAskMyDocs : Guide\nMore content"
        result = clean_text(text)
        assert "AskMyDocs" not in result
        assert "Actual content" in result
        assert "More content" in result

    def test_removes_confidential_boilerplate(self) -> None:
        text = "Company Confidential\nActual content\nConfidential | Secret\nMore content"
        result = clean_text(text)
        assert "Confidential" not in result
        assert "Company" not in result or "Actual" in result  # Company in content OK

    def test_unicode_normalization(self) -> None:
        # NFC vs NFKC - should normalize to NFKC
        text = "café"
        result = clean_text(text)
        assert "café" in result


class TestCleanTextEdgeCases:
    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_only_whitespace(self) -> None:
        result = clean_text("   \n\t\n   ")
        assert result == ""

    def test_only_control_chars(self) -> None:
        result = clean_text("\x00\x01\x02\x7f")
        assert result == ""

    def test_only_boilerplate(self) -> None:
        text = "AskMyDocs | Page 1\n-----\n***"
        result = clean_text(text)
        assert result == ""

    def test_mixed_language(self) -> None:
        text = "Hello 你好Bonjour    spaces"
        result = clean_text(text)
        assert "Hello" in result
        assert "spaces" in result

    def test_long_text(self) -> None:
        # Ensure no max length issues
        text = "word " * 10000
        result = clean_text(text)
        assert "word" in result

    def test_no_false_positive_on_dashes_in_content(self) -> None:
        # Em dash in actual content should be preserved
        text = "It's a useful feature—and should work."
        result = clean_text(text)
        assert "feature" in result
        assert "should work" in result
