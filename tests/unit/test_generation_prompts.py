"""Unit tests for prompt templates."""

from __future__ import annotations

from app.generation.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    build_prompts,
    format_evidence,
)
from app.retrieval.models import RetrievalResult, RetrieverType


class TestFormatEvidence:
    """Tests for evidence formatting."""

    def test_empty_candidates(self) -> None:
        """No candidates yields 'No evidence' placeholder."""
        result = format_evidence([])
        assert result == "No evidence provided."

    def test_single_candidate(self) -> None:
        """Single candidate is formatted with [C1] prefix."""
        candidates = [
            RetrievalResult(
                chunk_id="auth:0",
                document_id="auth",
                text="Access tokens expire after 60 minutes.",
                score=0.9,
                source="auth.md",
                retriever=RetrieverType.HYBRID,
                rank=0,
            ),
        ]
        result = format_evidence(candidates)
        assert "[C1]" in result
        assert "auth" in result
        assert "Access tokens expire after 60 minutes." in result

    def test_multiple_candidates(self) -> None:
        """Multiple candidates get sequential citation IDs."""
        candidates = [
            RetrievalResult(
                chunk_id="auth:0",
                document_id="auth",
                text="First text.",
                score=0.9,
                source="auth.md",
                retriever=RetrieverType.HYBRID,
                rank=0,
            ),
            RetrievalResult(
                chunk_id="rate:1",
                document_id="rate",
                text="Second text.",
                score=0.8,
                source="rate.md",
                retriever=RetrieverType.HYBRID,
                rank=1,
                section="Limits",
                page_number=3,
            ),
        ]
        result = format_evidence(candidates)
        assert "[C1]" in result
        assert "[C2]" in result
        assert "section: Limits" in result
        assert "page: 3" in result

    def test_document_id_fallback_to_source(self) -> None:
        """document_id falls back to source when absent."""
        candidates = [
            RetrievalResult(
                chunk_id="x:0",
                document_id="",  # empty
                text="text",
                score=0.9,
                source="fallback.md",
                retriever=RetrieverType.VECTOR,
                rank=0,
            ),
        ]
        result = format_evidence(candidates)
        assert "fallback.md" in result

class TestBuildPrompts:
    """Tests for prompt construction."""

    def test_system_prompt_includes_rules(self) -> None:
        """System prompt contains grounding rules."""
        sys, _ = build_prompts("How do I authenticate?", [])
        assert "Answer based ONLY on the supplied evidence" in sys
        assert "[C1]" in sys  # references citation format
        assert "refusal" in sys.lower()

    def test_user_prompt_includes_question(self) -> None:
        """User prompt contains the question."""
        _, usr = build_prompts("What is the token lifetime?", [])
        assert "What is the token lifetime?" in usr

    def test_user_prompt_includes_evidence(self) -> None:
        """User prompt includes formatted evidence."""
        candidates = [
            RetrievalResult(
                chunk_id="auth:0",
                document_id="auth",
                text="JWT lifetime is 3600s.",
                score=0.9,
                source="auth.md",
                retriever=RetrieverType.HYBRID,
                rank=0,
            ),
        ]
        _, usr = build_prompts("token lifetime?", candidates)
        assert "[C1]" in usr
        assert "JWT lifetime is 3600s." in usr

    def test_empty_evidence_in_user_prompt(self) -> None:
        """Empty evidence is indicated in the user prompt."""
        _, usr = build_prompts("question", [])
        assert "No evidence" in usr

class TestPromptContentRequirements:
    """Verify prompts enforce all spec requirements."""

    def test_system_prompt_requires_json_format(self) -> None:
        """System prompt instructs the model to return JSON."""
        assert '"answer"' in SYSTEM_PROMPT_TEMPLATE
        assert '"citations"' in SYSTEM_PROMPT_TEMPLATE
        assert '"confidence"' in SYSTEM_PROMPT_TEMPLATE
        assert '"refused"' in SYSTEM_PROMPT_TEMPLATE

    def test_system_prompt_forbids_fabrication(self) -> None:
        """System prompt explicitly forbids inventing facts."""
        assert "Do NOT invent" in SYSTEM_PROMPT_TEMPLATE
        assert "Only use citation IDs that appear" in SYSTEM_PROMPT_TEMPLATE

    def test_user_prompt_structured_format(self) -> None:
        """User prompt has the ## Evidence / ## Question structure."""
        assert "## Evidence" in USER_PROMPT_TEMPLATE
        assert "## Question" in USER_PROMPT_TEMPLATE
        assert "Produce your JSON response now." in USER_PROMPT_TEMPLATE

class TestBuildPromptsGeneric:
    """Tests for the generic-mode prompt variant."""

    def test_strict_contract_forbids_invention(self) -> None:
        """Default (strict) prompt tells the model to refuse, not invent."""
        sys, _ = build_prompts("token lifetime?", [])
        assert "Do NOT make up an answer" in sys

    def test_generic_contract_permits_general_knowledge(self) -> None:
        """Generic prompt allows general knowledge instead of refusal."""
        sys, _ = build_prompts("token lifetime?", [], allow_generic=True)
        assert "general knowledge" in sys
        assert "Do NOT make up an answer" not in sys

    def test_generic_prompt_keeps_json_contract(self) -> None:
        """Generic prompt still demands the same JSON response format."""
        sys, _ = build_prompts("token lifetime?", [], allow_generic=True)
        assert '"answer"' in sys
        assert '"citations"' in sys
        assert '"refused"' in sys
