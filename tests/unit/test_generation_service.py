"""Unit tests for the generation service and JSON parsing."""

from __future__ import annotations

import json

from app.generation.client import MockLLM
from app.generation.prompts import format_evidence
from app.generation.schemas import GroundingStatus
from app.generation.service import (
    GenerationService,
    _normalize_citation_id,
    extract_json,
    safe_parse_json,
)
from app.retrieval.models import RetrievalResult, RetrieverType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _candidate(
    chunk_id: str = "auth:0",
    text: str = "Access tokens expire after 60 minutes.",
    document_id: str = "auth",
    source: str = "auth.md",
    section: str | None = None,
    page_number: int | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=0.9,
        source=source,
        retriever=RetrieverType.HYBRID,
        rank=0,
        section=section,
        page_number=page_number,
    )

def _candidates() -> list[RetrievalResult]:
    return [
        _candidate("auth:0", "Access tokens expire after 60 minutes.", "auth"),
        _candidate("rate:0", "The rate limit is 1000 req/min.", "rate", section="Limits"),
    ]

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Tests for extract_json helper."""

    def test_bare_json(self) -> None:
        raw = '{"answer": "hello", "confidence": 0.9}'
        assert extract_json(raw) == raw

    def test_fenced_json(self) -> None:
        raw = 'Here is the result:\n```json\n{"answer": "hi"}\n```'
        assert extract_json(raw) == '{"answer": "hi"}'

    def test_fenced_without_language_tag(self) -> None:
        raw = "```\n{\"a\": 1}\n```"
        assert extract_json(raw) == '{"a": 1}'

    def test_extra_text_around_json(self) -> None:
        raw = "Before\n{" + '"answer": "test"}' + "\nAfter"
        assert extract_json(raw) == '{"answer": "test"}'

    def test_no_json_returns_none(self) -> None:
        assert extract_json("no json here") is None

    def test_recovers_truncated_answer_string(self) -> None:
        """A response cut off mid-answer is repaired into valid JSON."""
        raw = '{"answer": "The CV presents Siti Annisa Dahlan as a junior UI/UX'
        result = extract_json(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["answer"].startswith("The CV presents Siti Annisa")

    def test_recovers_truncated_nested_citations(self) -> None:
        """Unterminated nested containers are closed so the payload parses."""
        raw = (
            '{"answer": "Summary. [C1]", "citations": ['
            '{"citation_id": "[C1]", "chunk_id": "cv:0"'
        )
        result = extract_json(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["answer"] == "Summary. [C1]"
        assert parsed["citations"][0]["citation_id"] == "[C1]"

    def test_recovers_truncated_after_trailing_comma(self) -> None:
        """A dangling comma before truncation does not break the repair."""
        raw = '{"answer": "Done. [C1]", "citations": [],'
        result = extract_json(raw)
        assert result is not None
        assert json.loads(result)["answer"] == "Done. [C1]"

    def test_repair_does_not_mask_structural_corruption(self) -> None:
        """Genuinely broken JSON still fails rather than being silently fixed."""
        assert extract_json('{"answer": "x"]}') is None

class TestSafeParseJson:
    """Tests for safe_parse_json helper."""

    def test_valid_json(self) -> None:
        data = {"a": 1, "b": "test"}
        assert safe_parse_json(json.dumps(data)) == data

    def test_malformed_returns_none(self) -> None:
        assert safe_parse_json("{not json}") is None
        assert safe_parse_json("") is None
        assert safe_parse_json("null") is None

class TestNormalizeCitationId:
    """Tests for citation ID normalization."""

    def test_already_normal(self) -> None:
        assert _normalize_citation_id("[C1]") == "[C1]"

    def test_strips_whitespace(self) -> None:
        assert _normalize_citation_id("  [C2]  ") == "[C2]"

    def test_adds_brackets(self) -> None:
        assert _normalize_citation_id("C3") == "[C3]"
        assert _normalize_citation_id("C4]") == "[C4]"

    def test_mixed_case(self) -> None:
        assert _normalize_citation_id("[c1]") == "[c1]"

# ---------------------------------------------------------------------------
# GenerationService tests
# ---------------------------------------------------------------------------

class TestGenerationService:
    """Tests for the GenerationService."""

    def _make_service(self, mock: MockLLM) -> GenerationService:
        return GenerationService(client=mock)

    def test_generate_with_valid_json_answer(self) -> None:
        """Service parses a valid JSON response correctly."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "Tokens last 60 minutes. [C1]",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "auth:0",
                    "text": "Access tokens expire after 60 minutes.",
                }
            ],
            "confidence": 0.95,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("token lifetime?", _candidates())

        assert resp.answer == "Tokens last 60 minutes. [C1]"
        assert len(resp.citations) == 1
        assert resp.citations[0].citation_id == "[C1]"
        assert resp.confidence == 0.95
        assert resp.refused is False
        assert resp.model == "mock/test"

    def test_generate_with_refusal(self) -> None:
        """Service handles refusal responses correctly."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "I cannot answer this.",
            "citations": [],
            "confidence": 0.1,
            "refused": True,
            "refused_reason": "No supporting evidence found.",
        }))
        service = self._make_service(mock)
        resp = service.generate("unrelated question?", [])

        assert resp.refused is True
        assert resp.grounding_status == GroundingStatus.REFUSED
        assert resp.confidence == 0.1

    def test_generate_drops_unknown_citation_ids(self) -> None:
        """Citations with unknown IDs are silently dropped."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "Answer using [C99]",
            "citations": [
                {"citation_id": "[C99]", "chunk_id": "fake:0", "text": "fake text"}
            ],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", _candidates())

        assert len(resp.citations) == 0  # [C99] is not in our candidates
        # When all citations are dropped, grounding status is UNGROUNDED
        assert resp.grounding_status == GroundingStatus.UNGROUNDED

    def test_generate_generic_mode_sets_flag(self) -> None:
        """allow_generic=True flags the response without breaking grounding."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "Tokens last 60 minutes. [C1]",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "auth:0",
                    "text": "Access tokens expire after 60 minutes.",
                }
            ],
            "confidence": 0.95,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("token lifetime?", _candidates(), allow_generic=True)

        assert resp.generic is True
        assert resp.grounded is True
        assert resp.refused is False

    def test_generate_strict_mode_flag_defaults_false(self) -> None:
        """Without allow_generic the response is never flagged generic."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "Tokens last 60 minutes. [C1]",
            "citations": [
                {
                    "citation_id": "[C1]",
                    "chunk_id": "auth:0",
                    "text": "Access tokens expire after 60 minutes.",
                }
            ],
            "confidence": 0.95,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("token lifetime?", _candidates())

        assert resp.generic is False

    def test_generate_resolves_citation_to_candidate_despite_wrong_chunk_id(self) -> None:
        """A valid [Cn] marker resolves to the authoritative candidate.

        Citation markers map 1:1 to the retrieved candidate list, so the
        service trusts the marker — not the chunk_id/text the model echoes
        back (models often truncate or rephrase those fields). Unknown
        markers like [C99] are still dropped (see previous test).
        """
        mock = MockLLM(response_text=json.dumps({
            "answer": "Answer [C1]",
            "citations": [
                {"citation_id": "[C1]", "chunk_id": "unknown-chunk:99", "text": "text"}
            ],
            "confidence": 0.8,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", _candidates())

        assert len(resp.citations) == 1
        assert resp.citations[0].citation_id == "[C1]"
        assert resp.citations[0].chunk_id == "auth:0"
        assert resp.citations[0].text == "Access tokens expire after 60 minutes."

    def test_generate_deduplicates_repeated_citation_ids(self) -> None:
        """A repeated marker yields one citation, not duplicates.

        Models frequently echo the same marker several times. Keeping both
        produced two entries with an identical citation_id, which the UI
        rendered as duplicate rows under duplicate React keys.
        """
        mock = MockLLM(response_text=json.dumps({
            "answer": "Tokens expire. [C1] Rates are capped. [C1] See also [C2].",
            "citations": [
                {"citation_id": "[C1]", "chunk_id": "auth:0", "text": "first seen"},
                {"citation_id": "[C1]", "chunk_id": "auth:0", "text": "duplicate"},
                {"citation_id": "[C2]", "chunk_id": "rate:0", "text": "rate limit"},
                {"citation_id": "[C2]", "chunk_id": "rate:0", "text": "duplicate"},
            ],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", _candidates())

        ids = [c.citation_id for c in resp.citations]
        assert ids == ["[C1]", "[C2]"]
        assert len(ids) == len(set(ids))
        # First occurrence wins, and evidence still resolves from the candidate.
        assert resp.citations[0].text == "Access tokens expire after 60 minutes."

    def test_generate_handles_fenced_json(self) -> None:
        """Service correctly extracts JSON from fenced markdown."""
        mock = MockLLM(
            response_text='```json\n{"answer": "yes", "citations": [], '
            '"confidence": 0.9, "refused": false, "refused_reason": null}\n```',
        )
        service = self._make_service(mock)
        resp = service.generate("q?", [])
        assert resp.answer == "yes"

    def test_generate_handles_extra_text_around_json(self) -> None:
        """Service extracts JSON even with surrounding text."""
        mock = MockLLM(
            response_text='Here is the answer:\n'
            '{"answer": "parsed", "citations": [], "confidence": 0.7, '
            '"refused": false, "refused_reason": null}\nThanks!',
        )
        service = self._make_service(mock)
        resp = service.generate("q?", [])
        assert resp.answer == "parsed"

    def test_generate_returns_error_response_on_malformed_output(self) -> None:
        """Non-JSON output results in a safe refusal, not a crash."""
        mock = MockLLM(response_text="This is not JSON at all!!!")
        service = self._make_service(mock)
        resp = service.generate("q?", _candidates())

        assert resp.refused is True
        assert resp.grounding_status == GroundingStatus.REFUSED
        assert "JSON" in (resp.refused_reason or "")

    def test_generate_uses_temperature(self) -> None:
        """Temperature is passed through to the client."""
        mock = MockLLM(
            response_text=(
                '{"answer": "x", "citations": [], '
                '"confidence": 0.5, "refused": false, "refused_reason": null}'
            )
        )
        service = self._make_service(mock)
        service.generate("q?", [], temperature=0.3)

        assert mock.calls[0]["temperature"] == 0.3

    def test_confidence_normalized_to_0_1(self) -> None:
        """Confidence values outside [0, 1] are clamped."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "answer",
            "citations": [],
            "confidence": 1.5,  # out of range
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", [])
        assert resp.confidence == 1.0  # clamped to 1.0

    def test_confidence_negative_normalized(self) -> None:
        """Negative confidence is clamped to 0.0."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "answer",
            "citations": [],
            "confidence": -0.5,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", [])
        assert resp.confidence == 0.0

    def test_latency_is_recorded(self) -> None:
        """Response includes total_latency_ms."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "ok",
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", [])
        assert resp.total_latency_ms is not None
        assert resp.total_latency_ms >= 0

    def test_model_name_from_response(self) -> None:
        """model field is populated from the response."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "ok",
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        resp = service.generate("q?", [])
        assert resp.model == "mock/test"

    def test_calls_record_system_and_user_prompts(self) -> None:
        """Mock records the prompts for verification."""
        mock = MockLLM(response_text=json.dumps({
            "answer": "ok",
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refused_reason": None,
        }))
        service = self._make_service(mock)
        cands = _candidates()
        service.generate("What is the token lifetime?", cands)

        call = mock.calls[0]
        assert "system_prompt" in call
        assert "user_prompt" in call
        assert "What is the token lifetime?" in call["user_prompt"]
        assert "[C1]" in call["user_prompt"]  # evidence formatted

class TestEvidenceFormattingIntegration:
    """Verify evidence formatting with real RetrievalResult lists."""

    def test_format_evidence_with_section_and_page(self) -> None:
        cands = [
            _candidate("doc:5", "Page content here.", "doc", page_number=3, section="Chapter 1"),
        ]
        result = format_evidence(cands)
        assert "[C1]" in result
        assert "Chapter 1" in result
        assert "page: 3" in result
