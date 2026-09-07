"""Unit tests for the evidence selector."""

from __future__ import annotations

from app.retrieval.evidence import EvidenceSelector
from app.retrieval.models import RetrievalResult, RetrieverType


def _result(
    chunk_id: str,
    document_id: str,
    text: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        source=f"{document_id}.md",
        retriever=RetrieverType.HYBRID,
        rank=0,
    )

class TestEvidenceSelector:
    def test_respects_final_context_k(self) -> None:
        """No more than final_context_k chunks are returned."""
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 3, "max_context_tokens": 100000}
        )
        candidates = [_result(f"doc{i}:0", "doc", "x" * 100, 1.0 - i * 0.1) for i in range(10)]
        selected = selector.select(candidates)
        assert len(selected) == 3

    def test_respects_token_budget(self) -> None:
        """Total estimated tokens stay within max_context_tokens."""
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 10, "max_context_tokens": 50}
        )
        # Each "x" is ~1/4 token, so 200 chars ~ 50 tokens
        candidates = [
            _result("a:0", "a", "x" * 200, 1.0),
            _result("b:0", "b", "x" * 200, 0.9),
            _result("c:0", "c", "x" * 200, 0.8),
        ]
        selected = selector.select(candidates)
        total_tokens = sum(len(c.text) / 4 for c in selected)
        assert total_tokens <= 50

    def test_deduplicates_by_chunk_id(self) -> None:
        """Same chunk_id appearing twice is included only once."""
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 5, "max_context_tokens": 100000}
        )
        candidates = [
            _result("a:0", "a", "text a", 0.9),
            _result("a:0", "a", "text a duplicate", 0.85),
            _result("b:0", "b", "text b", 0.8),
        ]
        selected = selector.select(candidates)
        assert len(selected) == 2
        chunk_ids = [c.chunk_id for c in selected]
        assert "a:0" in chunk_ids
        assert "b:0" in chunk_ids

    def test_empty_candidates_returns_empty(self) -> None:
        """Empty list returns empty list."""
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 5, "max_context_tokens": 1000}
        )
        assert selector.select([]) == []

    def test_top_k_override(self) -> None:
        """Passing top_k overrides the configured final_context_k."""
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 10, "max_context_tokens": 100000}
        )
        candidates = [_result(f"d{i}:0", f"d{i}", "x", 1.0 - i * 0.01) for i in range(20)]
        selected = selector.select(candidates, top_k=2)
        assert len(selected) == 2

    def test_deterministic_output(self) -> None:
        """Calling select twice with the same input returns the same order."""
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 5, "max_context_tokens": 100000}
        )
        candidates = [
            _result("a:0", "a", "text", 0.9),
            _result("b:0", "b", "text", 0.8),
            _result("c:0", "c", "text", 0.7),
        ]
        first = selector.select(candidates)
        second = selector.select(candidates)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_metadata_preserved(self) -> None:
        """Retrieved metadata (source, page_number, section) is preserved."""
        result = RetrievalResult(
            chunk_id="a:0",
            document_id="a",
            text="some text",
            score=0.9,
            source="guide.md",
            page_number=5,
            section="Intro",
            retriever=RetrieverType.HYBRID,
            rank=0,
        )
        selector = EvidenceSelector(settings=None)  # type: ignore[arg-type]
        selector._settings = selector._settings.model_copy(
            update={"final_context_k": 5, "max_context_tokens": 100000}
        )
        selected = selector.select([result])
        assert selected[0].source == "guide.md"
        assert selected[0].page_number == 5
        assert selected[0].section == "Intro"
