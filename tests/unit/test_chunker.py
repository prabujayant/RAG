"""Tests for the paragraph-aware chunker (Phase 1 determinism)."""

from __future__ import annotations

import pytest
from app.ingestion.chunker import (
    Chunker,
    ChunkingStrategy,
    content_hash,
    estimate_tokens,
)

SAMPLE_TEXT = """---
doc_id: sample
title: Sample
---

## Section One (#section-one)

The first paragraph discusses tokens that expire after 60 minutes.

A second paragraph continues the discussion with more granular detail.

## Section Two (#section-two)

* bullet item one
* bullet item two

```bash
echo hello
```
"""

@pytest.fixture
def chunker() -> Chunker:
    return Chunker(strategy=ChunkingStrategy.PARAGRAPH, chunk_size=512, chunk_overlap=64)

def test_chunk_ids_are_deterministic(chunker: Chunker) -> None:
    a = chunker.chunk_text(SAMPLE_TEXT, document_id="doc-x", document_name="Doc X", source="test")
    b = chunker.chunk_text(SAMPLE_TEXT, document_id="doc-x", document_name="Doc X", source="test")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.content_hash for c in a] == [c.content_hash for c in b]
    assert [c.text for c in a] == [c.text for c in b]

def test_chunk_ids_follow_doc_index_pattern(chunker: Chunker) -> None:
    chunks = chunker.chunk_text(SAMPLE_TEXT, document_id="doc-x", document_name="Doc X", source="test")
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"doc-x:{i}"
        assert c.index == i

def test_section_boundaries_start_new_chunks(chunker: Chunker) -> None:
    chunks = chunker.chunk_text(SAMPLE_TEXT, document_id="doc-x", document_name="Doc X", source="test")
    sections = [c.section for c in chunks]
    # Section headings (including their anchors) should appear as section values.
    assert "Section One (#section-one)" in sections
    assert "Section Two (#section-two)" in sections
    # The first chunk is the front matter; sections start at index 1.
    assert len(chunks) >= 3

def test_metadata_preserved(chunker: Chunker) -> None:
    chunks = chunker.chunk_text(
        SAMPLE_TEXT,
        document_id="doc-x",
        document_name="Doc X",
        source="pdf/test.pdf",
        page_number=3,
        section=None,
    )
    for c in chunks:
        assert c.document_id == "doc-x"
        assert c.document_name == "Doc X"
        assert c.source == "pdf/test.pdf"
        assert c.page_number == 3
        assert c.content_hash == content_hash(c.text)

def test_fixed_size_chunker_produces_multiple_chunks() -> None:
    fixed = Chunker(strategy=ChunkingStrategy.FIXED, chunk_size=50, chunk_overlap=10)
    chunks = fixed.chunk_text(
        "This is a long sentence that continues. And here is another one that keeps going on and on. "
        "A third sentence for good measure, with plenty of words to force multiple chunks. "
        "Finally a trailing sentence to finish the text."
        * 3,
        document_id="doc-x",
        document_name="Doc X",
        source="test",
    )
    assert len(chunks) > 1

def test_estimate_tokens() -> None:
    assert estimate_tokens("a" * 400) == 100

def test_content_hash_stable() -> None:
    assert content_hash("hello world") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello wrld")

def test_chunk_payload_shape(chunker: Chunker) -> None:
    chunks = chunker.chunk_text(SAMPLE_TEXT, document_id="doc-x", document_name="Doc X", source="test")
    payload = chunks[0].to_retrieval_payload()
    assert payload["chunk_id"].startswith("doc-x:")
    assert payload["document_id"] == "doc-x"
    assert "text" in payload
    assert "content_hash" in payload
