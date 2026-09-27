"""Unit tests for :mod:`app.evaluation.chunk_refs`.

These tests are pure — they never touch a database. They pin the behaviour
that makes retrieval metrics meaningful: golden-dataset slug references must
expand to the concrete chunk ids the pipeline emits, and expansions that
cannot exist (out-of-range indices for a given rendering) must be dropped
rather than injected as phantom references.
"""

from __future__ import annotations

import pytest
from app.evaluation.chunk_refs import ChunkRefResolver

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_pairs_groups_multiple_doc_ids_per_slug():
    resolver = ChunkRefResolver.from_pairs(
        [
            ("authentication-guide", "aaa"),
            ("authentication-guide", "bbb"),
            ("api-reference", "ccc"),
        ]
    )
    assert resolver.slug_to_doc_ids["authentication-guide"] == ["aaa", "bbb"]
    assert resolver.slug_to_doc_ids["api-reference"] == ["ccc"]


def test_from_pairs_deduplicates_repeated_doc_ids():
    resolver = ChunkRefResolver.from_pairs(
        [("doc", "aaa"), ("doc", "aaa"), ("doc", "bbb")]
    )
    assert resolver.slug_to_doc_ids["doc"] == ["aaa", "bbb"]


def test_empty_resolver_reports_is_empty():
    assert ChunkRefResolver().is_empty is True
    assert ChunkRefResolver.from_pairs([("doc", "aaa")]).is_empty is False


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_expands_slug_to_all_renderings():
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa"), ("doc", "bbb")])
    assert resolver.resolve("doc:3") == ["aaa:3", "bbb:3"]


def test_resolve_unknown_slug_returns_empty():
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa")])
    assert resolver.resolve("missing:0") == []


def test_resolve_without_index_returns_empty():
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa")])
    assert resolver.resolve("doc") == []


def test_resolve_non_numeric_index_returns_empty():
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa")])
    assert resolver.resolve("doc:abc") == []


def test_resolve_all_flattens_and_deduplicates():
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa"), ("doc", "bbb")])
    resolved = resolver.resolve_all(["doc:0", "doc:0", "doc:1"])
    assert resolved == {"aaa:0", "bbb:0", "aaa:1", "bbb:1"}


# ---------------------------------------------------------------------------
# Range filtering (the phantom-reference guard)
# ---------------------------------------------------------------------------


def test_out_of_range_index_is_dropped():
    """A 2-chunk PDF cannot have chunk 2, so that expansion must be dropped."""
    resolver = ChunkRefResolver.from_pairs(
        [("doc", "md"), ("doc", "pdf")],
        chunk_counts={"md": 5, "pdf": 2},
    )
    assert resolver.resolve("doc:2") == ["md:2"]
    assert resolver.resolve("doc:1") == ["md:1", "pdf:1"]


def test_index_at_upper_boundary_is_excluded():
    resolver = ChunkRefResolver.from_pairs([("doc", "d")], chunk_counts={"d": 3})
    assert resolver.resolve("doc:2") == ["d:2"]
    assert resolver.resolve("doc:3") == []


def test_document_with_no_chunks_is_never_resolved():
    resolver = ChunkRefResolver.from_pairs(
        [("doc", "empty")], chunk_counts={"empty": 0}
    )
    assert resolver.resolve("doc:0") == []


def test_document_missing_from_counts_is_never_resolved():
    resolver = ChunkRefResolver.from_pairs(
        [("doc", "known"), ("doc", "unknown")], chunk_counts={"known": 3}
    )
    assert resolver.resolve("doc:0") == ["known:0"]


def test_no_chunk_counts_disables_range_filtering():
    """Without counts we cannot know the range, so nothing is dropped."""
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa")])
    assert resolver.resolve("doc:99") == ["aaa:99"]


# ---------------------------------------------------------------------------
# Unknown-slug reporting
# ---------------------------------------------------------------------------


def test_unknown_slugs_reports_missing_documents():
    resolver = ChunkRefResolver.from_pairs([("known", "aaa")])
    assert resolver.unknown_slugs(["known:0", "bogus:1"]) == {"bogus"}


def test_unknown_slugs_treats_indexless_ref_as_unknown():
    resolver = ChunkRefResolver.from_pairs([("known", "aaa")])
    assert resolver.unknown_slugs(["known"]) == {"known"}


@pytest.mark.parametrize("ref", ["doc:0", "doc:12"])
def test_resolve_is_stable_for_valid_refs(ref):
    resolver = ChunkRefResolver.from_pairs([("doc", "aaa")], chunk_counts={"aaa": 20})
    assert resolver.resolve(ref) == [f"aaa:{ref.split(':')[1]}"]
