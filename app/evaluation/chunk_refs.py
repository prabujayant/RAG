"""Resolve golden-dataset chunk references to the ids the pipeline actually emits.

Why this exists
---------------
The golden dataset (``evals/dataset/golden.jsonl``) is authored by humans and
references evidence with *readable* chunk ids such as ``authentication-guide:2``
— a document slug plus a chunk index.

The ingestion pipeline, however, assigns document ids as
``sha1(canonical_path)[:32]`` (see ``app.ingestion.pipeline._doc_id_from_path``)
and emits chunk ids of the form ``{doc_id}:{chunk_index}``. A slug therefore
never equals a stored document id, so comparing the two directly makes every
retrieval metric structurally zero — the harness silently measures nothing.

This module bridges the two by mapping each slug to the concrete document id(s)
that ingestion produced, so retrieval metrics compare like with like.

Duplicate slugs
---------------
The corpus ships the same document in several formats (``markdown/``, ``pdf/``,
``docx/``, ``html/``). Ten slugs therefore map to two document ids each. A
reference to such a slug is expanded to *all* of them: retrieving either format
counts as a hit, which is the honest interpretation — the evidence is present
regardless of which rendering the retriever surfaced.

Out-of-range indices
--------------------
The same document rendered in different formats yields a different number of
chunks (a 5-chunk markdown file may be a 2-chunk PDF). A reference such as
``authentication-guide:2`` is therefore valid for one rendering and impossible
for another. Expanding blindly would inject *phantom* ids that can never be
retrieved, deflating recall for reasons that have nothing to do with retrieval
quality. The resolver therefore consults the real chunk count per document and
drops any expansion whose index is out of range.

Usage
-----
::

    resolver = ChunkRefResolver.from_database()
    resolver.resolve("authentication-guide:2")
    # -> ["ca0aa574...:2"]   (the PDF rendering only has 2 chunks, so it is dropped)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChunkRefResolver:
    """Maps human-readable ``slug:index`` references to concrete chunk ids.

    Attributes
    ----------
    slug_to_doc_ids:
        Mapping of document slug (e.g. ``"authentication-guide"``) to the
        concrete document id(s) ingestion assigned. A slug may map to more
        than one id when the corpus contains the same document in several
        formats.
    doc_id_to_chunk_count:
        Number of chunks stored for each document id. Used to discard
        expansions whose chunk index does not exist for that rendering.
        When empty, no range filtering is applied.
    """

    slug_to_doc_ids: dict[str, list[str]] = field(default_factory=dict)
    doc_id_to_chunk_count: dict[str, int] = field(default_factory=dict)

    # ---- Construction ---------------------------------------------------

    @classmethod
    def from_database(cls) -> ChunkRefResolver:
        """Build a resolver from the ``documents`` and ``chunks`` tables.

        Document titles are used as slugs because ``scripts/ingest_corpus.py``
        registers each file with ``title=path.stem``.
        """
        # Imported lazily so this module stays importable without a database.
        from sqlalchemy import func, select

        from app.db.models import Chunk, Document
        from app.db.session import session_scope

        mapping: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        with session_scope() as session:
            rows = session.execute(select(Document.title, Document.document_id)).all()
            for title, doc_id in rows:
                if not title or not doc_id:
                    continue
                mapping.setdefault(title, [])
                if doc_id not in mapping[title]:
                    mapping[title].append(doc_id)

            count_rows = session.execute(
                select(Chunk.document_id, func.count(Chunk.id)).group_by(Chunk.document_id)
            ).all()
            counts = {doc_id: int(n) for doc_id, n in count_rows}

        logger.info(
            "ChunkRefResolver: %d slugs -> %d documents (%d with chunk counts)",
            len(mapping),
            sum(len(v) for v in mapping.values()),
            len(counts),
        )
        return cls(slug_to_doc_ids=mapping, doc_id_to_chunk_count=counts)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]],
        chunk_counts: dict[str, int] | None = None,
    ) -> ChunkRefResolver:
        """Build a resolver from ``(slug, doc_id)`` pairs (used in tests)."""
        mapping: dict[str, list[str]] = {}
        for slug, doc_id in pairs:
            mapping.setdefault(slug, [])
            if doc_id not in mapping[slug]:
                mapping[slug].append(doc_id)
        return cls(slug_to_doc_ids=mapping, doc_id_to_chunk_count=dict(chunk_counts or {}))

    # ---- Resolution -----------------------------------------------------

    def _in_range(self, doc_id: str, index: int) -> bool:
        """True when *index* exists for *doc_id* (or when counts are unknown)."""
        if not self.doc_id_to_chunk_count:
            return True
        count = self.doc_id_to_chunk_count.get(doc_id)
        if count is None:
            # Document has no recorded chunks — nothing to retrieve.
            return False
        return 0 <= index < count

    def resolve(self, chunk_ref: str) -> list[str]:
        """Expand a single ``slug:index`` reference into concrete chunk ids.

        Expansions whose chunk index does not exist for a given rendering are
        dropped. Returns an empty list when the slug is unknown or every
        expansion is out of range, so callers can decide whether to treat that
        as a hard error.
        """
        slug, sep, index_str = chunk_ref.rpartition(":")
        if not sep:
            # No index component — treat the whole string as a slug with no
            # specific chunk, which cannot be resolved to a chunk id.
            return []
        doc_ids = self.slug_to_doc_ids.get(slug)
        if not doc_ids:
            return []
        try:
            index = int(index_str)
        except ValueError:
            return []
        return [
            f"{doc_id}:{index}"
            for doc_id in doc_ids
            if self._in_range(doc_id, index)
        ]

    def resolve_all(self, chunk_refs: Iterable[str]) -> set[str]:
        """Expand many references into a flat set of concrete chunk ids."""
        resolved: set[str] = set()
        for ref in chunk_refs:
            resolved.update(self.resolve(ref))
        return resolved

    def unknown_slugs(self, chunk_refs: Iterable[str]) -> set[str]:
        """Return the slugs in *chunk_refs* that are not present in the corpus."""
        missing: set[str] = set()
        for ref in chunk_refs:
            slug, sep, _ = ref.rpartition(":")
            if not sep or slug not in self.slug_to_doc_ids:
                missing.add(slug or ref)
        return missing

    @property
    def is_empty(self) -> bool:
        """True when no documents were loaded (resolver cannot resolve anything)."""
        return not self.slug_to_doc_ids
