"""Configurable text chunking for document ingestion.

Implements two chunking strategies:

1. **fixed**  — token-aware fixed-size windows with configurable overlap.
2. **paragraph** — paragraph/heading-aware chunking that prefers to keep
   section boundaries intact and groups paragraphs until ``CHUNK_SIZE``
   tokens are reached.

Both strategies emit :class:`Chunk` objects carrying deterministic index
metadata so chunk ids of the form ``{doc_id}:{chunk_index}`` are
reproducible across runs (required for the golden evaluation dataset).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

# Rough token estimate: ~4 characters per token for Latin text.
CHARS_PER_TOKEN = 4.0

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```")


class ChunkingStrategy(StrEnum):
    """Supported chunking strategies."""

    FIXED = "fixed"
    PARAGRAPH = "paragraph"


@dataclass
class Chunk:
    """A single indexed chunk with source metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    source: str
    page_number: int | None
    section: str | None
    index: int
    content_hash: str
    metadata: dict = field(default_factory=dict)

    def to_retrieval_payload(self) -> dict:
        """Payload for vector/BM25 indexers."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "text": self.text,
            "page_number": self.page_number,
            "section": self.section,
            "source": self.source,
            "index": self.index,
            "content_hash": self.content_hash,
        }


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text (chars / 4)."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def truncate_to_chars(text: str, max_tokens: int) -> str:
    """Truncate text so its estimated token count is at most ``max_tokens``."""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def content_hash(text: str) -> str:
    """SHA-256 hash of the normalized chunk text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _is_heading(line: str) -> bool:
    return bool(_MARKDOWN_HEADING_RE.match(line))


def _heading_section(line: str) -> str | None:
    m = _MARKDOWN_HEADING_RE.match(line)
    if not m:
        return None
    return m.group(2).strip()


def _split_blocks(text: str) -> list[list[str]]:
    """Split text into blocks (paragraphs), preserving markdown structure.

    Returns a list of blocks; each block is a list of lines that should be
    kept together (e.g. a table, a code fence, or a paragraph).
    """
    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if current:
            blocks.append(list(current))
            current.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _FENCE_RE.match(stripped):
            flush()
            current.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            current.append(line)
            continue
        if _is_heading(stripped):
            flush()
            current.append(line)
            continue
        # Table row: keep consecutive table rows together.
        if stripped.startswith("|") and current and current[-1].strip().startswith("|"):
            current.append(line)
            continue
        flush()
        current.append(line)
    flush()
    return blocks


def _block_text(block: list[str]) -> str:
    return "\n".join(block).strip()


class Chunker:
    """Chunk plain text into :class:`Chunk` objects.

    Parameters
    ----------
    strategy:
        ``"fixed"`` or ``"paragraph"``.
    chunk_size:
        Target chunk size in estimated tokens.
    chunk_overlap:
        Overlap between adjacent chunks (tokens) for the fixed strategy;
        ignored for the paragraph strategy.
    """

    def __init__(
        self,
        strategy: str | ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.strategy = ChunkingStrategy(strategy)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -- fixed-size strategy ------------------------------------------------
    def _chunk_fixed(
        self,
        text: str,
        *,
        document_id: str,
        document_name: str,
        source: str,
        page_number: int | None,
        section: str | None,
    ) -> list[Chunk]:
        sentences = _SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
        chunks: list[Chunk] = []
        buffer = ""
        index = 0

        def flush() -> None:
            nonlocal index
            if not buffer.strip():
                return
            chunks.append(
                self._make_chunk(
                    text=buffer.strip(),
                    index=index,
                    document_id=document_id,
                    document_name=document_name,
                    source=source,
                    page_number=page_number,
                    section=section,
                )
            )
            index += 1

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if estimate_tokens(sentence) > self.chunk_size:
                flush()
                # Split an oversized sentence into hard windows.
                for part in self._hard_split(sentence):
                    chunks.append(
                        self._make_chunk(
                            text=part,
                            index=index,
                            document_id=document_id,
                            document_name=document_name,
                            source=source,
                            page_number=page_number,
                            section=section,
                        )
                    )
                    index += 1
                buffer = ""
                continue
            if buffer and estimate_tokens(buffer) + estimate_tokens(sentence) > self.chunk_size:
                flush()
                buffer = " ".join(
                    buffer.split()[-int(self.chunk_overlap * CHARS_PER_TOKEN) :]
                )
            buffer = " ".join(filter(None, (buffer, sentence)))
        flush()
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        max_chars = int(self.chunk_size * CHARS_PER_TOKEN)
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    # -- paragraph strategy ------------------------------------------------
    def _chunk_paragraph(
        self,
        text: str,
        *,
        document_id: str,
        document_name: str,
        source: str,
        page_number: int | None,
        section: str | None,
    ) -> list[Chunk]:
        """Chunk by paragraphs, starting a new chunk at each heading.

        Section boundaries are preserved: a heading always begins a new
        chunk, and subsequent paragraphs are packed into that chunk until
        ``chunk_size`` tokens are reached.
        """
        blocks = _split_blocks(text)
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        index = 0
        current_section = section

        def flush() -> None:
            nonlocal index, buffer, buffer_tokens
            if not buffer:
                return
            chunk_text = "\n\n".join(b.strip() for b in buffer if b.strip()).strip()
            if chunk_text:
                chunks.append(
                    self._make_chunk(
                        text=chunk_text,
                        index=index,
                        document_id=document_id,
                        document_name=document_name,
                        source=source,
                        page_number=page_number,
                        section=current_section,
                    )
                )
                index += 1
            buffer = []
            buffer_tokens = 0

        for block in blocks:
            block_text = _block_text(block).strip()
            if not block_text:
                continue
            first_line = block[0].strip()
            is_heading = _is_heading(first_line)
            if is_heading:
                # A heading always starts a new section chunk. Flush whatever
                # accumulated in the previous section first.
                flush()
                heading_section = _heading_section(first_line)
                if heading_section is not None:
                    current_section = heading_section
            block_tokens = estimate_tokens(block_text)
            # If a single block exceeds the size, split it on its own.
            if block_tokens > self.chunk_size:
                flush()
                for sub in self._split_oversized_block(block_text):
                    chunks.append(
                        self._make_chunk(
                            text=sub,
                            index=index,
                            document_id=document_id,
                            document_name=document_name,
                            source=source,
                            page_number=page_number,
                            section=current_section,
                        )
                    )
                    index += 1
                continue
            if buffer and buffer_tokens + block_tokens > self.chunk_size:
                flush()
            buffer.append(block_text)
            buffer_tokens += block_tokens
        flush()
        return chunks

    def _split_oversized_block(self, text: str) -> list[str]:
        """Split an oversized block (e.g. a huge table or paragraph)."""
        max_chars = int(self.chunk_size * CHARS_PER_TOKEN)
        parts: list[str] = []
        current = ""
        for line in text.splitlines():
            if len(current) + len(line) + 1 > max_chars and current:
                parts.append(current.strip())
                current = line
            else:
                current = " ".join(filter(None, (current, line)))
        if current.strip():
            parts.append(current.strip())
        return parts

    # -- shared ---------------------------------------------------------------
    def _make_chunk(
        self,
        *,
        text: str,
        index: int,
        document_id: str,
        document_name: str,
        source: str,
        page_number: int | None,
        section: str | None,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document_id}:{index}",
            document_id=document_id,
            document_name=document_name,
            text=text,
            source=source,
            page_number=page_number,
            section=section,
            index=index,
            content_hash=content_hash(text),
        )

    def chunk_text(
        self,
        text: str,
        *,
        document_id: str,
        document_name: str,
        source: str,
        page_number: int | None = None,
        section: str | None = None,
    ) -> list[Chunk]:
        """Chunk arbitrary text into :class:`Chunk` objects."""
        if self.strategy is ChunkingStrategy.FIXED:
            return self._chunk_fixed(
                text=text,
                document_id=document_id,
                document_name=document_name,
                source=source,
                page_number=page_number,
                section=section,
            )
        return self._chunk_paragraph(
            text=text,
            document_id=document_id,
            document_name=document_name,
            source=source,
            page_number=page_number,
            section=section,
        )

    def chunk(
        self,
        text: str,
        *,
        document_id: str,
        document_name: str,
        source: str,
        page_number: int | None = None,
        section: str | None = None,
    ) -> list[Chunk]:
        """Alias for :meth:`chunk_text` with a shorter name."""
        return self.chunk_text(
            text=text,
            document_id=document_id,
            document_name=document_name,
            source=source,
            page_number=page_number,
            section=section,
        )

    def chunk_document(self, document: dict) -> list[Chunk]:
        """Chunk a parsed document dict (text + metadata) into Chunks.

        The document dict is expected to have keys: ``text``, ``document_id``,
        ``document_name``, ``source``, and optional ``page_number`` and
        ``section``.
        """
        return self.chunk_text(
            text=document["text"],
            document_id=document["document_id"],
            document_name=document.get("document_name") or document["document_id"],
            source=document.get("source") or "",
            page_number=document.get("page_number"),
            section=document.get("section"),
        )


def chunk_all(
    documents: Iterable[dict],
    *,
    strategy: str | ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Helper to chunk a batch of parsed documents."""
    chunker = Chunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))
    return chunks