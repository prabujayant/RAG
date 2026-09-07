"""Dataset loading and validation for the golden evaluation set.

Loads ``evals/dataset/golden.jsonl`` (one JSON object per line) and exposes
strongly-typed :class:`GoldenExample` and :class:`GoldenDataset` objects for
use by the evaluation runner. Pydantic-style validation is implemented
manually to keep this module lightweight and free of external dependencies
beyond the standard library.

Schema (per spec §21)
---------------------
::

    {
        "id":                 "q001",            # str, unique
        "question":           "...",             # str, non-empty
        "expected_answer":    "...",             # str, may be empty for unanswerable
        "relevant_chunk_ids": ["doc-id:0", ...], # list[str], empty for unanswerable
        "answerable":         true,              # bool
        "difficulty":         "easy"             # easy | medium | hard | unanswerable
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------


class Difficulty(StrEnum):
    """Question difficulty buckets used to stratify evaluation metrics."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNANSWERABLE = "unanswerable"


_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+:\d+$")


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenExample:
    """A single golden evaluation example.

    Attributes
    ----------
    id:
        Unique question identifier, e.g. ``"q001"``.
    question:
        The natural-language question.
    expected_answer:
        The expected reference answer. May be empty for unanswerable
        questions.
    relevant_chunk_ids:
        Chunk ids of the form ``"{doc_id}:{chunk_index}"`` that are
        expected to be retrieved. Empty for unanswerable questions.
    answerable:
        ``True`` for answerable questions, ``False`` for unanswerable.
    difficulty:
        Difficulty bucket.
    """

    id: str
    question: str
    expected_answer: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    answerable: bool = True
    difficulty: Difficulty = Difficulty.EASY

    def __post_init__(self) -> None:
        # Coerce difficulty (accept str)
        if not isinstance(self.difficulty, Difficulty):
            object.__setattr__(self, "difficulty", Difficulty(str(self.difficulty)))

    @property
    def is_unanswerable(self) -> bool:
        """Return True if this example is unanswerable."""
        return not self.answerable

    def to_dict(self) -> dict:
        """Serialize to a plain dict matching the JSONL schema."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "answerable": self.answerable,
            "difficulty": self.difficulty.value,
        }


@dataclass
class DatasetSummary:
    """Lightweight summary of a loaded dataset.

    Useful for diagnostics and for the ``evals/reports`` artifacts.
    """

    total: int = 0
    answerable: int = 0
    unanswerable: int = 0
    by_difficulty: dict[str, int] = field(default_factory=dict)
    with_relevant_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "by_difficulty": dict(self.by_difficulty),
            "with_relevant_chunks": self.with_relevant_chunks,
        }


class GoldenDataset:
    """In-memory golden dataset, loaded from a JSONL file.

    Use :meth:`load` to read from disk, or :meth:`from_examples` to
    construct directly in tests.
    """

    _examples: list[GoldenExample]
    _path: Path | None

    def __init__(self, examples: list[GoldenExample], *, _path: Path | None = None) -> None:
        self._examples = list(examples)
        self._path = _path

    @property
    def path(self) -> Path | None:
        """Path the dataset was loaded from (None for in-memory datasets)."""
        return self._path

    # ---- Construction ---------------------------------------------------

    @classmethod
    def from_examples(cls, examples: list[GoldenExample]) -> GoldenDataset:
        """Build a dataset directly from a list of examples (used in tests)."""
        return cls(examples)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        validate_chunk_ids: bool = True,
    ) -> GoldenDataset:
        """Load and validate the golden dataset from *path*.

        Parameters
        ----------
        path:
            Filesystem path to the ``golden.jsonl`` file.
        validate_chunk_ids:
            When True (default) chunk ids must match ``{doc_id}:{chunk_index}``.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is malformed, has duplicate ids, or contains
            invalid chunk references.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Golden dataset not found: {p}")

        examples: list[GoldenExample] = []
        seen_ids: set[str] = set()
        line_number = 0

        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line_number += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {p}: {exc}"
                ) from exc

            example = _parse_example(obj, line_number, validate_chunk_ids=validate_chunk_ids)
            if example.id in seen_ids:
                raise ValueError(f"Duplicate question id on line {line_number}: {example.id!r}")
            seen_ids.add(example.id)
            examples.append(example)

        return cls(examples, _path=p)

    # ---- Accessors -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[GoldenExample]:
        return iter(self._examples)

    def __getitem__(self, idx: int) -> GoldenExample:
        return self._examples[idx]

    @property
    def examples(self) -> list[GoldenExample]:
        """Return the underlying list (read-only copy is the caller's choice)."""
        return list(self._examples)

    # ---- Filtering -------------------------------------------------------

    def filter(
        self,
        *,
        difficulty: Difficulty | str | None = None,
        answerable: bool | None = None,
    ) -> GoldenDataset:
        """Return a new dataset filtered by difficulty and/or answerability.

        Parameters
        ----------
        difficulty:
            Optional :class:`Difficulty` to keep only matching questions.
        answerable:
            Optional bool; ``True`` keeps only answerable questions,
            ``False`` keeps only unanswerable ones, ``None`` keeps both.
        """
        diff = Difficulty(difficulty) if difficulty is not None else None
        out = [
            ex
            for ex in self._examples
            if (diff is None or ex.difficulty == diff)
            and (answerable is None or ex.answerable == answerable)
        ]
        return GoldenDataset(out)

    def limit(self, n: int) -> GoldenDataset:
        """Return a new dataset containing at most *n* examples."""
        return GoldenDataset(self._examples[: max(0, n)])

    # ---- Summary ---------------------------------------------------------

    def summary(self) -> DatasetSummary:
        """Return aggregate statistics about the dataset."""
        s = DatasetSummary(total=len(self._examples))
        for ex in self._examples:
            if ex.answerable:
                s.answerable += 1
            else:
                s.unanswerable += 1
            key = ex.difficulty.value
            s.by_difficulty[key] = s.by_difficulty.get(key, 0) + 1
            if ex.relevant_chunk_ids:
                s.with_relevant_chunks += 1
        return s

    def to_jsonl(self, path: str | Path) -> None:
        """Write the dataset to a JSONL file.

        Parameters
        ----------
        path:
            Filesystem path to write to. Parent directories are created
            if they do not exist.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for ex in self._examples:
                obj = {
                    "id": ex.id,
                    "question": ex.question,
                    "expected_answer": ex.expected_answer,
                    "relevant_chunk_ids": ex.relevant_chunk_ids,
                    "answerable": ex.answerable,
                    "difficulty": ex.difficulty.value,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


_DEFAULT_DATASET_PATH = "evals/dataset/golden.jsonl"


def load_golden_dataset(
    path: str | Path | None = None,
    *,
    validate_chunk_ids: bool = True,
) -> GoldenDataset:
    """Load the golden dataset from the default or specified *path*.

    Thin wrapper around :meth:`GoldenDataset.load` for convenience.
    """
    p = Path(path) if path is not None else Path(_DEFAULT_DATASET_PATH)
    return GoldenDataset.load(p, validate_chunk_ids=validate_chunk_ids)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "question",
    "expected_answer",
    "relevant_chunk_ids",
    "answerable",
    "difficulty",
)


def _parse_example(
    obj: dict,
    line_number: int,
    *,
    validate_chunk_ids: bool,
) -> GoldenExample:
    if not isinstance(obj, dict):
        raise ValueError(
            f"Line {line_number}: expected JSON object, got {type(obj).__name__}"
        )

    missing = [k for k in _REQUIRED_FIELDS if k not in obj]
    if missing:
        raise ValueError(
            f"Line {line_number}: missing required fields: {missing}"
        )

    qid = str(obj["id"]).strip()
    if not qid:
        raise ValueError(f"Line {line_number}: 'id' is empty")

    question = str(obj["question"]).strip()
    if not question:
        raise ValueError(f"Line {line_number}: 'question' is empty for id={qid!r}")

    expected_answer = str(obj.get("expected_answer", ""))
    relevant_raw = obj.get("relevant_chunk_ids", [])
    if not isinstance(relevant_raw, list):
        raise ValueError(
            f"Line {line_number}: 'relevant_chunk_ids' must be a list, "
            f"got {type(relevant_raw).__name__}"
        )
    relevant_chunk_ids = [str(cid) for cid in relevant_raw]

    if validate_chunk_ids:
        for cid in relevant_chunk_ids:
            if not _CHUNK_ID_RE.match(cid):
                raise ValueError(
                    f"Line {line_number}: invalid chunk id format {cid!r} "
                    "(expected '{doc_id}:{chunk_index}')"
                )

    answerable = bool(obj["answerable"])
    try:
        difficulty = Difficulty(str(obj["difficulty"]))
    except ValueError as exc:
        valid = ", ".join(d.value for d in Difficulty)
        raise ValueError(
            f"Line {line_number}: invalid difficulty {obj['difficulty']!r}; "
            f"must be one of: {valid}"
        ) from exc

    # Cross-field invariants
    if not answerable and relevant_chunk_ids:
        # Warn but don't fail — useful when an unanswerable question has
        # accidental relevant chunks from a re-indexed corpus.
        logger.debug(
            "Unanswerable question %s has non-empty relevant_chunk_ids; "
            "this is allowed but should be reviewed.",
            qid,
        )
    if answerable and not relevant_chunk_ids:
        # Also informational; some "easy" question categories may be marked
        # answerable but routed via a different mechanism.
        logger.debug(
            "Answerable question %s has no relevant_chunk_ids; "
            "retrieval metrics will be 0 for this row.",
            qid,
        )

    return GoldenExample(
        id=qid,
        question=question,
        expected_answer=expected_answer,
        relevant_chunk_ids=relevant_chunk_ids,
        answerable=answerable,
        difficulty=difficulty,
    )
