"""Golden evaluation dataset loader & validator for AskMyDocs.

The dataset lives at ``evals/dataset/golden.jsonl``. Each line is a JSON
object with the schema:

    {
      "id": "q001",
      "question": "...",
      "expected_answer": "...",
      "relevant_chunk_ids": ["chunk_12"],
      "answerable": true,
      "difficulty": "medium"
    }

``relevant_chunk_ids`` reference deterministic chunk ids of the form
``{chunk_id}:{chunk_index}`` which the ingestion pipeline assigns in
document order. The ids are resolved by ``scripts/resolve_relevant_chunks.py``
once documents are ingested.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"
UNANSWERABLE = "unanswerable"

DIFFICULTIES = (EASY, MEDIUM, HARD, UNANSWERABLE)
PER_DIFFICULTY_MIN = {
    EASY: 10,
    MEDIUM: 20,
    HARD: 10,
    UNANSWERABLE: 10,
}


class GoldenExample(BaseModel):
    """A single golden evaluation example."""

    id: str = Field(description="Unique question id, e.g. q001")
    question: str
    expected_answer: str = Field(default="", description="Reference answer for Ragas factual metrics")
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids (resolved after ingestion) that support the answer",
    )
    answerable: bool = Field(default=True, description="Whether the corpus can answer this question")
    difficulty: str = Field(default=MEDIUM, description="easy | medium | hard | unanswerable")

    @property
    def is_unanswerable(self) -> bool:
        return self.difficulty == UNANSWERABLE or not self.answerable


class GoldenDataset(BaseModel):
    """Container with validation helpers."""

    examples: list[GoldenExample]

    @classmethod
    def from_jsonl(cls, path: Path | str) -> GoldenDataset:
        path = Path(path)
        examples: list[GoldenExample] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                examples.append(GoldenExample.model_validate(json.loads(line)))
        return cls(examples=examples)

    def to_jsonl(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for ex in self.examples:
                fh.write(ex.model_dump_json() + "\n")

    def filter_answerable(self, answerable: bool = True) -> list[GoldenExample]:
        return [ex for ex in self.examples if ex.answerable == answerable]

    def filter_difficulty(self, difficulty: str) -> list[GoldenExample]:
        return [ex for ex in self.examples if ex.difficulty == difficulty]

    def summary(self) -> dict[str, int]:
        counts = {d: 0 for d in DIFFICULTIES}
        for ex in self.examples:
            counts[ex.difficulty] = counts.get(ex.difficulty, 0) + 1
        counts["total"] = len(self.examples)
        return counts


def validate_distribution(dataset: GoldenDataset) -> list[str]:
    """Return a list of problems, or an empty list if the dataset is valid.

    Enforces minimum counts per difficulty, unique ids, and consistency
    between ``answerable`` and ``difficulty``.
    """
    problems: list[str] = []
    summary = dataset.summary()
    for difficulty, minimum in PER_DIFFICULTY_MIN.items():
        if summary.get(difficulty, 0) < minimum:
            problems.append(
                f"difficulty '{difficulty}' has {summary.get(difficulty, 0)} examples, minimum {minimum}"
            )

    seen: set[str] = set()
    for ex in dataset.examples:
        if ex.id in seen:
            problems.append(f"duplicate id '{ex.id}'")
        seen.add(ex.id)
        if ex.difficulty == UNANSWERABLE and ex.answerable:
            problems.append(f"{ex.id}: difficulty=unanswerable but answerable=true")
        if ex.difficulty != UNANSWERABLE and not ex.answerable:
            problems.append(f"{ex.id}: answerable=false but difficulty is not unanswerable")
        if ex.difficulty == UNANSWERABLE and ex.relevant_chunk_ids:
            problems.append(f"{ex.id}: unanswerable example lists relevant_chunk_ids")
    return problems


def load_default() -> GoldenDataset:
    """Load the golden dataset shipped in the repository."""
    # scripts/eval_dataset.py -> project root
    root = Path(__file__).resolve().parent.parent
    return GoldenDataset.from_jsonl(root / "evals" / "dataset" / "golden.jsonl")