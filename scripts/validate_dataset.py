"""Validate the golden evaluation dataset.

Checks:
- JSONL parses and schema is valid
- difficulty counts meet minimums (easy>=10, medium>=20, hard>=10, unanswerable>=10)
- ids are unique
- answerable flag is consistent with difficulty
- unanswerable examples carry no relevant_chunk_ids
- referenced document ids exist in the corpus manifest

Run:
    python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_dataset import load_default, validate_distribution  # noqa: E402


def _load_manifest() -> dict[str, str]:
    manifest_path = ROOT / "data" / "corpus" / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    try:
        dataset = load_default()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load dataset: {exc}")
        return 1

    problems = validate_distribution(dataset)
    summary = dataset.summary()
    print(f"Total examples: {summary['total']}")
    for d in ("easy", "medium", "hard", "unanswerable"):
        print(f"  {d}: {summary.get(d, 0)}")

    manifest = _load_manifest()
    if manifest:
        known_docs = set(manifest)
        for ex in dataset.examples:
            for chunk_id in ex.relevant_chunk_ids:
                doc_part, _, _ = chunk_id.rpartition(":")
                if doc_part not in known_docs:
                    problems.append(f"{ex.id}: unknown document '{doc_part}' in {chunk_id}")
    else:
        print("WARNING: corpus manifest not found; skipping document reference check")

    if problems:
        print("Validation FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Dataset validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())