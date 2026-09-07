"""Test configuration for AskMyDocs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ----------------------------------------------------------------------
# Pytest markers — registered here so `-m unit`, `-m integration`, and
# `-m evaluation` work across the test suite without decorating every
# test function individually.
# ----------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: unit tests — no I/O, no service dependencies")
    config.addinivalue_line("markers", "integration: integration tests — may require service containers")
    config.addinivalue_line("markers", "evaluation: evaluation system tests — run against mock pipeline")


# ----------------------------------------------------------------------
# Auto-apply markers based on file path so test files don't need individual
# decorator lines. Markers are applied in order of specificity so that
# evaluation unit tests (under tests/unit/evaluation*) override the default
# unit classification.
# ----------------------------------------------------------------------
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        # fspath is py.path.local; rootdir comes from the pytest Config object
        root = item.config.rootdir
        rel_path = item.fspath.relto(root)
        # Normalise to Unix-style path for consistent cross-platform matching
        path_str = rel_path.replace("\\", "/")

        if "/integration/" in path_str or path_str.startswith("tests/integration/"):
            item.add_marker(pytest.mark.integration)
        elif "/evaluation/" in path_str or any(
            path_str.startswith(f"tests/unit/test_evaluation{ext}")
            for ext in [".py", "_datasets", "_metrics", "_citation", "_ragas", "_regression"]
        ):
            item.add_marker(pytest.mark.evaluation)
        else:
            item.add_marker(pytest.mark.unit)

        if "/integration/" in path_str or path_str.startswith("tests/integration/"):
            item.add_marker(pytest.mark.integration)
        elif "/evaluation/" in path_str or any(
            path_str.startswith(f"tests/unit/test_evaluation{ext}")
            for ext in [".py", "_datasets", "_metrics", "_citation", "_ragas", "_regression"]
        ):
            item.add_marker(pytest.mark.evaluation)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def corpus_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "corpus"