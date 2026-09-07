# AskMyDocs — development and evaluation tasks.
# Requires Python 3.12+. Dependencies are managed with `pip` (or `uv` if installed).

PY       ?= python
PYTEST   ?= pytest
UNAME    := $(shell uname -s 2>/dev/null || echo Windows)

.PHONY: help setup install dev deps docker-up docker-down init-db lint format typecheck test test-unit test-integration test-eval test-fast test-coverage eval bench benchmark baseline baseline-update run corpus validate

help:
	@echo "AskMyDocs development commands:"
	@echo "  make setup            Create venv and install dependencies"
	@echo "  make docker-up        Start PostgreSQL, Qdrant, OpenSearch"
	@echo "  make docker-down      Stop all infra containers"
	@echo "  make init-db          Initialize PostgreSQL schema and start services"
	@echo "  make corpus           Regenerate the document corpus (mixed formats)"
	@echo "  make validate         Validate golden dataset + chunk references"
	@echo "  make lint             Run ruff"
	@echo "  make format           Auto-format with ruff"
	@echo "  make typecheck        Run mypy"
	@echo "  make test             Run full test suite"
	@echo "  make test-unit        Run unit tests (no external services)"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-eval        Run evaluation tests"
	@echo "  make test-fast        Run non-slow tests"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make eval             Run the full evaluation pipeline"
	@echo "  make benchmark        Generate evals/reports/benchmark.md"
	@echo "  make baseline         Show the current evaluation baseline"
	@echo "  make baseline-update   Record the latest eval as the new baseline"
	@echo "  make run              Start the FastAPI dev server"

setup:
ifeq ($(UNAME),Windows)
	$(PY) -m venv .venv
	.venv\Scripts\pip install -e ".[dev]"
else
	$(PY) -m venv .venv
	.venv/bin/pip install -e ".[dev]"
endif

install:
	$(PY) -m pip install -e ".[dev]"

dev:
	$(PY) -m pip install -e ".[dev]"

docker-up:
	docker compose up -d

docker-down:
	docker compose down

init-db:
	docker compose up -d postgres qdrant opensearch
	$(PY) scripts/init_db.py

corpus:
	$(PY) scripts/generate_corpus.py

validate:
	$(PY) scripts/validate_dataset.py
	$(PY) scripts/check_chunk_refs.py

lint:
	ruff check app tests scripts

format:
	ruff check --fix app tests scripts
	ruff format app tests scripts

typecheck:
	mypy app

test:
	$(PYTEST) tests

test-unit:
	$(PYTEST) tests/unit -m unit

test-integration:
	$(PYTEST) tests/integration -m integration

test-eval:
	$(PYTEST) tests/evaluation -m evaluation

test-fast:
	$(PYTEST) tests -m "not slow"

test-coverage:
	$(PYTEST) tests --cov=app --cov-report=term-missing --cov-report=html

eval:
	$(PYTHON) -m app.evaluation.run

benchmark:
	$(PY) scripts/generate_benchmark.py

baseline:
	$(PY) scripts/show_baseline.py

baseline-update:
	$(PY) scripts/update_baseline.py

run:
	uvicorn app.main:app --reload --port 8000