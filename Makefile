# AskMyDocs — development and evaluation tasks.
# Requires Python 3.12+. Dependencies are managed with `pip` (or `uv` if installed).

PY       ?= python
# Invoke pytest as a module: the .venv\Scripts\pytest.exe launcher embeds an
# absolute interpreter path and breaks if the virtualenv is moved.
PYTEST   ?= $(PY) -m pytest
UNAME    := $(shell uname -s 2>/dev/null || echo Windows)

.PHONY: help setup install dev deps docker-up docker-down init-db lint format typecheck test test-unit test-integration test-eval test-fast test-coverage eval bench benchmark baseline baseline-update start run worker corpus validate prod-build prod-up prod-down prod-restart prod-health prod-logs prod-logs-worker hf-login hf-stage hf-push hf-clean

help:
	@echo "AskMyDocs development commands:"
	@echo "  make setup            Create venv and install dependencies"
	@echo "  make docker-up        Start PostgreSQL, Qdrant, Redis"
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
	@echo "  make start            Start infra (Docker) + FastAPI dev server"
	@echo "  make run              Start the FastAPI dev server (infra must be up)"
	@echo "  make worker           Start Celery worker (Redis must be up)"
	@echo "  make prod-build       Build production images"
	@echo "  make prod-up          Start production stack (volumes preserved)"
	@echo "  make prod-down        Stop production stack (volumes preserved)"
	@echo "  make prod-restart     Restart api + worker"
	@echo "  make prod-logs        Tail API logs"
	@echo "  make hf-login         Log in to Hugging Face (interactive; needs a write token)"
	@echo "  make hf-stage         Assemble ./hf_staging for the Docker Space"
	@echo "  make hf-push SPACE=user/name   Push ./hf_staging to a HF Space"
	@echo "  make hf-clean         Remove the local ./hf_staging directory"

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
	docker compose up -d postgres qdrant
	$(PY) scripts/init_db.py

corpus:
	$(PY) scripts/generate_corpus.py

validate:
	$(PY) scripts/validate_dataset.py
	$(PY) scripts/check_chunk_refs.py

lint:
	$(PY) -m ruff check app tests scripts

format:
	$(PY) -m ruff check --fix app tests scripts
	$(PY) -m ruff format app tests scripts

typecheck:
	$(PY) -m mypy app

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
	$(PY) -m app.evaluation.run

benchmark:
	$(PY) scripts/generate_benchmark.py

baseline:
	$(PY) scripts/show_baseline.py

baseline-update:
	$(PY) scripts/update_baseline.py

start:
	docker compose up -d
	$(PY) -m uvicorn app.main:app --reload --reload-dir app --reload-dir scripts --port 8000

run:
	$(PY) -m uvicorn app.main:app --reload --reload-dir app --reload-dir scripts --port 8000

worker:
ifeq ($(UNAME),Windows)
	$(PY) -m celery -A app.celery_app.celery_app worker -Q ingestion,evaluation,benchmark -l info --pool=solo
else
	$(PY) -m celery -A app.celery_app.celery_app worker -Q ingestion,evaluation,benchmark -l info --concurrency 2
endif

prod-build:
	docker compose -f docker-compose.prod.yml build

prod-up:
	docker compose -f docker-compose.prod.yml up -d

prod-down:
	docker compose -f docker-compose.prod.yml stop

prod-restart:
	docker compose -f docker-compose.prod.yml restart api worker

prod-logs:
	docker compose -f docker-compose.prod.yml logs --tail 100 api

prod-logs-worker:
	docker compose -f docker-compose.prod.yml logs --tail 100 worker

# --- Hugging Face Docker Space ---------------------------------------------
# Stage/push the all-in-one Space. See deploy/huggingface/README.md.
hf-login:
	hf auth login

hf-stage:
	$(PY) scripts/hf_space_stage.py --output ./hf_staging

hf-push:
ifndef SPACE
	$(error SPACE is required, e.g. make hf-push SPACE=username/askmydocs)
endif
	$(PY) scripts/hf_space_stage.py --push $(SPACE)

hf-clean:
ifneq ($(UNAME),Windows)
	rm -rf ./hf_staging
else
	if exist hf_staging rmdir /s /q hf_staging
endif