# How to Run AskMyDocs

## Quick Start (after one-time setup)

```bash
make start          # starts infra (Docker) + the API in one command
make worker         # in a second terminal: Celery worker (see note below)
```

Or without `make`:

```powershell
.\start.ps1          # infra + API
.\start.ps1 -Init    # also initialize the DB schema (first run)
.\start.ps1 -Worker  # also start the Celery worker (background ingestion)
```

Then open `http://127.0.0.1:8000/docs`.

> **Background ingestion needs a worker.** The web UI uploads with
> `?background=true`, which hands the parse/embed/index job to Celery. Run
> `make worker` (or `.\start.ps1 -Worker`) alongside the API. If no worker is
> running the API automatically falls back to ingesting synchronously, so
> uploads still succeed — just without the async speed-up.
>
> **First upload after a (re)start is slow (~1–2 min).** The worker pre-warms
> the embedding model at boot; give it a minute after starting, then uploads
> take seconds. Every worker restart repeats the one-off load — keep the
> worker window open instead of restarting it.

## Prerequisites
- Python 3.12+
- Docker & Docker Compose (for PostgreSQL, Qdrant, Redis)
- An OpenRouter API key

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

## Start Infrastructure

```bash
# Start PostgreSQL, Qdrant, and Redis
docker compose up -d

# Initialize the PostgreSQL schema and services
make init-db
```

## Ingest the Corpus

```bash
# Generate / embed the document corpus (downloads BAAI/bge-m3 on first run)
make corpus
# or: python scripts/generate_corpus.py
```

## Run the API

```bash
make run
# or:
uvicorn app.main:app --reload --port 8000
```

The API is then available at `http://127.0.0.1:8000` (docs at `/docs`).

## Development Commands

| Command | Description |
|---------|-------------|
| `make start` | Start infra (Docker) + API in one command |
| `make run` | Start the API only (infra must already be up) |
| `make worker` | Start the Celery worker (ingestion/evaluation/benchmark queues) |
| `make lint` | Run ruff lint |
| `make format` | Auto-format with ruff |
| `make typecheck` | Run mypy |
| `make test` | Run full test suite |
| `make test-unit` | Run unit tests (no external services) |
| `make test-integration` | Run integration tests (requires Docker) |
| `make eval` | Run evaluation pipeline |
| `make benchmark` | Generate benchmark reports |

## API Endpoints

- `GET /health` - Health check
- `GET /documents` - List indexed documents
- `POST /query` - Query with a question

```json
POST /query
{
  "question": "How do I rotate API keys?",
  "top_k": 5
}
```