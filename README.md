# AskMyDocs — RAG Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system for querying technical documentation with hybrid search, cross-encoder reranking, OpenAI-style LLM-judge grounding, per-claim citations, tool-use agents, cost observability, and deterministic safety screening.

## Overview

AskMyDocs answers natural-language questions about software products using trusted documentation. It combines dense vector search (Qdrant) with keyword retrieval (Postgres tsvector), reranks results with a cross-encoder model, generates structured answers via OpenRouter, and grounds each answer with an LLM judge that verifies every claim against the cited evidence.

## Architecture

```
                    User Query
                        │
                        ▼
        ┌─────────────────────────────────────────────┐
        │  Hybrid Retrieval (Vector + BM25)           │
        │  • Vector : BAAI/bge-m3 embeddings → Qdrant │
        │  • BM25   : Postgres tsvector keyword search      │
        │  • Reciprocal Rank Fusion combines scores   │
        └───────────────────┬─────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Reranking (Cross-Encoder)                  │
        │  • MiniLM multilingual cross-encoder       │
        │    reorders top-k (RERANKER_MODEL)         │
        └───────────────────┬─────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Grounded Generation (LLM)                  │
        │  • OpenRouter (default: nex-agi/nex-n2.5-mini:free) │
        │  • Structured JSON with inline [C1] markers │
        │  • LLM_REASONING_EFFORT tunes think-vs-speed │
        └───────────────────┬─────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Grounding Validation (OpenAI-style)        │
        │  • Claim extraction from answer sentences   │
        │  • Per-claim LLM judge vs cited evidence    │
        │  • Aggregate grounded / partially / refused │
        └───────────────────┬─────────────────────────┘
                            │
                            ▼
              Structured QueryResponse
```

## Data Pipeline

Ingestion supports multiple document formats: **Markdown, HTML, DOCX, and PDF**. The pipeline runs in four stages:

1. **Parsing** — format-specific parsers extract clean text.
2. **Cleaning** — removes boilerplate, headers, and noise.
3. **Chunking** — splits text into overlapping paragraph-level chunks.
4. **Embedding & indexing** — embeds chunks with BGE-M3 and stores them in Qdrant (dense vectors) plus Postgres keyword postings (tsvector).

The corpus lives under `data/corpus/` (organized by format) with a `manifest.json`. Existing corpora are indexed into the vector store on ingestion.

## Grounding & Citations

Answers carry inline citation markers (`[C1]`, `[C2]`, …). Each marker maps 1:1 to a retrieved chunk that was shown to the model.

### Citation resolution
Citation evidence text is resolved **authoritatively from the retrieved candidate** (`[C{i}]` ↔ `candidates[i-1]`) rather than trusting the model to echo back chunk IDs verbatim. This makes grounding robust when the model truncates or rephrases the cited text.

### Grounding status
After generation, the answer is split into atomic claims. Each cited claim is sent to an **LLM judge** (OpenAI-style groundedness check) together with its evidence; the judge returns `supported` / `partially_supported` / `unsupported`. A deterministic token-overlap validator serves as a fast fallback when the judge is unavailable.

| Status | Meaning |
|--------|---------|
| `grounded` | Every claim is supported by its cited evidence |
| `partially_grounded` | At least one claim is supported, others are not fully verified |
| `ungrounded` | No claim is supported by evidence |
| `refused` | The model refused to answer / no claims extracted |

## API

AskMyDocs exposes three REST endpoints:

### `GET /health`
```json
{ "status": "healthy", "version": "0.1.0" }
```

### `GET /documents`
Lists the documents currently indexed in the corpus.

### `POST /query`
```http
POST /query
Content-Type: application/json

{
  "question": "How do I rotate API keys?",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "API keys can be rotated from the Admin Console, which invalidates the previous key immediately. [C1]",
  "citations": [
    {
      "citation_id": "[C1]",
      "chunk_id": "authentication-guide:3",
      "text": "Tokens can be rotated from the Admin Console.",
      "page_number": null,
      "section": "Token Lifecycle"
    }
  ],
  "claims": [
    {
      "claim": "API keys can be rotated from the Admin Console",
      "citation_ids": ["[C1]"],
      "status": "supported",
      "reason": "The evidence explicitly supports this claim."
    }
  ],
  "grounded": true,
  "grounding_status": "grounded",
  "confidence": 0.95,
  "refused": false,
  "refused_reason": null,
  "latency_ms": 2345.6,
  "model": "nex-agi/nex-n2.5-mini:free",
  "usage": {
    "prompt_tokens": 1153,
    "completion_tokens": 236,
    "total_tokens": 1389,
    "cost_usd": 0.0
  }
}
```

Request fields:

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | The user's question (5–2000 chars) |
| `top_k` | int? | Override evidence chunk count (1–50) |
| `temperature` | float? | LLM sampling temperature override |
| `document_ids` | string[]? | Restrict retrieval to these documents |
| `anchor_document_ids` | string[]? | Anchor uploads; other uploads are excluded |
| `allow_generic` | bool | Allow general-knowledge answers when evidence is thin |

Additional endpoints:

- `POST /query/stream` — same pipeline as `/query`, streamed as Server-Sent Events (`started → retrieval → reranking → evidence → generation → grounding → done`). The UI uses this so progress renders instead of spinning.
- `POST /query/agent` — tool-use path: the LLM gathers evidence itself via `search_documents` / `fetch_chunk` function calls (multi-turn ReAct loop), then answers under the same grounding contract plus a full tool trace. Prototype-grade; see `scripts/agent_smoke.py` for a live check.
- `POST /documents/upload?background=true` / `POST /documents/{id}/ingest?background=true` — enqueue ingestion on the Celery/Redis worker and return immediately (`processing`/`running`); without a worker they fall back to synchronous ingestion. Follow progress at `GET /documents/{id}/jobs/{job_id}/stream` (SSE) or `GET /documents/{id}/jobs`.
- `GET /ready` — per-dependency health (PostgreSQL, Qdrant, keyword search, Redis; Redis is informational since sync fallback exists).
- `GET /metrics` — per-stage timing attribution (avg/p95/max/last): `ingest.parse/clean/chunk/postgres/embed/qdrant/keyword`, `query.retrieval/rerank/evidence/generation/grounding`; plus `llm_usage` token/cost totals per caller label (`generation`/`grounding`/`agent`) and event `counters` (e.g. `safety.question_blocked.*`, `safety.chunks_dropped`, `safety.pii_redacted`).

### Safety

All three query endpoints screen requests deterministically (`app/safety/`, no LLM calls, no extra latency to speak of):

1. **Question gate** — disallowed requests (explosives, malware, hacking how-tos, …) and instruction-takeover phrasing are refused in ~0ms before any retrieval or model spend.
2. **Evidence filter** — retrieved chunks carrying prompt-injection payloads are dropped before they reach any model prompt (fixed pipeline and agent alike).
3. **PII redaction** — emails, phones, SSNs, API keys, and Luhn-checked card numbers are masked (`[REDACTED]`) in answers and citation excerpts.

Blocks are counted in `GET /metrics` → `counters`. Proven live against uploaded attack documents; red-team battery in `tests/unit/test_safety.py` (28 tests, incl. a capturing-mock proof that tainted text never reaches the prompt).

## Web Frontend

A Next.js app lives in `frontend/` (proxy routes in `frontend/src/app/api/` forward to the backend at `http://127.0.0.1:8000`). It provides:

- Free-text question input (Enter to submit), streamed stage labels
- Answer rendering with **grounded** / confidence badges and a confidence bar
- Collapsible **Claims & Grounding** section with per-claim status
- **Citations** listing with source metadata
- A recent-questions history sidebar
- Document upload with background-ingest progress (elapsed timer + status polling)
- **Usage tab** on every answer (prompt/completion/total tokens + estimated cost) and a **cumulative usage pill** in the header (per-source breakdown, auto-refreshes)
- **Shared-corpus toggle** — default OFF (answers only from your upload); when ON, the shared corpus becomes searchable while your upload stays anchored (other uploads remain excluded server-side)

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000 (NEXT_PUBLIC_API_URL points at :8000)
```

## Setup

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (PostgreSQL, Qdrant, Redis)
- Node.js 18+ (frontend)
- An OpenRouter API key (free `:free` models work; paid models bill per token)

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix

# Install dependencies
pip install -e .[dev]

# Copy environment template
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

### Start Infrastructure

```bash
# Start PostgreSQL, Qdrant, and Redis (Celery broker)
docker compose up -d

# Initialize the PostgreSQL schema and services
make init-db
```

### Ingest the Corpus

```bash
# Generate / embed the document corpus (downloads BAAI/bge-m3 on first run)
make corpus
# or: python scripts/generate_corpus.py
```

### Run the API (+ worker for background ingest)

```bash
make run
# uvicorn app.main:app --reload --port 8000

# In a second terminal — Celery worker for ?background=true uploads/evals:
make worker
# celery -A app.celery_app worker --queues ingestion,evaluation,benchmark
# Windows one-liner: .\start.ps1 -Worker
```

The API is then available at `http://127.0.0.1:8000` (docs at `/docs`, readiness at `/ready`, stage timings at `/metrics`). Start the frontend from `frontend/` (`npm run dev`). Without a worker, background uploads fall back to synchronous ingestion.

See [DEPLOY.md](DEPLOY.md) for free production deployment (Oracle ARM VM +
Vercel): `Dockerfile`, `docker-compose.prod.yml`, Caddy HTTPS, and ops
commands (`make prod-up`, `make prod-logs`, …).

## Configuration

Configuration is loaded from `app/config/settings.py`, overridable via environment variables / `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required. Key for LLM generation & judge |
| `OPENROUTER_MODEL` | `nex-agi/nex-n2.5-mini:free` | LLM used for generation (`:free` = $0; other models bill) |
| `DATABASE_URL` | `postgresql+psycopg://…` | PostgreSQL metadata connection |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim, local, CPU) |
| `EMBEDDING_DIM` | `1024` | Vector dimension |
| `EMBEDDING_BATCH_SIZE` | `64` | Encode batch size (CPU throughput) |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder reranker |
| `QDRANT_URL` | `http://localhost:6333` | Vector store |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0(1)` | Background task queue + results |
| `VECTOR_TOP_K` / `BM25_TOP_K` | `20` / `20` | Per-retriever candidate pool |
| `FINAL_CONTEXT_K` | `5` | Chunks sent to the LLM |
| `LLM_REASONING_EFFORT` | `high` | Reasoning effort (`high`\|`medium`\|`low`\|`none`); lower is faster, `high` = provider default. Measured: `medium` ≈23% faster, equal quality on a 2-question sample — too thin to flip; re-test before lowering |

## Performance Notes

- **Embedding model is a process-wide singleton.** BGE-M3 is cached at module scope via `lru_cache`, so it loads once (warmed at startup when `WARMUP_MODELS=true`) and is shared across requests. First boot still pays the load; keep the API + worker running instead of restarting per upload.
- **Ingest batch + threads.** `EMBEDDING_BATCH_SIZE=64` with torch pinned to all CPU cores. Large PDFs are still CPU-bound (expect tens of seconds for 100+ chunks); `GET /metrics` shows whether `ingest.embed` dominates.
- **Background ingest + SSE.** `?background=true` moves parse→embed→index off the request (Celery/Redis) with sync fallback when no worker runs. `/query/stream` and job streams emit progress so the UI never looks hung.
- **LLM judge calls run concurrently.** Per-claim grounding checks execute in a thread pool (`min(8, n_claims)` workers) rather than sequentially.
- **Reasoning effort is tunable.** Generation latency is ~75% of query time and dominated by the reasoning model's thinking tokens. `LLM_REASONING_EFFORT=medium` measured ≈23% faster generation with equal answer quality on a small sample; the default stays `high` until a larger comparison justifies flipping.
- **HTTP connections are pooled.** The OpenRouter client reuses a keep-alive `httpx.Client` instead of opening/closing a connection per call.
- Latency is dominated by LLM API round-trips (generation + one judge call per cited claim).

## Testing

```bash
make test          # Full test suite
make test-unit     # Unit tests only (fast, no services)
make test-integration  # Integration tests (requires Docker)
make test-eval     # Evaluation tests
make lint          # Ruff lint
make typecheck     # Mypy
make test-coverage # Tests with coverage report
```

## Evaluation

The evaluation suite uses RAGAs metrics and a golden dataset at `evals/dataset/golden.jsonl` (60 questions: easy/medium/hard plus unanswerable):

| Metric | Description |
|--------|-------------|
| `answer_correctness` | Semantic alignment between generated and reference answer |
| `answer_relevancy` | How well the answer addresses the question |
| `context_precision` | Relevance of retrieved chunks to the answer |
| `context_recall` | Coverage of reference context in retrieved chunks |
| `faithfulness` | Adherence to retrieved context without fabrication |
| `grounded_answer_rate` | Fraction of answers fully supported by evidence |

```bash
make eval          # Run the full evaluation pipeline
make benchmark     # Generate evals/reports/benchmark.md
make baseline      # Show current baseline
make baseline-update  # Record latest eval as the new baseline
```

Regression thresholds live in `evals/thresholds.yaml` (single source of truth, loaded by `app/evaluation/regression.py:load_thresholds()`); the `check_regression` tool alerts on statistically significant degradation. Built-in `DEFAULT_THRESHOLDS` are fallback-only, with unit tests asserting the two stay in sync.

## Portfolio and interview guide

This repository is designed to be read as an engineering case study, not just as a demo chatbot. Its strongest story is the complete path from untrusted documents to an answer whose evidence is explicit, inspectable, and evaluated.

### 30-second project pitch

> AskMyDocs is a grounded RAG service for technical documentation. It parses mixed-format documents, creates deterministic chunks, retrieves evidence with both dense vector search and BM25, fuses and reranks candidates, generates structured answers, and validates citations at the claim level. The system includes mockable dependencies, regression-aware evaluation, API tests, and production-oriented observability so quality can be measured instead of assumed.

### What this project demonstrates

| Engineering signal | Evidence in the repository |
|---------------------|----------------------------|
| Retrieval quality | Qdrant vector search, Postgres tsvector keyword search, reciprocal-rank fusion, reranking, evidence selection |
| Reliability | Deterministic chunk IDs, content hashes, typed schemas, retry handling, safe refusals |
| Model discipline | Lazy model loading, injectable embedders and LLMs, dimension checks, mock mode |
| Grounded generation | Structured JSON output, inline citations, claim extraction, citation validation, grounding status |
| Evaluation thinking | Golden dataset, retrieval metrics, citation metrics, Ragas adapter, baselines, regression checks |
| Product engineering | FastAPI routes, document ingestion, request IDs, health/readiness endpoints, persistence |
| Operational awareness | Docker Compose services, structured logging, tracing abstraction, CI workflows, benchmark reports |
| Testing maturity | Unit, integration, API, storage, retrieval, generation, grounding, and evaluation tests |

### Recommended interview walkthrough

Use this order when presenting the project:

1. Start with the failure mode: a fluent answer is not necessarily a correct answer, so the system must retrieve evidence and make unsupported claims visible.
2. Show ingestion: parsers preserve source metadata, the cleaner removes noise, and the chunker produces stable IDs such as `authentication-guide:2`.
3. Explain why retrieval is hybrid: BM25 catches exact identifiers, error codes, and configuration names; embeddings catch paraphrases and semantic matches.
4. Explain RRF and reranking: the system combines complementary candidate lists, then uses a cross-encoder to improve ordering before context is selected.
5. Show the answer contract: the LLM returns structured data with `[C1]` markers rather than an opaque string.
6. Show grounding: claims are checked against cited chunks, and the system can return `grounded`, `partially_grounded`, `ungrounded`, or `refused`.
7. Finish with evaluation: the golden dataset and regression checks make retrieval and grounding changes measurable.

### Questions this project should answer clearly

- Why use BM25 and embeddings together?
- Why use reciprocal-rank fusion instead of adding raw scores?
- Why rerank only a small candidate pool?
- How are duplicate chunks and repeated documents handled?
- What happens when the LLM returns malformed JSON?
- How does the system behave when evidence is insufficient?
- How can a citation be wrong even when the answer sounds plausible?
- How do you test the system without downloading models or calling a paid API?
- Which metrics reveal retrieval failure versus generation failure?
- What are the latency and cost trade-offs of claim-level validation?
- How would you scale ingestion and query traffic independently?
- What would you change for multi-tenant data isolation and access control?

### Resume-ready bullets

Adapt these only after confirming the implementation and measured results in your own environment:

- Built a production-oriented RAG service for mixed-format technical documentation using FastAPI, PostgreSQL (metadata + tsvector keyword search), Qdrant, sentence-transformer embeddings, and an OpenAI-compatible LLM client.
- Implemented hybrid retrieval with BM25, dense vector search, reciprocal-rank fusion, cross-encoder reranking, and token-budgeted evidence selection.
- Designed claim-level citation validation with structured outputs, deterministic fallbacks, safe refusal behavior, and explicit groundedness states.
- Created a deterministic evaluation framework with a golden question set, retrieval/citation metrics, mock LLM execution, benchmark reports, and regression baselines.
- Added API, storage, retrieval, generation, grounding, and evaluation tests so the system can be developed without live model calls or external service access.
- Built a tool-use (ReAct) query path with function calling, a token/cost observability layer (per-answer usage, cumulative header totals, `/metrics` aggregation), and deterministic safety screening (question gate, evidence filter, PII redaction) proven by live attack probes.

Do not claim latency, accuracy, scale, cost savings, or production usage unless those numbers are recorded in `evals/reports/benchmark.md` or another reproducible report.

### Final portfolio checklist

Before sharing this repository:

- Replace placeholder configuration values and remove all secrets from commits.
- Run the full test suite and record the result.
- Run linting and type checking.
- Run a mock evaluation and commit only reproducible reports.
- Run a live evaluation only when the API key and model usage are authorized.
- Review the README commands from a clean virtual environment.
- Add a short architecture diagram or recorded demo if the repository is being used in an application.
- Be ready to discuss one failure, one trade-off, and one improvement you would make next.

This section is portfolio guidance, not official OpenAI hiring guidance. Hiring decisions depend on the role, interview performance, and the broader evidence of a candidate's work.

## Limitations

- **Corpus freshness**: Ingested documents are snapshotted at ingest time; updates require re-ingestion.
- **LLM dependency**: Answer and grounding quality depend on OpenRouter model capabilities.
- **Upload cold start**: the first upload after a worker (re)start pays the embedding-model load (~1–2 min); the worker pre-warms at boot and later uploads take seconds.
- **Chunk boundary artifacts**: Answers may reference partial sentences near chunk boundaries.
- **Grounding strictness**: An answer is only marked `grounded` when every factual claim carries a citation the judge can verify; models that tag only the final sentence yield `partially_grounded`.
