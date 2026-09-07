# AskMyDocs — RAG Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system for querying technical documentation with hybrid search, cross-encoder reranking, OpenAI-style LLM-judge grounding, and per-claim citations.

## Overview

AskMyDocs answers natural-language questions about software products using trusted documentation. It combines dense vector search (Qdrant) with sparse BM25 retrieval (OpenSearch), reranks results with a cross-encoder model, generates structured answers via OpenRouter, and grounds each answer with an LLM judge that verifies every claim against the cited evidence.

## Architecture

```
                    User Query
                        │
                        ▼
        ┌─────────────────────────────────────────────┐
        │  Hybrid Retrieval (Vector + BM25)           │
        │  • Vector : BAAI/bge-m3 embeddings → Qdrant │
        │  • BM25   : OpenSearch keyword search       │
        │  • Reciprocal Rank Fusion combines scores   │
        └───────────────────┬─────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Reranking (Cross-Encoder)                  │
        │  • BAAI/bge-reranker-v2-m3 reorders top-k   │
        └───────────────────┬─────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │  Grounded Generation (LLM)                  │
        │  • OpenRouter (default: openai/gpt-4o-mini) │
        │  • Structured JSON with inline [C1] markers │
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
4. **Embedding & indexing** — embeds chunks with BGE-M3 and stores them in both Qdrant (dense vectors) and OpenSearch (BM25).

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
  "model": "openai/gpt-4o-mini"
}
```

Request fields:

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | The user's question (5–2000 chars) |
| `top_k` | int? | Override evidence chunk count (1–50) |
| `temperature` | float? | LLM sampling temperature override |

## Web Frontend

A single-file, dark-themed web UI ships at `frontend/index.html`. It connects to the running API at `http://127.0.0.1:8000` and provides:

- Free-text question input (Enter to submit)
- Answer rendering with **grounded** / confidence badges and a confidence bar
- Collapsible **Claims & Grounding** section with per-claim status
- **Citations** listing with source metadata
- A recent-questions history sidebar

To use it, start the API, then open `frontend/index.html` in a browser (or serve the folder statically).

## Setup

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (PostgreSQL, Qdrant, OpenSearch)
- An OpenRouter API key

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
# Start PostgreSQL, Qdrant, and OpenSearch
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

### Run the API

```bash
make run
# uvicorn app.main:app --reload --port 8000
```

The API is then available at `http://127.0.0.1:8000` (docs at `/docs`). Load the frontend from `frontend/index.html`.

## Configuration

Configuration is loaded from `app/config/settings.py`, overridable via environment variables / `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required. Key for LLM generation & judge |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | LLM used for generation |
| `DATABASE_URL` | `postgresql+psycopg://…` | PostgreSQL metadata connection |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim) |
| `EMBEDDING_DIM` | `1024` | Vector dimension |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `QDRANT_URL` | `http://localhost:6333` | Vector store |
| `OPENSEARCH_URL` | `http://localhost:9200` | BM25 store |
| `VECTOR_TOP_K` / `BM25_TOP_K` | `20` / `20` | Per-retriever candidate pool |
| `FINAL_CONTEXT_K` | `5` | Chunks sent to the LLM |

## Performance Notes

- **Embedding model is a process-wide singleton.** BGE-M3 (~15 s to load) is cached at module scope via `lru_cache`, so it loads once and is shared across all requests — warm queries skip reloading entirely.
- **LLM judge calls run concurrently.** Per-claim grounding checks execute in a thread pool (`min(8, n_claims)` workers) rather than sequentially.
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

The evaluation suite uses RAGAs metrics and a golden dataset at `evals/dataset/golden.jsonl`:

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

Regression thresholds are defined per metric; the `check_regression` tool alerts on statistically significant degradation.

## Portfolio and interview guide

This repository is designed to be read as an engineering case study, not just as a demo chatbot. Its strongest story is the complete path from untrusted documents to an answer whose evidence is explicit, inspectable, and evaluated.

### 30-second project pitch

> AskMyDocs is a grounded RAG service for technical documentation. It parses mixed-format documents, creates deterministic chunks, retrieves evidence with both dense vector search and BM25, fuses and reranks candidates, generates structured answers, and validates citations at the claim level. The system includes mockable dependencies, regression-aware evaluation, API tests, and production-oriented observability so quality can be measured instead of assumed.

### What this project demonstrates

| Engineering signal | Evidence in the repository |
|---------------------|----------------------------|
| Retrieval quality | Qdrant vector search, OpenSearch BM25, reciprocal-rank fusion, reranking, evidence selection |
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

- Built a production-oriented RAG service for mixed-format technical documentation using FastAPI, PostgreSQL, Qdrant, OpenSearch, sentence-transformer embeddings, and an OpenAI-compatible LLM client.
- Implemented hybrid retrieval with BM25, dense vector search, reciprocal-rank fusion, cross-encoder reranking, and token-budgeted evidence selection.
- Designed claim-level citation validation with structured outputs, deterministic fallbacks, safe refusal behavior, and explicit groundedness states.
- Created a deterministic evaluation framework with a golden question set, retrieval/citation metrics, mock LLM execution, benchmark reports, and regression baselines.
- Added API, storage, retrieval, generation, grounding, and evaluation tests so the system can be developed without live model calls or external service access.

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
- **No streaming**: `/query` is synchronous; responses are gated on LLM generation plus grounding.
- **Chunk boundary artifacts**: Answers may reference partial sentences near chunk boundaries.
- **Grounding strictness**: An answer is only marked `grounded` when every factual claim carries a citation the judge can verify; models that tag only the final sentence yield `partially_grounded`.
