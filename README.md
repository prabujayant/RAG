# AskMyDocs — RAG Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system for querying technical documentation with hybrid search, reranking, and grounded citations.

## Overview

AskMyDocs answers natural-language questions about software products using trusted documentation. It combines dense vector search with sparse BM25 retrieval, reranks results with a cross-encoder model, and grounds responses with precise inline citations extracted from retrieved chunks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Query                                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Hybrid Retrieval (Vector + BM25)                           │
│  • Vector: SentenceTransformer embeddings (all-MiniLM-L6-v2) │
│  • BM25: keyword search via rank_bm25                        │
│  • Reciprocal Rank Fusion combines scores                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Reranking (Cross-Encoder)                                   │
│  • ms-marco-MiniLM-L-6-v2 reorders top-k candidates          │
│  • Maximizes relevance to query intent                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Answer Generation (LLM)                                     │
│  • OpenRouter API (GPT-4o-mini, Claude 3 Haiku, etc.)       │
│  • Strict prompt isolation between experiments              │
│  • Citation labels extracted from returned chunks            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Citation Extraction                                        │
│  • Labeled spans from chunks map to source files/sections   │
│  • Results returned with source tracking metadata            │
└─────────────────────────────────────────────────────────────┘
```

## Hybrid Retrieval Rationale

Pure vector search excels at semantic similarity but misses domain-specific terminology. BM25 excels at exact keyword matches. Combining both via Reciprocal Rank Fusion captures the strengths of each approach:

- **Vector (dense)**: Handles paraphrases, synonyms, conceptual matches
- **BM25 (sparse)**: Handles exact technical terms, product names, version numbers
- **RRF fusion**: Non-parametric combination, no training required

## Reranking

Initial retrieval returns top-50 candidates. The cross-encoder reranker scores each (query, passage) pair and returns the top-10, dramatically improving precision for question-answering workloads.

## Grounding and Citations

Answers include inline citation markers (e.g., `[source:api-errors.md#L15]`) that trace claims back to specific chunks. This enables:

- **Auditability**: Every claim is verifiable against source docs
- **Reduced hallucination**: LLM conditioned on retrieved context
- **User trust**: Readers can inspect original context

## Evaluation

The evaluation suite uses RAGAs metrics:

| Metric | Description |
|--------|-------------|
| `answer_correctness` | Semantic alignment between generated and reference answer |
| `answer_relevancy` | How well the answer addresses the original question |
| `context_precision` | Relevance of retrieved chunks to the answer |
| `context_recall` | Coverage of reference answer context in retrieved chunks |
| `faithfulness` | Adherence to retrieved context without fabrication |

### Regression Detection

`evals/thresholds.yaml` defines minimum acceptable values and tolerances per metric. The `check_regression` tool compares experiment runs against baselines and alerts on statistically significant degradation.

## Setup

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for PostgreSQL and eval runner)
- OpenRouter API key

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

### Database Setup

```bash
# Initialize PostgreSQL schema
docker compose up -d postgres
docker compose exec postgres psql -U askmydocs -d askmydocs -f /docker/postgres/init.sql

# Initialize the database
python scripts/init_db.py
```

### Corpus Ingestion

```bash
# Generate corpus embeddings (downloads ~90MB model on first run)
python scripts/generate_corpus.py

# Or with custom corpus path
python scripts/generate_corpus.py --corpus data/corpus
```

## Configuration

Configuration is loaded from `app/config/settings.py` with environment variable overrides. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required. API key for LLM generation |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `sentence-transformers/...` | HuggingFace model for embeddings |
| `RERANKER_MODEL` | `cross-encoder/...` | Cross-encoder for reranking |
| `LLM_MODEL` | `openrouter/gpt-4o-mini` | Default LLM for answer generation |
| `VECTOR_TOP_K` | `50` | Initial retrieval pool size |
| `RERANK_TOP_K` | `10` | Final number of chunks for generation |

See `.env.example` for the full list.

## API

AskMyDocs exposes a single `/ask` endpoint:

```http
POST /ask
Content-Type: application/json

{
  "question": "How do I rotate API keys?",
  "top_k": 5,
  " llm_model": "openrouter/gpt-4o-mini"
}
```

Response:

```json
{
  "answer": "API keys can be rotated via...",
  "citations": [
    {"source": "api-keys.md", "chunk_id": "ck_abc123", "text": "..."},
    {"source": "security-guide.md", "chunk_id": "ck_def456", "text": "..."}
  ],
  "retrieval_metrics": {
    "hit_rate": 0.85,
    "mrr": 0.72
  }
}
```

## Testing

```bash
# Run all tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests (requires Docker)
make test-integration

# Evaluation tests (requires Docker)
make test-eval

# Lint
make lint

# Type check
make typecheck

# Full CI pipeline
make ci
```

## CI/CD

GitHub Actions workflows:

- **`ci.yml`**: Lint → Type check → Unit tests → Integration tests → Evaluation tests
- **`evaluation.yml`**: Nightly benchmark runs with Docker services

## Limitations

- **Corpus freshness**: Ingested documents are snapshotted at ingest time; updates require re-ingestion
- **LLM dependency**: Answer quality depends on OpenRouter model capabilities
- **No streaming**: `/ask` is synchronous; large corpora may have higher latency
- **Chunk boundary artifacts**: Answers may reference partial sentences near chunk boundaries
- **English-only**: Corpus and queries are English-language; multilingual support untested
