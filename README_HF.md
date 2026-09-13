# AskMyDocs

Grounded RAG question-answering platform for technical documentation.

**Features:**
- Hybrid retrieval: BM25 (Postgres tsvector) + dense vectors (Qdrant) with RRF fusion
- Cross-encoder reranking (MiniLM or BGE-v2-m3)
- Grounded answers with source citations
- Document upload with background ingestion
- SSE streaming for query stages

## Quick Start

The Space runs all services (Postgres, Qdrant, Redis, FastAPI, Celery) in a single container. Set these **Space secrets**:

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | LLM key for answer generation ([openrouter.ai](https://openrouter.ai)) |
| `HF_TOKEN` | No | Enables build-time model caching (accept BAAI model licenses first) |
| `HF_BUCKET` | No | Storage bucket name for persistent state (e.g. `user/askmydocs-data`) |
| `RESTORE_ON_BOOT` | No | Set `true` to restore state from bucket on restart |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`; credentials are auto-disabled for wildcard origins) |

## API Endpoints

- `GET /health` — liveness check
- `GET /ready` — readiness (postgres, qdrant, keyword, redis)
- `POST /query` — grounded question answering
- `POST /query/stream` — SSE streaming query
- `POST /query/agent` — tool-use (agentic) query path with tool trace
- `POST /documents/upload` — upload a document (add `?background=true` for async)
- `GET /metrics` — per-stage timings + LLM token/cost totals + event counters

## Architecture

```
Caddy (:7860) → uvicorn (127.0.0.1:8000) → FastAPI
                                            ├─ PostgreSQL (tsvector keyword search)
                                            ├─ Qdrant (dense vector search)
                                            ├─ Redis + Celery (background tasks)
                                            └─ OpenRouter (LLM)
```
