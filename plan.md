# AskMyDocs — Implementation Plan

Production-oriented, domain-specific RAG, grounding & evaluation platform.

## TL;DR

Greenfield build in the empty workspace `c:\Users\salag\OneDrive\Desktop\Germany\RAG`.
Full-stack RAG platform: **FastAPI** + **PostgreSQL** (metadata) + **Qdrant** (vectors) +
**OpenSearch** (BM25) + **BGE-M3** embeddings + **BGE cross-encoder** reranker +
**OpenRouter** LLM + **Ragas** evaluation + optional **Langfuse** + **Docker Compose** +
**GitHub Actions** CI with quality gates and regression detection.

Domain: enterprise software platform docs (~30 documents, 50-question golden dataset).
Four benchmark experiments (vector-only / BM25-only / hybrid / final grounded system).
`MockLLM` for deterministic, key-free tests.

## Confirmed decisions

- **Package manager**: `uv` — fast, modern; `pyproject.toml` + `uv.lock`, with the `Makefile` wrapping uv commands.
- **OpenRouter key**: user will provide → run live eval, generate real measured benchmark numbers (no fabrication). `MockLLM` still used for deterministic tests.
- **Corpus**: mixed formats — the generator emits Markdown plus a few PDF/DOCX/HTML files to exercise every parser end-to-end.
- Python 3.12, Pydantic v2, `openai` SDK pointed at OpenRouter base URL.
- Embeddings: `BAAI/bge-m3` via sentence-transformers; Reranker: `BAAI/bge-reranker-v2-m3` via sentence-transformers `CrossEncoder` (both configurable via env).
- Ragas isolated behind `app/evaluation/ragas_eval.py` interface (modern API: `EvaluationDataset.from_list`, `evaluate(dataset=..., metrics=[...], llm=LangchainLLMWrapper(ChatOpenAI(base_url=openrouter)))`).
- Citation policy (deterministic): claim without supporting citation → flagged; if ≥1 claim unsupported after validation → answer marked `grounded=false`; refusal when evidence insufficient.
- Unanswerable questions: retrieval confidence gate + LLM instructed to refuse; evaluated separately.
- Langfuse optional — no-op tracer when credentials absent.
- Baseline stored in `evals/baselines/`; regression = current < baseline − tolerance OR below absolute threshold; baseline updates explicit via `make baseline-update`.
- No LangChain in app code (only Ragas' internal langchain wrapper for eval).

## Architecture

```
                         ┌──────────────────────┐
                         │      Documents       │
                         │ PDF / DOCX / HTML    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Document Ingestion   │
                         │ Parse → Clean → Chunk │
                         │ Metadata Preservation│
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
                ┌───────────────┐       ┌────────────────┐
                │ BGE-M3        │       │ OpenSearch     │
                │ Embeddings    │       │ BM25 Index     │
                └───────┬───────┘       └───────┬────────┘
                        │                       │
                        ▼                       ▼
                 ┌─────────────┐        ┌──────────────┐
                 │ Qdrant      │        │ BM25 Results │
                 │ Vector DB   │        └──────┬───────┘
                 └──────┬──────┘               │
                        │                      │
                        └──────────┬───────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Hybrid Fusion       │
                         │ RRF / configurable  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │ Re-Ranker           │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence Selection  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ OpenRouter LLM      │
                         │ Answer Generation   │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Claim / Citation    │
                         │ Validation          │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Final Answer        │
                         │ + Citations         │
                         └─────────────────────┘

             OFFLINE / CI EVALUATION PIPELINE

        Golden Dataset
              │
              ▼
        Retrieval Tests
              │
              ▼
      End-to-End RAG Tests
              │
              ▼
       Ragas Evaluation
              │
              ▼
      Custom Metrics
              │
              ▼
        Regression Check
              │
        ┌─────┴──────┐
        │            │
       PASS         FAIL
        │            │
      Merge      Block CI
```

## Phases

### Phase 0 — Scaffolding
- `pyproject.toml` (deps: fastapi, uvicorn, pydantic v2, sqlalchemy 2, psycopg, qdrant-client, opensearch-py, sentence-transformers, openai, ragas, langfuse, pytest, pytest-asyncio, ruff, mypy, httpx, python-multipart, pypdf, python-docx, beautifulsoup4, markdown, tenacity, numpy, pandas)
- `docker-compose.yml`: postgres:16, qdrant, opensearch (single-node, disabled security)
- `.env.example`, `.gitignore`, `Makefile` (setup, dev, test, lint, typecheck, eval, baseline-update, benchmark)
- `app/config/settings.py` — pydantic-settings, all env vars from spec §31

### Phase 1 — Domain corpus + eval dataset
- `scripts/generate_corpus.py` → `data/corpus/` ~30 markdown docs (auth, OAuth, API, errors, config, deployment, monitoring, security, user mgmt, DB, troubleshooting, rate limits, webhooks) with headings/tables/examples
- `evals/dataset/golden.jsonl` — 50 questions: 10 easy, 20 medium, 10 hard, 10 unanswerable; schema per spec §21; `relevant_chunk_ids` resolved post-ingestion via `scripts/resolve_relevant_chunks.py` (chunk ids deterministic: `{doc_id}:{chunk_index}`)

### Phase 2 — Core models
- `app/retrieval/models.py`: `RetrievalResult` (chunk_id, document_id, text, score, source, page_number, section, retriever)
- `app/generation/schemas.py`: `Citation`, `AnswerClaim`, `AnswerResponse` (per spec §13)
- `app/db/models.py`: documents, chunks, ingestion_jobs, queries, evaluation_runs (SQLAlchemy)

### Phase 3 — Ingestion
- `app/ingestion/parsers/`: pdf.py (pypdf, page numbers), docx.py (python-docx), html.py (bs4), markdown.py
- `app/ingestion/cleaner.py`: normalize whitespace, strip boilerplate
- `app/ingestion/chunker.py`: fixed-size + paragraph-aware strategies, section-boundary preservation, configurable CHUNK_SIZE/CHUNK_OVERLAP
- `app/ingestion/pipeline.py`: parse → clean → chunk → metadata (content_hash, page, section) → embed → index Qdrant + OpenSearch + PG; ingestion_jobs state tracking

### Phase 4 — Storage & embeddings
- `app/embeddings/embedder.py`: sentence-transformers wrapper, lazy model load, configurable model
- `app/db/` — SQLAlchemy engine/session, init script `scripts/init_db.py`
- `app/retrieval/vector.py`: Qdrant client, collection init (cosine), upsert, search with metadata filter
- `app/retrieval/bm25.py`: OpenSearch index init (standard analyzer), bulk index, match query, filter support

### Phase 5 — Retrieval pipeline
- `app/retrieval/fusion.py`: RRF (configurable k), merge duplicates
- `app/retrieval/hybrid.py`: orchestrates bm25 + vector → fusion
- `app/retrieval/reranker.py`: CrossEncoder, ENABLE_RERANKER toggle, RERANK_TOP_K
- `app/retrieval/evidence.py`: evidence selection — FINAL_CONTEXT_K, MAX_CONTEXT_TOKENS, diversity/source coverage (dedupe per document)

### Phase 6 — Generation
- `app/generation/client.py`: `LLMClient` protocol + `OpenRouterClient` (openai SDK, base_url from env) + `MockLLM` (tests)
- `app/generation/prompts.py`: system prompt with grounding rules, citation format `[C1]`, refusal behavior, JSON output schema
- `app/generation/service.py`: build context from evidence, call LLM, parse structured JSON (robust to malformed JSON)

### Phase 7 — Grounding
- `app/grounding/claims.py`: split answer into claims (LLM-assisted + deterministic fallback), attach citation ids
- `app/grounding/citation_validator.py`: verify each citation's chunk supports the claim (LLM judge via OpenRouter + deterministic overlap fallback); returns supported/partially_supported/unsupported
- `app/grounding/grounding_validator.py`: aggregate → grounded flag, confidence

### Phase 8 — API
- `app/api/routes/`: health, documents (upload/ingest/list/get), query
- `app/api/schemas/`: request/response models (query response per spec §18)
- `app/main.py`: FastAPI app, error handlers (no stack traces), request_id logging middleware

### Phase 9 — Observability
- `app/observability/tracing.py`: Langfuse tracer when creds present, no-op otherwise; spans: ingestion, embedding, bm25_retrieval, vector_retrieval, hybrid_fusion, reranking, generation, citation_validation, grounding_validation
- `app/observability/logging.py`: structured logging (request_id, query_id, latencies, grounding status); never log secrets

### Phase 10 — Evaluation
- `app/evaluation/datasets.py`: load golden.jsonl
- `app/evaluation/retrieval_metrics.py`: recall@k, precision@k, MRR, NDCG (pure functions)
- `app/evaluation/citation_metrics.py`: citation correctness, completeness, precision, grounded answer rate
- `app/evaluation/ragas_eval.py`: isolated Ragas wrapper (faithfulness, answer relevance, context relevance/recall)
- `app/evaluation/runner.py`: `python -m app.evaluation.run` — runs 4 experiments (A/B/C/D), collects raw outputs + metrics, writes `evals/reports/`, compares baseline, non-zero exit on failure
- `app/evaluation/regression.py`: threshold + regression checks
- `evals/baselines/` + `evals/reports/benchmark.md` generation

### Phase 11 — Tests
- `tests/unit/`: chunker, metadata, RRF fusion, ranking, citation parsing, claim extraction, citation validation, metrics, config
- `tests/integration/`: full pipeline with MockLLM + real Qdrant/OpenSearch/PG via docker compose; mocked LLM responses
- `tests/evaluation/`: dataset validity, metric correctness, regression logic

### Phase 12 — CI
- `.github/workflows/ci.yml`: lint (ruff) + typecheck (mypy) + unit tests + integration tests (docker services) + eval suite + quality gates
- `.github/workflows/evaluation.yml`: nightly/full eval with baseline comparison
- Quality thresholds in `evals/thresholds.yaml` (spec §27)

### Phase 13 — Docs
- `README.md` per spec §41 (overview, architecture, hybrid rationale, reranking, grounding, citations, evaluation, setup, config, API, testing, CI, limitations)
- `evals/reports/benchmark.md` — real measured numbers only

### Phase 14 — Verification
1. `make lint` + `make typecheck` clean
2. `make test` (unit) green
3. `docker compose up -d` → `make init-db` → integration tests green
4. `make eval` with MockLLM (no key) → report generated, exit codes correct
5. Live eval if OpenRouter key provided
6. Validate CI YAML syntax, README commands against repo

## Time estimate

| Phase | Est. time |
|---|---|
| 0 — Scaffolding | ~45 min |
| 1 — Corpus + golden dataset | ~1.5 h |
| 2–3 — Core models + ingestion | ~2 h |
| 4–5 — Embeddings, Qdrant, OpenSearch, hybrid/RRF/reranker/evidence | ~2 h |
| 6–7 — Generation + grounding | ~2 h |
| 8–9 — API + observability | ~1 h |
| 10 — Evaluation | ~2 h |
| 11 — Tests | ~1.5 h |
| 12 — CI workflows | ~45 min |
| 13–14 — README + verification + fixes | ~1.5 h |

**Active implementation: ~12–16 hours** (2–3 working days).
**Additional wall-clock**: model downloads (BGE-M3 + reranker, ~4.6 GB), Docker image pulls, live Ragas eval on 50 questions via OpenRouter (30–90 min), debugging buffer.
**Total wall-clock: ~1.5–3 days.**

## Relevant files (key)

- `app/config/settings.py`, `app/retrieval/{models,bm25,vector,hybrid,fusion,reranker,evidence}.py`
- `app/ingestion/{pipeline,chunker,cleaner}.py`, `app/ingestion/parsers/*`
- `app/generation/{client,prompts,schemas,service}.py`
- `app/grounding/{claims,citation_validator,grounding_validator}.py`
- `app/evaluation/{runner,retrieval_metrics,citation_metrics,ragas_eval,regression,datasets}.py`
- `app/api/routes/*`, `app/main.py`, `app/db/models.py`
- `evals/dataset/golden.jsonl`, `evals/thresholds.yaml`, `evals/baselines/*`
- `docker-compose.yml`, `pyproject.toml`, `Makefile`, `.env.example`, `.github/workflows/{ci,evaluation}.yml`

## Verification

1. `make lint && make typecheck` — zero errors
2. `make test` — all unit tests pass (no API keys)
3. `docker compose up -d && make init-db && make test-integration` — green
4. `make eval` — runs 4 experiments, writes reports, correct exit codes (0 pass / 1 fail)
5. `make benchmark` — benchmark.md with only measured values
6. CI YAML validated; README commands verified against repo

## Scope boundaries

- **Included**: everything in spec §45 Definition of Done
- **Excluded**: Kubernetes/Terraform/cloud infra, authentication system, notebook/Streamlit UI, LangChain in app code