# AskMyDocs — AGENTS.md

This file compiles high-signal, repo-specific guidance for future OpenCode sessions.
Every line below answers: "Would an agent likely miss this without help?"
If not, it's left out.

## Root manifests & config (read first)

- `pyproject.toml` — project deps, tool config (ruff, mypy, coverage), package discovery (`include = ["app*"]`). Dependencies include: fastapi, uvicorn, sqlalchemy 2, psycopg, qdrant-client, sentence-transformers, torch, openai, celery[redis]>=5.4.
- `Makefile` — dev/lint/test/deploy commands. Key targets: `make start` (infra + API), `make run` (API only), `make worker` (Celery), `make lint`/`format`/`typecheck`, `make test`/`test-unit`/`test-integration`, `make corpus`, `make eval`, `make benchmark`, `make init-db`, `make prod-*`.
- `docker-compose.yml` — dev stack: postgres, qdrant, redis. **Daemon must be running** (`docker compose up -d`); if `docker` becomes unreachable, restart Docker Desktop.
- `docker-compose.prod.yml` — production stack: postgres, qdrant, redis, api, worker, caddy. No `--reload`. Caddy on 80/443 → api:8000.
- `DEPLOY.md` — full Oracle VM + Vercel deployment guide. See before deploying.
- `RUN_ME.md` — quick-start dev guide (venv + docker compose + make start / make worker).
- `plan.md` — long-term implementation plan with phases and architecture.
- `.env.example` — never commit .env. Copy to `.env` and fill. `POSTGRES_PASSWORD` is sticky on first volume init.
- `opencode.json` — OpenCode provider config (Ollama + tokenharbor). Local LLM models: `qwen2.5-coder:7b` (tool-capable, first in picker), `gemma3:4b` (no tool support, backup).

## Developer setup (non-obvious)

1. **One-shot start**: `make start` (starts Docker infra + API). Or: `python scripts/hf_space_stage.py --output ./hf_staging` for HF Spaces staging.
2. **Ven + deps**: `python -m venv .venv && .venv\Scripts\activate && pip install -e ".[dev]"`
3. **Env**: `cp .env.example .env` → set `OPENROUTER_API_KEY`. Leave `DATABASE_URL`/`QDRANT_URL`/`CELERY_*` at localhost defaults; they're overridden by compose.
4. **Initialize DB**: `make init-db` (or `python scripts/init_db.py`). Then `python scripts/migrate_tsvector.py` (backfills keyword postings; idempotent).
5. **Ingest corpus**: `make corpus` (generates/embeds ~30 docs). Or `python scripts/generate_corpus.py` then `python scripts/ingest_corpus.py`.
6. **Start worker** (required for background uploads): `make worker` or `.\start.ps1 -Worker`. Without a worker, API falls back to synchronous ingestion (uploads still succeed but block).

## Test suite

- **Unit tests**: `make test-unit` — no external services needed. 438+ tests pass; 2 pre-existing failures in `test_phase4_storage.py` are vector-fake issues (proven pre-existing; not caused by current changes).
- **Integration tests**: `make test-integration` — requires Docker running (`docker compose up -d`). Connects PG + Qdrant + keyword search.
- **Slow/expensive**: `make test-fast` skips slow tests. `make eval` runs evaluation pipeline (needs OpenRouter key for live runs; `MockLLM` for deterministic tests).
- **Ruff**: `make lint` + `make format`. `pyproject.toml` has `per-file-ignores` for some scripts (E501 line-length).
- **Mypo**: `make typecheck`. `disallow_untyped_defs = false`; many functions are intentionally untyped.

## Architecture entrypoints

- **App**: `app/main.py` — FastAPI create_app(), lifespan warmup (`_warmup_models`), middleware (CORS, tracing, request_id), exception handlers.
- **Retrieval**: `app/retrieval/` — bm25.py (Postgres tsvector BM25Indexer), vector.py (Qdrant), hybrid.py (RRF fusion), reranker.py (CrossEncoder, `enable_reranker` toggle), evidence.py (FINAL_CONTEXT_K, MAX_CONTEXT_TOKENS, dedupe + greedy diversity-adjusted selection — `_score_with_diversity` is live, not dead).
- **Ingestion**: `app/ingestion/pipeline.py` — stages with timing (parse/clean/chunk/postgres/embed/qdrant/keyword). `scripts/ingest_corpus.py` — walks `data/corpus`, registers docs + pipeline.queues.
- **Generation**: `app/generation/service.py` — build context from evidence, call LLM, parse structured JSON (robust to malformed JSON). `client.py` — `generate()` + `chat_with_tools()` (native function calling; `LLM_REASONING_EFFORT` setting, default `high` = provider default; medium ≈23% faster on n=2, too thin to flip).
- **Agent (prototype)**: `app/generation/agent.py` (ReAct loop) + `app/generation/tools.py` (`search_documents`, `fetch_chunk`) + `POST /query/agent`. Verify live: `python scripts/agent_smoke.py`. Not wired into main `/query` or the UI.
- **Grounding**: `app/grounding/claims.py` (split into claims), `citation_validator.py` (LLM judge + deterministic fallback), `grounding_validator.py` (aggregate → grounded flag).
- **Safety**: `app/safety/` — deterministic question gate, evidence filter, PII redaction on all 3 query paths; counters in `/metrics`. Red-team battery: `tests/unit/test_safety.py`. Proven live vs uploaded attack docs.
- **Cost**: `app/observability/cost.py` — pricing table + per-call recording (labels `generation`/`grounding`/`agent`); `/metrics` → `llm_usage`; per-answer `usage` block; UI Usage tab + header cumulative pill.
- **Observability**: `app/observability/metrics.py` — per-stage timing registry + event `count()` counters, `GET /metrics`. `app/observability/tracing.py` — Langfuse spans when creds present.
- **Eval thresholds**: `evals/thresholds.yaml` is the single source of truth (`load_thresholds()`); `DEFAULT_THRESHOLDS` is fallback-only with sync tests. Do NOT reintroduce a second source.
- **API routes**: `app/api/routes/health.py`, `documents.py`, `query.py` — all read `process.env.NEXT_PUBLIC_API_URL` from the frontend proxy.

## Common gotchas (would trip an agent)

1. **Disk full mid-session**: 0 bytes free on C: due to 10.26GB HF cache + Docker data. Fixed by clearing cache and recovering ~17.8GB free. Keep an eye on free GB.
2. **Docker daemon unreachable**: happens after hibernation/sleep. Fix: restart Docker Desktop. Then `docker compose up -d`.
3. **pyproject.toml truncated to 0 bytes**: happened during a failed edit when disk was full. Recovered from `git show HEAD:pyproject.toml`. The `celery[redis]>=5.4` dep addition was lost; re-add if doing further edits.
4. **OpenSearch → tsvector migration (done, remnants removed)**: `app/retrieval/bm25.py` is Postgres `tsvector` (same public interface). `KeywordPosting` model in `app/db/models.py`. `scripts/migrate_tsvector.py` is idempotent + backfills. The temporary A/B gate (`scripts/compare_keyword_backends.py`, deleted after serving its purpose) caught a real bug: `plainto_tsquery` (AND) gave 0.03 top-5 agreement → fixed to OR-rewritten `websearch_to_tsquery` → 0.66 agreement. `opensearch-py` dep, settings fields, `.env` vars, and compose volume declarations are all removed; on-disk `rag_opensearch_data` volume (if any) is untouched — `docker volume rm` to clean manually.
5. **2 pre-existing test failures**: `tests/integration/test_phase4_storage.py` has 2 failures (`test_end_to_end_chunk_embed_index`, `test_ingestion_pipeline_writes_to_all_stores`) proven pre-existing via git stash check — not caused by current changes.
6. **OpenSearch fully removed**: no container, no service, no volume declarations, no dep. On-disk `rag_opensearch_data` volume (if any) is untouched — `docker volume rm` to clean manually.
7. **CORS**: `Settings.cors_origins` defaults to `"*"` for local dev, and `allow_credentials` auto-disables for wildcard origins (browsers reject `*` + credentials). In prod, set explicitly (e.g. `CORS_ORIGINS=https://askmydocs.vercel.app`). Frontend reads `NEXT_PUBLIC_API_URL`.
8. **Embedding model on CPU**: BGE-M3 is slow (~10+ min for 100 pages/100+ chunks). Batch size 64 + `torch.set_num_threads(cpu_count)` helps. Consider `ENABLE_RERANKER=false` on free-tier HF Spaces.
9. **Local LLM via Ollama**: `qwen2.5-coder:7b` supports tools; `gemma3:4b` does not. Config at `C:\Users\salag\.config\opencode\opencode.json`. Set `OLLAMA_NUM_CTX=8192` env var to match opencode context cap.
10. **HF Spaces deployment**: all-in-one container (Postgres + Qdrant + Redis + FastAPI + Celery + Caddy). `scripts/hf_space_stage.py` assembles the repo; `scripts/hf_state_backup.py` backs up state to a Storage Bucket. `warmup_models.py` was fixed to bake the MiniLM reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) instead of BGE-reranker-v2-m3. Dockerfile expects `README.md` (staging script copies `README_HF.md` as `README.md`).
11. **Oracle VM**: 2 OCPU ARM, 12 GB RAM, Always Free. `$1 temp hold on card drops off; stays free if limits observed. No-card fallback (Cloudflare Tunnel) rejected — PC must stay online.
12. **OpenRouter key exposure**: `sk-or-v1-...` was exposed in chat history. Rotate if using the same key beyond this session.
13. **`data/uploads_seed/`**: needed by HF Dockerfile `COPY data/uploads_seed/ ./data/uploads_seed/`. Contains `.gitkeep` so the directory is tracked in git.
13. **`data/uploads/`**: in `.gitignore` (runtime artifacts only). User uploads persist via Docker volumes; not in git.
14. **Health/ready checks**: `GET /health` (liveness), `GET /ready` (postgres + qdrant + keyword + redis + celery). `/metrics` has per-stage timing + `llm_usage` totals + `counters`.
15. **Celery fallback**: If worker is down, `/ready` reports `celery: unhealthy` (informational only). API still accepts queries; ingestion runs synchronously (blocks until complete).

## Long-running processes

Never host static files with `python -m http.server` inside a bash tool call —
the process is reaped when the call returns. For static HTML, open directly with
`Start-Process <file>`. For real servers, launch detached:
`Start-Process pwsh -ArgumentList '-NoExit','-Command','<cmd>'`.
Never run `Get-Process <name> | Stop-Process` — it kills unrelated processes.

## Research tools

When you need library/framework docs, use `context7` tools. When unsure how to
implement something, use `gh_grep` to search real GitHub code first.

## Commands cheat sheet (exact invocations)

```bash
# Dev start (one command)
make start          # starts Docker infra + API

# Dev separately
make run            # API only (infra must be up)
make worker         # Celery worker (background ingestion)

# Init & ingest
make init-db        # PostgreSQL schema + tsvector index
python scripts/migrate_tsvector.py  # backfill keyword postings (idempotent)
make corpus         # generate + embed corpus

# Lint / typecheck / test
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
make test           # full suite (unit + integration if Docker up)
make test-unit      # unit only (no external services)
make test-integration  # integration (requires Docker)
make eval           # evaluation pipeline (MockLLM by default)
make benchmark      # generate benchmark report

# Prod
make prod-build     # build production images
make prod-up        # start prod stack (volumes preserved)
make prod-down      # stop WITHOUT -v (preserves volumes!)
make prod-restart   # restart api + worker
make prod-health    # curl /ready
make prod-logs      # tail API logs
make prod-logs-worker # tail worker logs

# HF Spaces staging / deploy
python scripts/hf_space_stage.py --output ./hf_staging    # assemble locally
python scripts/hf_space_stage.py --output ./hf_staging --push username/space-name  # push to Space
python scripts/hf_state_backup.py --bucket username/askmydocs-data  # back up state

# Oracle VM (from DEPLOY.md)
sudo apt-get install docker-ce docker-compose-plugin  # once
docker compose -f docker-compose.prod.yml up -d postgres qdrant redis  # first boot
python3 scripts/init_db.py
python3 scripts/migrate_tsvector.py
make prod-build && make prod-up
```

## Repo structure boundaries

- `app/` — core Python package (FastAPI, retrieval, ingestion, generation, grounding, evaluation, DB)
- `scripts/` — one-shot scripts (ingest, init DB, corpus gen, state backup, HF staging)
- `data/` — corpus (46 files: docx/html/pdf/markdown), uploads (runtime), uploads_seed (HF scaffold)
- `frontend/` — Next.js app (pages/api/query/*, /upload, /stream), proxies backend via `NEXT_PUBLIC_API_URL`
- `deploy/huggingface/` — HF Space scaffold: Dockerfile, start.sh, warmup_models.py (now MiniLM reranker), README.md (operator guide)
- `tests/` — unit + integration + evaluation tests
- `evals/` — golden dataset, baselines, reports, thresholds
- `.github/workflows/` — CI (lint, typecheck, unit, integration, eval)

## Framework/toolchain quirks

- **Pydantic v2 + pydantic-settings**: all config through `Settings` class in `app/config/settings.py`. `get_settings()` is cached with `lru_cache`. `.env` loading is case-insensitive; `extra="ignore"` ignores unknown vars.
- **SQLAlchemy 2 + async**: `app/db/session.py` has `init_db()`, `session_scope()`. No async ORM in production — sync only.
- **Keyword search is tsvector**: Postgres `tsvector` + GIN trigger. Interface is identical (`BM25Indexer` class).
- **Celery + Redis**: broker on DB 0, results on DB 1. `--concurrency 1` in production (each child loads its own model copy). Fallback to sync ingestion if worker unavailable.
- **Sentence-Transformers**: `BAAI/bge-m3` embeddings (1024 dim). `CrossEncoder` reranker. `Embedder(settings=settings).embed_queries(["warmup"])` warms the model during startup.
- **OpenRouter LLM**: `nex-agi/nex-n2.5-mini:free` is default. OpenAI SDK pointed at `https://openrouter.ai/api/v1`. JSON mode (`llm_json_mode=true`) for structured answers.
- **Langfuse**: optional distributed tracing. Zero-op when creds absent (`langfuse_enabled` property).
- **Ragas evaluation**: isolated behind `app/evaluation/ragas_eval.py`. `MockLLM` for deterministic key-free tests. Baselines in `evals/baselines/`.

## What to verify before finishing a session

1. `make lint && make typecheck` — zero errors
2. `make test-unit` — all green (no API keys needed)
3. If Docker is up: `make init-db` + integration tests green
4. Local LLM config: `/models` → select `qwen2.5-coder:7b`; confirm one small edit fires a tool call
5. If working on HF deployment: `docker build` of staged dir succeeds; `start.sh` reaches port 7860
6. OpenRouter key not exposed in chat history (rotate if needed)
7. Disk free > 10 GB (clear HF cache / Docker prune if needed)