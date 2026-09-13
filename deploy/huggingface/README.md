# AskMyDocs — Hugging Face Docker Space Deployment

## Overview

Deploy AskMyDocs as an all-in-one Docker Space on Hugging Face. The container runs:
- PostgreSQL 16 (embedded)
- Qdrant (embedded binary)
- Redis (embedded)
- FastAPI + uvicorn
- Celery worker
- Caddy reverse proxy (port 7860 → 8000)

## Prerequisites

> ### ⚠️ Hugging Face PRO is required for Docker Spaces
>
> Docker and Gradio Spaces are **no longer free**. Creating one on `cpu-basic`
> returns `402 Payment Required`:
>
> ```
> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
> on free cpu-basic requires a PRO subscription.
> ```
>
> A PRO plan ($9/month) is required, confirmed on
> [huggingface.co/pricing](https://huggingface.co/pricing) under
> "Host ZeroGPU, Gradio & Docker Spaces". Static Spaces remain free, but this
> app cannot run as one — it needs Python, PostgreSQL and Qdrant.
>
> **Free alternative:** see [`DEPLOY.md`](../../DEPLOY.md) for the
> $0/month path (Oracle Cloud Always Free ARM VM + Vercel Hobby).

1. Hugging Face account **on a PRO plan** (see warning above) with a Docker
   Space created
2. Accepted licenses for gated models:
   - [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (accept on the model page)
3. OpenRouter API key ([openrouter.ai](https://openrouter.ai))

## Space Secrets

Set these in your Space **Settings → Secrets**:

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | LLM key for answer generation |
| `HF_TOKEN` | No | Enables build-time model caching |
| `HF_BUCKET` | No | Storage bucket for persistent state |
| `RESTORE_ON_BOOT` | No | Set `true` to restore from bucket |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`) |
| `LLM_REASONING_EFFORT` | No | Reasoning effort `high`/`medium`/`low`/`none` (default `high`) |

## Persistence

HF Spaces have **ephemeral disk** — data is wiped on every restart. To persist:

1. Create a Storage Bucket (requires a recent `huggingface_hub`):
   ```bash
   hf buckets create username/askmydocs-data --private
   ```
2. Set `HF_BUCKET` secret to your bucket id (e.g. `username/askmydocs-data`)
3. Set `RESTORE_ON_BOOT=true` to restore state at boot
4. Run `python scripts/hf_state_backup.py --bucket username/askmydocs-data` to
   back up current state to the bucket (`start.sh` downloads
   `https://huggingface.co/buckets/<bucket_id>/resolve/state.tar.zst`)

## Ports

| Port | Service | Notes |
|------|---------|-------|
| 7860 | Caddy (public) | HF Spaces exposes this port only |
| 8000 | uvicorn (internal) | FastAPI app |
| 5432 | PostgreSQL (internal) | Not exposed externally |
| 6333 | Qdrant (internal) | Not exposed externally |
| 6379 | Redis (internal) | Not exposed externally |

## Environment Variables

All are optional at runtime (defaults match `start.sh`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | LLM key (without it, queries fail) |
| `APP_PORT` | `7860` | Public port (Caddy) |
| `INGEST_CORPUS` | `true` | Ingest `data/corpus` on first boot |
| `RUN_WORKER` | `true` | Start Celery worker |
| `RESTORE_ON_BOOT` | `false` | Restore state from bucket |
| `HF_BUCKET` | — | Storage bucket name |
| `HF_TOKEN` | — | Token for private bucket access |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins (credentials auto-disabled for `*`; set explicitly in prod) |
| `LLM_REASONING_EFFORT` | `high` | Reasoning effort; lower is faster (see root README) |
| `ENABLE_RERANKER` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker model |

## Staging & Deploy

Use the staging script to assemble a Space-ready directory:

```bash
# Stage locally
python scripts/hf_space_stage.py --output ./hf_staging

# Build locally (for testing)
cd hf_staging && docker build -t askmydocs-hf .

# Push to Space (requires huggingface_hub)
python scripts/hf_space_stage.py --push username/space-name
```

## State Backup

Back up the current state (Postgres, Qdrant, uploads) to a Storage Bucket:

```bash
python scripts/hf_state_backup.py --bucket username/askmydocs-data
```

## Troubleshooting

**Build fails on model download:**
- Accept the BAAI model licenses on Hugging Face Hub
- Set `HF_TOKEN` secret with a token that has read access

**Cold start is slow:**
- First boot ingests the corpus (~5-10 min on CPU)
- Set `HF_BUCKET` + `RESTORE_ON_BOOT=true` to skip re-ingestion

**Queries fail with "LLM error":**
- Ensure `OPENROUTER_API_KEY` is set correctly
- Check Space logs for connection errors

**Memory pressure on free tier:**
- Set `ENABLE_RERANKER=false` to disable reranking (~1GB savings)
- Reduce `CHUNK_SIZE` to 256 to lower memory usage
