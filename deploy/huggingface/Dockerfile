# AskMyDocs — all-in-one Hugging Face Docker Space.
#
# Everything the app needs runs inside this single container:
#   PostgreSQL 16 + Qdrant + Redis + FastAPI (uvicorn) + Celery worker
#   + a Caddy reverse proxy in front of the API.
#
# This file is copied to the root of the Space repository by
# scripts/hf_space_stage.py, so all COPY paths are relative to the repo root.
# The Space README.md must set `app_port: 7860`. Caddy listens on 7860 and
# proxies to uvicorn on 8000. This mirrors the production topology in
# docker-compose.prod.yml without Docker-in-Docker.
#
# HF Spaces run the container as UID 1000, so every step that writes owned
# files happens as that user, and no bind-mount tricks are used.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Sentence-transformers / HF Hub weight cache. Warmed at build time.
    HF_HOME=/home/user/hf_home \
    HF_HUB_OFFLINE=0 \
    # Embedded service paths (internal only; only 7860 is exposed).
    PGDATA=/home/user/pgdata \
    QDRANT__STORAGE__STORAGE_PATH=/home/user/qdrant_storage \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ---------------------------------------------------------------------------
# System packages: PostgreSQL 16, Redis, Caddy, and Qdrant runtime deps.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        postgresql \
        redis-server \
        caddy \
        curl \
        ca-certificates \
        build-essential \
        libgomp1 \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Qdrant ships self-contained musl binaries (no glibc dependency), which is
# far lighter than installing its .deb (which drags in all of Caddy).
ARG QDRANT_VERSION=v1.13.0
RUN curl -fsSL \
        "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz" \
        -o /tmp/qdrant.tgz \
    && mkdir -p /opt/qdrant \
    && tar -xzf /tmp/qdrant.tgz -C /opt/qdrant \
    && rm /tmp/qdrant.tgz

# ---------------------------------------------------------------------------
# Python dependencies (same set as the production Dockerfile).
# ---------------------------------------------------------------------------
WORKDIR /home/user/app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Application code + runtime scripts.
COPY app ./app
COPY scripts ./scripts

# Space entrypoint helpers.
COPY start.sh ./start.sh
COPY warmup_models.py ./warmup_models.py
RUN chmod +x ./start.sh

# Corpus: baked into the image so the Space is useful on a cold start (HF
# container disks are wiped on every restart). data/corpus is always tracked
# in git; data/uploads_seed is optional (see deploy/huggingface/README.md).
COPY data/corpus ./data/corpus
COPY data/uploads_seed/ ./data/uploads_seed/

# ---------------------------------------------------------------------------
# Build-time warm-up: bake BGE-M3 + the cross-encoder weights into the image.
# Both repos are ungated (no license acceptance needed). HF_TOKEN is an optional
# *build secret* for rate-limit relief; a failure here still leaves the image
# usable (the model then downloads on first startup instead).
# ---------------------------------------------------------------------------
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" \
    python warmup_models.py || echo "[warn] model warm-up skipped"

# ---------------------------------------------------------------------------
# Runtime user (HF Spaces use UID 1000) and writable directories.
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/pgdata /home/user/pgsocket /home/user/qdrant_storage /home/user/app/data/uploads \
    && chown -R user:user /home/user

USER user

EXPOSE 7860

# start.sh starts Postgres + Redis + Qdrant, initializes the schema, ingests
# the corpus on first boot, then supervises uvicorn/Celery/Caddy.
CMD ["./start.sh"]
