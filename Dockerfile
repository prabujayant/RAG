# AskMyDocs production image (multi-arch: amd64 + arm64).
# Single image serves both API and Celery worker (different commands).
# Local dev is unaffected: keep using `make run` / `.\start.ps1` there.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Persist the 2GB embedding/reranker weights on a volume, not in the
    # container layer, so redeploys never re-download the models.
    HF_HOME=/models \
    HF_HUB_OFFLINE=0

WORKDIR /srv/app

# System deps: build tools for any sdist fallback (torch pulls wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching (prod deps only, no dev tools).
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

# Application code + runtime scripts (corpus/data mount as volumes).
COPY app ./app
COPY scripts ./scripts

# Writable dirs (overridden by volumes in compose).
RUN mkdir -p /srv/app/data/uploads /models \
    && useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /srv/app /models
USER appuser

EXPOSE 8000

# No --reload in production. Overridden by the worker service in compose.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
