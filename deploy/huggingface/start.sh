#!/usr/bin/env bash
# AskMyDocs — Hugging Face Docker Space entrypoint.
#
# Starts the embedded services (PostgreSQL, Redis, Qdrant), waits for them,
# initializes the schema, ingests the baked corpus on first boot, performs an
# optional restore from a Storage Bucket, then supervises FastAPI, the Celery
# worker, and Caddy. The Space is served on 7860 (Caddy -> uvicorn:8000).
#
# Environment (all optional except OPENROUTER_API_KEY for real answers):
#   OPENROUTER_API_KEY   LLM key. Without it the Space still runs; queries fail.
#   HF_BUCKET            e.g. "username/askmydocs-data" — restore state at boot.
#   HF_TOKEN             token used to read a private HF_BUCKET (Space secret).
#   RESTORE_ON_BOOT      "true" to restore {bucket}/state.tar.zst before start
#   INGEST_CORPUS        "true" (default) to ingest data/corpus when empty
#   RUN_WORKER           "true" (default) to run a Celery worker
#   APP_PORT             public port (default 7860)

set -euo pipefail

log() { printf '\033[1;36m[hf-space]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[hf-space]\033[0m %s\n' "$*" >&2; }

APP_PORT="${APP_PORT:-7860}"
INGEST_CORPUS="${INGEST_CORPUS:-true}"
RUN_WORKER="${RUN_WORKER:-true}"
RUN_FRONTEND="${RUN_FRONTEND:-true}"
RESTORE_ON_BOOT="${RESTORE_ON_BOOT:-false}"
DATA_ROOT="/home/user"

export PGDATA="${PGDATA:-${DATA_ROOT}/pgdata}"
# Debian's PostgreSQL defaults unix_socket_directories to /var/run/postgresql,
# which is owned by postgres:postgres. This container runs as UID 1000, so the
# postmaster cannot create its lock file there and dies with:
#   FATAL: could not create lock file "/var/run/postgresql/.s.PGSQL.5432.lock"
# (surfacing only as "pg_ctl: could not start server"). Use a writable dir.
export PG_SOCKET_DIR="${PG_SOCKET_DIR:-${DATA_ROOT}/pgsocket}"
export QDRANT__STORAGE__STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-${DATA_ROOT}/qdrant_storage}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_home}"
export PATH="/home/user/.local/bin:$PATH"

PG_BIN="$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | head -n1 || true)"
if [ -z "$PG_BIN" ]; then PG_BIN="/usr/lib/postgresql/15/bin"; fi
DB_USER="${POSTGRES_USER:-askmydocs}"
DB_NAME="${POSTGRES_DB:-askmydocs}"

export DATABASE_URL="postgresql+psycopg://${DB_USER}:${DB_USER}@127.0.0.1:5432/${DB_NAME}"
export QDRANT_URL="http://127.0.0.1:6333"
export CELERY_BROKER_URL="redis://127.0.0.1:6379/0"
export CELERY_RESULT_BACKEND="redis://127.0.0.1:6379/1"
export APP_ENV="${APP_ENV:-production}"
export CORS_ORIGINS="${CORS_ORIGINS:-*}"

PIDS=()
cleanup() {
  log "shutting down"
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT TERM INT

wait_for_port() {
  local host="$1" port="$2" name="$3" tries="${4:-60}"
  for _ in $(seq 1 "$tries"); do
    if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then return 0; fi
    sleep 1
  done
  warn "$name did not open ${host}:${port} within ${tries}s"
  return 1
}

# --------------------------------------------------------------------------
# 0. Optional restore (must run before services open their data directories).
#    The archive mirrors $DATA_ROOT: pgdata/, qdrant_storage/, app/data/uploads/
# --------------------------------------------------------------------------
if [ "$RESTORE_ON_BOOT" = "true" ] && [ -n "${HF_BUCKET:-}" ]; then
  # Bucket download URL has no revision segment:
  #   {endpoint}/buckets/{bucket_id}/resolve/{path}
  url="https://huggingface.co/buckets/${HF_BUCKET}/resolve/state.tar.zst"
  auth=()
  if [ -n "${HF_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer ${HF_TOKEN}"); fi
  log "restoring state from ${url}"
  if curl -fsSL "${auth[@]}" "$url" -o /tmp/state.tar.zst; then
    tar --zstd -xf /tmp/state.tar.zst -C "$DATA_ROOT" \
      && log "state restored" || warn "restore extraction failed"
  else
    warn "restore download failed; continuing with empty state"
  fi
  rm -f /tmp/state.tar.zst
fi

# --------------------------------------------------------------------------
# 1. PostgreSQL
# --------------------------------------------------------------------------
if [ ! -f "$PGDATA/PG_VERSION" ]; then
  log "initializing PostgreSQL data directory at $PGDATA"
  mkdir -p "$PGDATA"
  "$PG_BIN/initdb" -D "$PGDATA" -U "$DB_USER" -A trust -E UTF8 >/dev/null
fi

mkdir -p "$PG_SOCKET_DIR"

log "starting PostgreSQL"
"$PG_BIN/pg_ctl" -D "$PGDATA" \
  -o "-c listen_addresses=127.0.0.1 -c unix_socket_directories=$PG_SOCKET_DIR -c fsync=off -c synchronous_commit=off" \
  -l "$PGDATA/server.log" -w start >/dev/null

if ! "$PG_BIN/psql" -h 127.0.0.1 -U "$DB_USER" -d postgres -tAc \
      "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  log "creating database ${DB_NAME}"
  "$PG_BIN/createdb" -h 127.0.0.1 -U "$DB_USER" "$DB_NAME"
fi

# --------------------------------------------------------------------------
# 2. Redis + Qdrant
# --------------------------------------------------------------------------
log "starting Redis"
redis-server --daemonize no --dir "$DATA_ROOT" --appendonly no --save "" \
  >/dev/null 2>&1 &
PIDS+=("$!")

log "starting Qdrant"
QDRANT__SERVICE__HOST=127.0.0.1 QDRANT__SERVICE__HTTP_PORT=6333 \
  /opt/qdrant/qdrant >/dev/null 2>&1 &
PIDS+=("$!")

wait_for_port 127.0.0.1 5432 postgres 60 || true
wait_for_port 127.0.0.1 6379 redis 30 || true
wait_for_port 127.0.0.1 6333 qdrant 60 || true

# --------------------------------------------------------------------------
# 3. Schema + API + worker + Caddy (supervised)
# --------------------------------------------------------------------------
log "initializing database schema"
python scripts/init_db.py || warn "init_db reported problems; continuing"

# Servers start BEFORE corpus ingestion so port 7860 is open promptly. HF
# treats a Space that never binds its port as failed, and first-boot ingestion
# can take 5-15 minutes on free CPU. Only these are background jobs, so the
# `wait -n` supervisor below is unaffected by ingestion running afterwards.
log "starting uvicorn on 127.0.0.1:8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 &

if [ "$RUN_WORKER" = "true" ]; then
  log "starting Celery worker"
  celery -A app.celery_app.celery_app worker \
    -Q ingestion,evaluation,benchmark -l info --concurrency 1 \
    --prefetch-multiplier 1 &
fi

# Next.js UI on 3000. Its /api/* route handlers proxy to the API above
# (NEXT_PUBLIC_API_URL defaults to http://127.0.0.1:8000), so the browser only
# ever talks to this one origin — no CORS configuration needed.
FRONTEND_DIR="${DATA_ROOT}/app/frontend"
if [ "$RUN_FRONTEND" = "true" ] && [ -d "$FRONTEND_DIR/.next" ]; then
  log "starting Next.js UI on 127.0.0.1:3000"
  ( cd "$FRONTEND_DIR" && PORT=3000 ./node_modules/.bin/next start -p 3000 ) &
else
  warn "frontend not started (RUN_FRONTEND=${RUN_FRONTEND}, .next present: $([ -d "$FRONTEND_DIR/.next" ] && echo yes || echo no))"
fi

log "starting Caddy on :${APP_PORT}"
# API paths go straight to uvicorn; everything else is the Next.js UI.
# Caddyfile syntax: a block's braces must not share a line with the opener,
# so each directive gets its own line.
cat > "${DATA_ROOT}/Caddyfile.runtime" <<EOF
{
	admin off
	auto_https off
}
:${APP_PORT} {
	@api path /health* /ready* /metrics* /docs* /redoc* /openapi.json /query* /documents*
	handle @api {
		reverse_proxy 127.0.0.1:8000
	}
	handle {
		reverse_proxy 127.0.0.1:3000
	}
}
EOF
caddy run --config "${DATA_ROOT}/Caddyfile.runtime" --adapter caddyfile &

# --------------------------------------------------------------------------
# 4. First-boot corpus ingestion (servers already serving; /ready flips green
#    once this finishes and the keyword/vector stores are populated).
# --------------------------------------------------------------------------
chunk_count="$(python - <<'PY' 2>/dev/null | tail -n1 || echo 0
try:
    from app.db.session import SessionLocal
    from app.db.models import Chunk

    with SessionLocal() as s:
        print(s.query(Chunk).count())
except Exception:
    print(0)
PY
)"
chunk_count="${chunk_count:-0}"

if [ "$INGEST_CORPUS" = "true" ] && [ "$chunk_count" -eq 0 ] && [ -d "data/corpus" ]; then
  # Ingest at the LOWEST CPU priority. On free cpu-basic (2 vCPU) a full corpus
  # ingest takes ~20+ min and would otherwise starve interactive work: user
  # uploads (queued on the Celery worker) sat there for 5+ minutes with the UI
  # stuck on "Parsing, embedding and indexing...". With `nice`, uploads and
  # queries always win; the corpus still finishes, just in the background.
  log "corpus empty — ingesting data/corpus at low priority (may take 20+ min on CPU)"
  nice -n 19 python scripts/ingest_corpus.py --root data/corpus \
    || warn "corpus ingest failed"
else
  log "skipping corpus ingest (chunks=$chunk_count)"
fi

log "AskMyDocs is up on port ${APP_PORT}"
wait -n
