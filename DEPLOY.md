# AskMyDocs — Free Production Deployment

Target: **$0/month** — Oracle Cloud Always Free ARM VM (backend + infra) +
Vercel Hobby (frontend). Local dev keeps working unchanged
(`docker-compose.yml` + `make run`).

Live shape when done:

- Frontend: `https://<your-app>.vercel.app`
- Backend: `https://<API_DOMAIN>` (e.g. `https://api.example.com`)

---

## 1. Oracle VM creation

1. Create an Oracle Cloud account (card verification required; stay within
   Always Free limits to avoid charges).
2. Launch an instance with:
   - Image: **Ubuntu 24.04**
   - Shape: **Ampere ARM**, **2 OCPU, 12 GB RAM**
   - Public IP: yes (note it as `<VM_IP>`)
   - SSH key: add yours, keep the private key safe
3. DNS: point an `A` record for your API domain at `<VM_IP>`
   (e.g. `api.example.com`). Caddy will issue HTTPS automatically.
   No-domain test path: use `<VM_IP>.nip.io` as the domain.
4. Security list / firewall — allow **only**:
   - `22/tcp` (SSH, ideally your IP only)
   - `80/tcp`, `443/tcp` (Caddy)
    - Nothing else. PostgreSQL/Qdrant/Redis are **not** published
      by `docker-compose.prod.yml` (no `ports:` on those services).

> Idle reclamation: pure-free Oracle accounts may reclaim idle VMs. Keep a
> trivial uptime ping on `/health`, or upgrade to pay-as-you-go while staying
> inside always-free usage (reclamation stops).

## 2. Docker installation (on the VM)

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER   # re-login after this
docker compose version
```

## 3. Repository setup (on the VM)

```bash
git clone <your-repo-url> RAG && cd RAG
cp .env.example .env
```

## 4. `.env` setup (on the VM — never commit this file)

Required values:

```bash
OPENROUTER_API_KEY=<key from https://openrouter.ai/keys>
API_DOMAIN=api.example.com            # your DNS name (or <VM_IP>.nip.io)
CORS_ORIGINS=https://<your-app>.vercel.app
APP_ENV=production
POSTGRES_PASSWORD=<long random string>  # only for FRESH volumes (see note)
```

> Set `CORS_ORIGINS` explicitly (never `*` in prod): the app auto-disables
> `allow_credentials` for wildcard origins because browsers reject the
> `*` + credentials combination, and credentialed cross-origin requests
> would otherwise fail silently.

Notes:

- `DATABASE_URL`, `QDRANT_URL`, `CELERY_*` are overridden
  with Docker service names by `docker-compose.prod.yml`; leave the localhost
  values in `.env` alone.
- `POSTGRES_PASSWORD` is read **only on first volume init**. Changing it later
  without wiping `postgres_data` breaks login — pick it once.
- Confirm `.env` is ignored: `git check-ignore .env` (must print `.env`).

## 5. Database initialization

```bash
docker compose -f docker-compose.prod.yml up -d postgres qdrant redis
python3 scripts/init_db.py   # or: docker compose -f docker-compose.prod.yml run --rm api python scripts/init_db.py
python3 scripts/migrate_tsvector.py  # backfill keyword postings for old DBs (idempotent)
```

## 6. Docker startup (full stack)

```bash
make prod-build
make prod-up
docker compose -f docker-compose.prod.yml ps
```

What starts: `postgres` (metadata + tsvector keyword search), `qdrant`,
`redis`, `api` (uvicorn, 1 worker, **no `--reload`**), `worker`
(Celery `--concurrency 1`), `caddy` (80/443 → api:8000).
Model weights (~2 GB) download once into the `models_cache` volume on first
boot; redeploys reuse them.

## 7. Celery startup

The worker is part of the stack (`make prod-up` starts it). Verify:

```bash
docker compose -f docker-compose.prod.yml logs --tail 20 worker  # expect "celery@... ready"
docker compose -f docker-compose.prod.yml exec redis redis-cli ping  # PONG
```

If no worker replies, uploads still work: the API falls back to synchronous
ingestion and `/ready` reports `celery: unhealthy` (informational only).

## 8. Corpus ingestion

The image ships only `app/` and `scripts/` — **not** `data/`. Instead,
`./data/corpus` is bind-mounted **read-only** into both `api` and `worker`
(see `docker-compose.prod.yml`). The corpus is committed to the repo, so a
fresh `git clone` already has it on the host.

```bash
# Option A — ingest the bundled corpus (reads data/corpus):
docker compose -f docker-compose.prod.yml exec api \
  python scripts/ingest_corpus.py --skip-existing
# Option B — upload through the UI (background ingest, recommended).
```

> `generate_corpus.py` (alias `make corpus`) **writes** corpus files, so it must
> run on the **host**, never in the container — the mount is read-only. The
> corpus is snapshotted at ingest time; re-run `ingest_corpus.py` after adding
> documents to `./data/corpus` (it is idempotent with `--skip-existing`).

## 9. Domain and HTTPS setup

1. `API_DOMAIN` set in `.env`, DNS `A` record → `<VM_IP>`.
2. `docker compose -f docker-compose.prod.yml up -d caddy` (already up via
   `make prod-up`; Caddy fetches the cert on first request).
3. Test: `curl https://<API_DOMAIN>/health` → `{"status":"healthy",...}`.
4. No-domain test: `API_DOMAIN=<VM_IP>.nip.io` works the same way.

## 10. Vercel deployment (frontend)

1. Import the repo in Vercel, root directory `frontend`.
2. Environment variable (exact name):
   - `NEXT_PUBLIC_API_URL=https://<API_DOMAIN>`
3. Deploy. Frontend proxy routes (`/api/query`, `/api/query/stream`,
   `/api/documents/upload`, `/api/documents/[documentId]`) forward to it —
   no other config needed. No localhost URLs remain in production behavior
   (the `http://127.0.0.1:8000` fallback only applies when the variable is
   unset, i.e. local `npm run dev`).

## 11. Operations

```bash
make prod-up            # start (volumes preserved)
make prod-down           # stop WITHOUT deleting volumes (uses `stop`, not `down -v`)
make prod-restart        # restart api + worker
make prod-health         # curl /ready
make prod-logs           # tail API logs
make prod-logs-worker    # tail worker logs

# Health + dependency detail (postgres, qdrant, keyword, redis, celery):
curl -s https://<API_DOMAIN>/ready | python3 -m json.tool
# Stage timings (find the slow stage before tuning):
curl -s https://<API_DOMAIN>/metrics | python3 -m json.tool

# Redis + broker checks:
docker compose -f docker-compose.prod.yml exec redis redis-cli ping
docker compose -f docker-compose.prod.yml exec worker \
  python -m celery -A app.celery_app.celery_app inspect ping
# Result-backend write/read:
docker compose -f docker-compose.prod.yml exec redis \
  redis-cli -n 1 set probe ok

# Real end-to-end checks:
curl -s -X POST https://<API_DOMAIN>/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I rotate API keys?"}' | head -c 500
curl -s -X POST 'https://<API_DOMAIN>/documents/upload?background=true' \
  -F 'file=@/path/to/doc.md'
```

## 12. Backup and rollback

```bash
# Backup (volumes hold postgres/qdrant/redis/uploads/models):
docker compose -f docker-compose.prod.yml stop
docker run --rm -v rag_postgres_data:/v -v "$PWD/backups":/b alpine \
  tar czf /b/postgres-$(date +%F).tgz -C /v .
# repeat for qdrant_data, uploads_data
make prod-up

# Rollback a bad deploy (images are local; previous code = previous commit):
git log --oneline -5
git stash list  # never deploy with a dirty tree if you can avoid it
git checkout <previous-sha> -- <paths>  # or full: git revert
make prod-build && make prod-restart
```

Never run `docker compose down -v` on the server — `-v` deletes every
volume (database, indexes, uploads, model cache).

## 13. Resources (12 GB ARM VM)

Measured budget: API ≈ 2.5 GB (BGE-M3 + reranker resident),
worker ≈ 2.5 GB, leaving headroom for Postgres/Qdrant/Redis/Caddy.
(Keyword search lives in Postgres now — the old 512 MB OpenSearch heap
is gone with the container.)

- Keep Celery `--concurrency 1`: each child loads its own 2 GB model copy.
- BGE-M3 on ARM CPU is slow (100 pages measured at 10+ min on x86 laptop
  CPU; ARM is slower). This is expected, not a bug.
- If memory is insufficient, fall back to a smaller embedding model:
  `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`,
  `EMBEDDING_DIM=384`, then **wipe and re-ingest everything**
  (vectors are incompatible across models):
  `docker compose -f docker-compose.prod.yml down -v` (destroys data —
  backup first), `make prod-up`, re-run init + ingestion.

## 14. Security risks still open (do before trusting this)

- No authentication on `/documents/*` or `/query` — anyone with the URL can
  ingest and query. Add an API-key header check before announcing the URL.
- No rate limiting on `/query/stream` or upload polling — add a reverse-proxy
  limit (Caddy `rate_limit`) or in-app throttle.
- Keyword postings live in Postgres behind the app (LAN-only by compose,
  no separate search service to secure).
- `POSTGRES_PASSWORD`/secrets live in plaintext `.env` on the VM — restrict
  file perms (`chmod 600 .env`) and SSH access.
