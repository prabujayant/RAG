# start.ps1 — one-command startup for AskMyDocs.
# Starts infrastructure (PostgreSQL, Qdrant, Redis) then the FastAPI dev server.
#
# Usage:
#   .\start.ps1              # start infra + API
#   .\start.ps1 -NoDocker    # skip infra (already running)
#   .\start.ps1 -Init        # also initialize the DB schema (first run only)
#   .\start.ps1 -Worker      # also start the Celery worker (background ingest)

[CmdletBinding()]
param(
    [switch]$NoDocker,
    [switch]$Init,
    [switch]$Worker
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

# Resolve the venv interpreter once. Console-script launchers in .venv\Scripts
# (uvicorn.exe, pytest.exe, ...) embed an absolute interpreter path and break if
# the environment is ever moved, so this script always calls `python -m <mod>`.
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

function Write-Step($message) {
    Write-Host "`n==> $message" -ForegroundColor Cyan
}

# 1. Ensure the virtual environment is active.
if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $root '.venv\Scripts\Activate.ps1'
    if (Test-Path $activate) {
        Write-Step 'Activating virtual environment'
        . $activate
    } else {
        Write-Warning 'No .venv found. Run: python -m venv .venv; .venv\Scripts\pip install -e ".[dev]"'
    }
}

# 2. Start infrastructure containers.
if (-not $NoDocker) {
    Write-Step 'Starting infrastructure (docker compose up -d)'
    docker compose up -d
} else {
    Write-Host 'Skipping Docker (infra assumed running).' -ForegroundColor DarkGray
}

# 3. Optionally initialize the database schema (first run only).
if ($Init) {
    Write-Step 'Initializing database schema'
    & $python scripts/init_db.py
}

# 4. Optionally start the Celery worker (background ingest + evals).
#    Requires Redis from `docker compose up -d`. Without a worker,
#    ?background=true calls fall back to synchronous ingestion.
#    NOTE: `-A app.celery_app.celery_app` is the module *attribute*, and
#    --pool=solo is required on Windows (the default prefork pool is
#    unsupported there).
if ($Worker) {
    Write-Step 'Starting Celery worker (ingestion,evaluation,benchmark)'
    Start-Process -NoNewWindow $python -ArgumentList @(
        '-m', 'celery', '-A', 'app.celery_app.celery_app', 'worker',
        '-Q', 'ingestion,evaluation,benchmark', '-l', 'info', '--pool=solo'
    ) -WorkingDirectory $root
}

# 5. Start the FastAPI dev server.
#    Watch only the source directories: the project root also holds .venv
#    (~48k files) and frontend/node_modules (~29k files), and watching those
#    makes reloads very slow.
Write-Step 'Starting API at http://127.0.0.1:8000 (docs at /docs)'
& $python -m uvicorn app.main:app --reload --reload-dir app --reload-dir scripts --port 8000
