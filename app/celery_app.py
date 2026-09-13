"""
Celery application singleton.

All worker processes import this module to get the configured Celery app.
Tasks are defined in ``app.tasks``.
"""

from __future__ import annotations

import logging
import time

from celery import Celery
from celery.signals import worker_ready

from app.config import get_settings

logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """Build and configure the Celery application."""
    settings = get_settings()

    app = Celery(
        "askmydocs",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=[
            "app.tasks.ingestion",
            "app.tasks.evaluation",
            "app.tasks.benchmark",
        ],
    )

    # ---- Broker -----------------------------------------------------------
    app.conf.broker_connection_retry_on_startup = True

    # ---- Task defaults ----------------------------------------------------
    app.conf.task_track_started = settings.celery_task_track_started
    app.conf.task_ignore_result = settings.celery_task_ignore_result
    app.conf.task_time_limit = settings.celery_task_time_limit
    app.conf.task_soft_time_limit = settings.celery_task_soft_time_limit
    app.conf.worker_prefetch_multiplier = settings.celery_worker_prefetch_multiplier

    # ---- Serialisation ----------------------------------------------------
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]
    app.conf.timezone = "UTC"

    # ---- Routes -----------------------------------------------------------
    app.conf.task_routes = {
        "app.tasks.ingestion.*": {"queue": "ingestion"},
        "app.tasks.evaluation.*": {"queue": "evaluation"},
        "app.tasks.benchmark.*": {"queue": "benchmark"},
    }

    # ---- Periodic tasks (Celery Beat) --------------------------------------
    app.conf.beat_schedule = {
        "nightly-benchmark": {
            "task": "app.tasks.benchmark.run_full_benchmark",
            "schedule": 86400.0,  # daily
            "options": {"queue": "benchmark"},
        },
    }

    return app


celery_app = create_celery_app()


@worker_ready.connect
def _warm_embedding_model(sender, **kwargs) -> None:
    """Pre-load the embedding model when the worker boots.

    Without this, the first background upload pays the full BGE-M3 cold
    load (~2 min on CPU) on the user's clock — a 26 KB / 3-chunk PDF was
    observed taking 125s end to end. Warming here moves that one-off cost
    to worker startup, where it belongs (mirrors the API's lifespan
    warmup in app.main). Failures are non-fatal: the model loads lazily
    on first use instead. Gated by WARMUP_MODELS like the API path.
    """
    settings = get_settings()
    if not settings.warmup_models:
        logger.info("Worker model warm-up disabled (WARMUP_MODELS=false)")
        return
    try:
        from app.embeddings.embedder import Embedder

        start = time.perf_counter()
        Embedder(settings=settings).embed_queries(["warmup"])
        logger.info(
            "Worker embedding model warmed up in %.1fs",
            time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Worker model warm-up failed; first task will load lazily: %s",
            exc,
        )
