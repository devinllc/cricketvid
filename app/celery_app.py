"""Celery application configuration for async video processing workers."""
import os

from celery import Celery


def _broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"


def _result_backend() -> str:
    return os.getenv("CELERY_RESULT_BACKEND") or os.getenv("REDIS_URL") or "redis://localhost:6379/1"


celery_app = Celery(
    "cricket_video",
    broker=_broker_url(),
    backend=_result_backend(),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_app.autodiscover_tasks(["app.workers"])
