"""Job store with Redis persistence and in-memory fallback.

If REDIS_URL is configured and reachable, jobs are stored durably in Redis.
Otherwise, a thread-safe in-memory store is used (development fallback).
"""
import threading
import uuid
import json
import os
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    import redis
except Exception:  # pragma: no cover - safe fallback when redis is unavailable
    redis = None

# Job status constants
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_processing_lock = threading.Lock()
_PROCESSING_LOCK_KEY = "cricket:processing-lock"
_PROCESSING_LOCK_TTL_SECONDS = int(os.getenv("PROCESSING_LOCK_TTL_SECONDS", "14400"))


def _redis_client():
    if redis is None:
        return None
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        client = redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _job_to_hash(job: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in job.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v)
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def _hash_to_job(data: Dict[str, str]) -> Dict[str, Any]:
    job: Dict[str, Any] = {}
    for k, v in data.items():
        if k in {"report"} and v:
            try:
                job[k] = json.loads(v)
                continue
            except Exception:
                pass
        job[k] = None if v == "" else v
    return job


def create_job(drill_type: str, filename: str) -> str:
    """Create a new job and return its job_id."""
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "drill_type": drill_type,
        "filename": filename,
        "status": STATUS_QUEUED,
        "report": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    client = _redis_client()
    if client is not None:
        client.hset(_job_key(job_id), mapping=_job_to_hash(job))
        client.expire(_job_key(job_id), 60 * 60 * 24 * 14)
        return job_id

    with _lock:
        _store[job_id] = job
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve job data by job_id."""
    client = _redis_client()
    if client is not None:
        data = client.hgetall(_job_key(job_id))
        if not data:
            return None
        return _hash_to_job(data)

    with _lock:
        return _store.get(job_id)


def update_status(job_id: str, status: str) -> None:
    """Update only the status field of a job."""
    client = _redis_client()
    if client is not None:
        if client.exists(_job_key(job_id)):
            client.hset(_job_key(job_id), mapping={"status": status})
        return

    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = status


def set_report(job_id: str, report: Dict[str, Any]) -> None:
    """Attach the completed report to a job and mark it complete."""
    client = _redis_client()
    if client is not None:
        if client.exists(_job_key(job_id)):
            client.hset(
                _job_key(job_id),
                mapping={
                    "status": STATUS_COMPLETE,
                    "report": json.dumps(report),
                    "completed_at": datetime.utcnow().isoformat(),
                },
            )
        return

    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = STATUS_COMPLETE
            _store[job_id]["report"] = report
            _store[job_id]["completed_at"] = datetime.utcnow().isoformat()


def set_error(job_id: str, error: str) -> None:
    """Mark a job as failed with an error message."""
    client = _redis_client()
    if client is not None:
        if client.exists(_job_key(job_id)):
            client.hset(
                _job_key(job_id),
                mapping={
                    "status": STATUS_FAILED,
                    "error": error,
                    "completed_at": datetime.utcnow().isoformat(),
                },
            )
        return

    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = STATUS_FAILED
            _store[job_id]["error"] = error
            _store[job_id]["completed_at"] = datetime.utcnow().isoformat()


def list_jobs() -> list:
    """Return a list of all job summaries."""
    client = _redis_client()
    if client is not None:
        jobs = []
        for key in client.scan_iter(match="job:*"):
            d = client.hgetall(key)
            if not d:
                continue
            j = _hash_to_job(d)
            jobs.append(
                {
                    "job_id": j.get("job_id"),
                    "drill_type": j.get("drill_type"),
                    "status": j.get("status"),
                    "created_at": j.get("created_at"),
                }
            )
        jobs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return jobs

    with _lock:
        return [
            {
                "job_id": v["job_id"],
                "drill_type": v["drill_type"],
                "status": v["status"],
                "created_at": v["created_at"],
            }
            for v in _store.values()
        ]


def _acquire_redis_processing_lock(client, job_id: str, timeout: Optional[int] = None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if client.set(_PROCESSING_LOCK_KEY, job_id, nx=True, ex=_PROCESSING_LOCK_TTL_SECONDS):
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the processing slot")
        time.sleep(1)


def _release_redis_processing_lock(client, job_id: str) -> None:
    try:
        current_holder = client.get(_PROCESSING_LOCK_KEY)
        if current_holder == job_id:
            client.delete(_PROCESSING_LOCK_KEY)
    except Exception:
        pass


@contextmanager
def processing_slot(job_id: str, timeout: Optional[int] = None):
    """Serialize all analysis jobs so only one runs at a time."""
    client = _redis_client()
    if client is not None:
        _acquire_redis_processing_lock(client, job_id, timeout=timeout)
        try:
            yield
        finally:
            _release_redis_processing_lock(client, job_id)
        return

    _processing_lock.acquire()
    try:
        yield
    finally:
        _processing_lock.release()
