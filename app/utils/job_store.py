"""
Thread-safe in-memory job store for tracking video processing jobs.
Uses a dictionary protected by a threading.Lock.
"""
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

# Job status constants
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def create_job(drill_type: str, filename: str) -> str:
    """Create a new job and return its job_id."""
    job_id = str(uuid.uuid4())
    with _lock:
        _store[job_id] = {
            "job_id": job_id,
            "drill_type": drill_type,
            "filename": filename,
            "status": STATUS_QUEUED,
            "report": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve job data by job_id."""
    with _lock:
        return _store.get(job_id)


def update_status(job_id: str, status: str) -> None:
    """Update only the status field of a job."""
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = status


def set_report(job_id: str, report: Dict[str, Any]) -> None:
    """Attach the completed report to a job and mark it complete."""
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = STATUS_COMPLETE
            _store[job_id]["report"] = report
            _store[job_id]["completed_at"] = datetime.utcnow().isoformat()


def set_error(job_id: str, error: str) -> None:
    """Mark a job as failed with an error message."""
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = STATUS_FAILED
            _store[job_id]["error"] = error
            _store[job_id]["completed_at"] = datetime.utcnow().isoformat()


def list_jobs() -> list:
    """Return a list of all job summaries."""
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
