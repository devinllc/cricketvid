"""
Video upload routes: POST /upload-video
"""
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.ai.cricket_scorer import SUPPORTED_DRILLS
from app.celery_app import celery_app
from app.services.video_processor import process_video
from app.utils import job_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500


@router.post("/upload-video", summary="Upload a cricket video for analysis")
async def upload_video(
    file: UploadFile = File(..., description="Cricket video file (.mp4 .mov .avi)"),
    drill_type: str = Form(..., description="Drill type to analyze"),
):
    """
    Upload a cricket video and start background analysis.

    Returns job_id and initial processing status.
    """
    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate drill type
    drill_key = drill_type.lower().replace(" ", "_")
    if drill_key not in SUPPORTED_DRILLS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported drill_type '{drill_type}'. Supported: {SUPPORTED_DRILLS}",
        )

    # Create job
    job_id = job_store.create_job(drill_key, file.filename or "unknown")
    logger.info(f"New job created: {job_id} | drill={drill_key} | file={file.filename}")

    # Save uploaded file
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"
    try:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            job_store.set_error(job_id, f"File too large: {size_mb:.1f} MB")
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB",
            )

        with open(save_path, "wb") as f:
            f.write(content)

        logger.info(f"[{job_id}] Saved {size_mb:.2f} MB → {save_path}")
    except HTTPException:
        raise
    except Exception as e:
        job_store.set_error(job_id, str(e))
        logger.error(f"[{job_id}] File save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Queue the job in Celery (production path), with local-thread fallback.
    queue_mode = "celery"
    try:
        celery_app.send_task(
            "app.workers.tasks.process_video_job",
            args=[job_id, str(save_path), drill_key],
        )
        logger.info(f"[{job_id}] Job queued in Celery")
    except Exception as e:
        queue_mode = "thread-fallback"
        logger.warning(f"[{job_id}] Celery unavailable ({e}) — using local thread fallback")
        thread = threading.Thread(
            target=process_video,
            args=(job_id, str(save_path), drill_key),
            daemon=True,
            name=f"processor-{job_id[:8]}",
        )
        thread.start()
        logger.info(f"[{job_id}] Background processing thread started")

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "drill_type": drill_key,
            "message": "Video uploaded successfully. Processing has started.",
            "queue_mode": queue_mode,
            "poll_url": f"/status/{job_id}",
            "report_url": f"/report/{job_id}",
        },
    )


@router.get("/status/{job_id}", summary="Check processing status of a job")
async def get_status(job_id: str):
    """Returns current processing status without the full report."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "drill_type": job["drill_type"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }


@router.get("/jobs", summary="List all jobs")
async def list_jobs():
    """Return summary of all submitted jobs."""
    return {"jobs": job_store.list_jobs()}
