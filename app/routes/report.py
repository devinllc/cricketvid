"""
Report routes: GET /report/{job_id}
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.utils import job_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/report/{job_id}", summary="Get assessment report for a job")
async def get_report(job_id: str, request: Request, format: str = "json"):
    """
    Retrieve the completed assessment report.
    Accepts ?format=json (default) or ?format=html for browser view.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == job_store.STATUS_PROCESSING:
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Analysis is still in progress. Please poll again in a few seconds.",
        }

    if job["status"] == job_store.STATUS_QUEUED:
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job is queued for processing.",
        }

    if job["status"] == job_store.STATUS_FAILED:
        raise HTTPException(
            status_code=500,
            detail={
                "job_id": job_id,
                "status": "failed",
                "error": job.get("error", "Unknown error"),
            },
        )

    report = job.get("report")
    if not report:
        raise HTTPException(status_code=500, detail="Report not found for completed job")

    if format.lower() == "html":
        return templates.TemplateResponse(
            "report.html",
            {"request": request, "report": report, "job_id": job_id},
        )

    return report
