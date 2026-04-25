"""Background worker tasks for video processing."""
from app.celery_app import celery_app
from app.services.video_processor import process_video


@celery_app.task(name="app.workers.tasks.process_video_job")
def process_video_job(job_id: str, video_path: str, drill_type: str) -> str:
    """Run the full analysis pipeline for a single uploaded video."""
    process_video(job_id, video_path, drill_type)
    return job_id
