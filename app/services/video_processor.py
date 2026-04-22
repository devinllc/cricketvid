"""
Video processor: orchestrates the complete analysis pipeline.
Runs in a background thread per job.
"""
import os
import time
from pathlib import Path
from typing import Optional

from app.ai import cricket_scorer, metric_calculator, pose_detector
from app.reports.report_builder import build_report
from app.services import frame_extractor, normalizer
from app.utils import job_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_video(job_id: str, video_path: str, drill_type: str) -> None:
    """
    Full pipeline:
    1. Normalize video (FFmpeg / OpenCV)
    2. Extract frames
    3. Run pose detection
    4. Compute metrics
    5. Score
    6. Build report
    7. Store in job_store
    """
    start_time = time.time()
    video_filename = Path(video_path).name

    try:
        logger.info(f"[{job_id}] Starting pipeline for '{video_filename}' ({drill_type})")
        job_store.update_status(job_id, job_store.STATUS_PROCESSING)

        # ── Step 1: Normalize video ──────────────────────────────────────
        logger.info(f"[{job_id}] Step 1/6: Normalizing video")
        job_dir = PROCESSED_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            normalized_path = normalizer.enhance_video(video_path, str(job_dir))
        except Exception as e:
            logger.warning(f"[{job_id}] Normalization failed ({e}) — using original")
            normalized_path = video_path

        # ── Step 2: Extract frames ───────────────────────────────────────
        logger.info(f"[{job_id}] Step 2/6: Extracting frames")
        frames, fps = frame_extractor.extract_frames(normalized_path)
        frame_count = len(frames)
        logger.info(f"[{job_id}] Extracted {frame_count} frames at {fps:.1f} fps")

        if frame_count == 0:
            raise RuntimeError("No frames could be extracted from video")

        # ── Step 3: Pose detection ───────────────────────────────────────
        logger.info(f"[{job_id}] Step 3/6: Running pose detection")
        landmark_sequence = pose_detector.detect_poses(frames)
        detected_count = sum(1 for lm in landmark_sequence if lm is not None)
        detection_rate = detected_count / frame_count
        logger.info(
            f"[{job_id}] Pose detected in {detected_count}/{frame_count} frames "
            f"({detection_rate:.1%})"
        )

        # ── Step 4: Compute metrics ──────────────────────────────────────
        logger.info(f"[{job_id}] Step 4/6: Computing biomechanical metrics")
        metrics = metric_calculator.compute_all_metrics(landmark_sequence, drill_type)
        logger.info(f"[{job_id}] Metrics: {metrics}")

        # ── Step 5: Score ────────────────────────────────────────────────
        logger.info(f"[{job_id}] Step 5/6: Applying cricket scoring logic")
        scoring_result = cricket_scorer.score(
            metrics, drill_type, detection_rate=detection_rate
        )
        drill_display_name = cricket_scorer.get_drill_display_name(drill_type)

        # ── Step 6: Build report ─────────────────────────────────────────
        logger.info(f"[{job_id}] Step 6/6: Building assessment report")
        processing_time = time.time() - start_time
        report = build_report(
            job_id=job_id,
            drill_type=drill_type,
            drill_display_name=drill_display_name,
            player_score=scoring_result["player_score"],
            metrics=metrics,
            issues=scoring_result["issues"],
            recommendations=scoring_result["recommendations"],
            frame_count=frame_count,
            detected_frame_count=detected_count,
            processing_time_sec=processing_time,
            video_filename=video_filename,
        )

        job_store.set_report(job_id, report)
        logger.info(
            f"[{job_id}] ✅ Pipeline complete in {processing_time:.1f}s — "
            f"score={scoring_result['player_score']}"
        )

        # Clean up frames from memory
        frames.clear()

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(f"[{job_id}] ❌ Pipeline failed after {elapsed:.1f}s: {exc}", exc_info=True)
        job_store.set_error(job_id, str(exc))
