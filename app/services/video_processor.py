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
from app.services.ball_tracker import analyze_shot_summary
from app.utils import job_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _build_shot_insight(traj) -> Optional[dict]:
    if not traj or not getattr(traj, "shot_type", None):
        return None

    shot_type = str(getattr(traj, "shot_type", "")).lower()
    conf = float(getattr(traj, "shot_confidence", 0.0) or 0.0)

    if shot_type == "aerial":
        return {
            "classification": "Lofted Stroke",
            "description": "Ball traveled in the air after bat impact (aerial trajectory).",
            "coaching": "For controlled loft, keep the head over the ball, present the full face of the bat, and finish high through the line.",
            "confidence": conf,
        }

    return {
        "classification": "Along-the-Ground Stroke",
        "description": "Ball stayed mostly along the turf after contact (ground stroke).",
        "coaching": "Maintain a stable base, stay side-on, and keep the bat path vertical through impact for cleaner along-the-ground execution.",
        "confidence": conf,
    }


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
        logger.info(f"[{job_id}] Waiting for the single processing slot")
        with job_store.processing_slot(job_id):
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
            video_height, video_width = frames[0].shape[:2] if frames else (0, 0)
            logger.info(f"[{job_id}] Extracted {frame_count} frames at {fps:.1f} fps ({video_width}x{video_height})")

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

            # ── Step 3.5: Shot direction summary ────────────────────────────
            logger.info(f"[{job_id}] Step 3.5/6: Building shot direction summary")
            shot_summary = None
            try:
                shot_summary = analyze_shot_summary(frames, fps, job_dir, landmark_sequence)
            except Exception as e:
                logger.warning(f"[{job_id}] Shot summary generation failed: {e}")

            # ── Step 4: Compute metrics ──────────────────────────────────────
            logger.info(f"[{job_id}] Step 4/6: Computing biomechanical metrics")
            metrics = metric_calculator.compute_all_metrics(landmark_sequence, drill_type)
            logger.info(f"[{job_id}] Metrics: {metrics}")

            # ── Step 5: Score ────────────────────────────────────────────────
            logger.info(f"[{job_id}] Step 5/6: Applying cricket scoring logic")
            scoring_result = cricket_scorer.score(
                metrics, drill_type, detection_rate=detection_rate
            )
            if shot_summary is not None and shot_summary.summary_text:
                scoring_result["recommendations"].insert(0, shot_summary.summary_text)
            drill_display_name = cricket_scorer.get_drill_display_name(drill_type)

            # ── Step 6: Build report ─────────────────────────────────────────
            logger.info(f"[{job_id}] Step 6/6: Building assessment report")
            processing_time = time.time() - start_time

            extra_payload = {}
            if shot_summary is not None:
                extra_payload["shot_summary"] = {
                    "summary_text": shot_summary.summary_text,
                    "shots": shot_summary.shots,
                    "wagon_wheel": shot_summary.wagon_wheel,
                    "impact_frame": shot_summary.impact_frame,
                    "impact_point": (
                        {"x": shot_summary.impact_point[0], "y": shot_summary.impact_point[1]}
                        if shot_summary.impact_point is not None
                        else None
                    ),
                    "landing_point": (
                        {"x": shot_summary.landing_point[0], "y": shot_summary.landing_point[1]}
                        if shot_summary.landing_point is not None
                        else None
                    ),
                    "shot_type": shot_summary.shot_type,
                    "shot_confidence": shot_summary.shot_confidence,
                    "region": shot_summary.region,
                    "side": shot_summary.side,
                }

            extra_payload["video_dimensions"] = {
                "width": video_width,
                "height": video_height,
                "aspect_ratio": f"{video_width}:{video_height}" if video_height > 0 else "16:9",
            }
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
                extra=(extra_payload if extra_payload else None),
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
