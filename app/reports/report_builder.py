"""
Report builder: assembles the final JSON assessment report.
"""
import time
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_report(
    job_id: str,
    drill_type: str,
    drill_display_name: str,
    player_score: float,
    metrics: Dict[str, float],
    issues: List[str],
    recommendations: List[str],
    frame_count: int,
    detected_frame_count: int,
    processing_time_sec: float,
    video_filename: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the complete assessment report JSON.
    """
    detection_rate = (
        round(detected_frame_count / frame_count, 3) if frame_count > 0 else 0.0
    )

    # Performance band
    if player_score >= 85:
        performance_band = "Elite"
    elif player_score >= 70:
        performance_band = "Advanced"
    elif player_score >= 55:
        performance_band = "Intermediate"
    elif player_score >= 40:
        performance_band = "Developing"
    else:
        performance_band = "Beginner"

    report = {
        "job_id": job_id,
        "player_score": player_score,
        "performance_band": performance_band,
        "drill_type": drill_display_name,
        "drill_type_key": drill_type,
        "metrics": metrics,
        "issues": issues,
        "recommendations": recommendations,
        "analysis_meta": {
            "video_filename": video_filename,
            "frame_count": frame_count,
            "detected_frame_count": detected_frame_count,
            "pose_detection_rate": detection_rate,
            "processing_time_sec": round(processing_time_sec, 2),
        },
    }

    if extra:
        report.update(extra)

    logger.info(
        f"Report built | job={job_id} | score={player_score} | "
        f"band={performance_band} | drill={drill_display_name}"
    )
    return report
