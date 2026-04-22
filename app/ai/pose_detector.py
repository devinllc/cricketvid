"""
Pose detector: runs MediaPipe PoseLandmarker (Tasks API) on extracted frames.
Returns normalized landmark data for each frame.

Uses the new Tasks API (mediapipe.tasks.python.vision.PoseLandmarker)
which is compatible with mediapipe 0.10.x on Python 3.9.
"""
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Model path ───────────────────────────────────────────
_AI_DIR = Path(__file__).parent
_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_PATH = _AI_DIR / _MODEL_FILENAME
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_model() -> str:
    """Download model if not present. Returns absolute path string."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    logger.info(f"Downloading PoseLandmarker model → {_MODEL_PATH}")
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        logger.info("Model download complete")
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise RuntimeError(f"Cannot download MediaPipe pose model: {e}")
    return str(_MODEL_PATH)


# MediaPipe PoseLandmarker landmark indices
LANDMARK_NAMES = {
    0:  "NOSE",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}


class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.4):
        model_path = _ensure_model()
        options = PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe PoseLandmarker (Tasks API) initialized")

    def detect(self, frames: List[np.ndarray]) -> List[Optional[Dict[str, Any]]]:
        """
        Run pose detection on a list of BGR frames.

        Returns:
            List of dicts keyed by landmark name with {x, y, z, visibility},
            or None for frames where no pose was detected.
        """
        results = []
        detected = 0

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks_raw = result.pose_landmarks[0]
                world = (
                    result.pose_world_landmarks[0]
                    if result.pose_world_landmarks
                    else None
                )
                frame_data: Dict[str, Any] = {}
                for idx, name in LANDMARK_NAMES.items():
                    if idx < len(landmarks_raw):
                        lm = landmarks_raw[idx]
                        wlm = world[idx] if world and idx < len(world) else None
                        frame_data[name] = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z if wlm is None else wlm.z,
                            "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                        }
                results.append(frame_data)
                detected += 1
            else:
                results.append(None)

        detection_rate = (detected / len(frames) * 100) if frames else 0
        logger.info(
            f"Pose detection: {detected}/{len(frames)} frames "
            f"({detection_rate:.1f}%)"
        )
        return results

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def detect_poses(frames: List[np.ndarray]) -> List[Optional[Dict[str, Any]]]:
    """Convenience function: detect poses in all frames."""
    with PoseDetector() as detector:
        return detector.detect(frames)
