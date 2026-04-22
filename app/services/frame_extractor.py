"""
Frame extractor: samples frames evenly from a video file using OpenCV.
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_FRAMES = 60  # Maximum frames to sample from any video


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """
    Extract up to MAX_FRAMES evenly spaced frames from a video.

    Returns:
        frames: list of BGR numpy arrays
        fps: original frames per second
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps

    logger.info(
        f"Video: {Path(video_path).name} | "
        f"Frames: {total_frames} | FPS: {fps:.1f} | "
        f"Duration: {duration_sec:.1f}s"
    )

    # Calculate evenly spaced sample indices
    sample_count = min(MAX_FRAMES, total_frames)
    if sample_count <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    sample_indices = set(
        int(i * (total_frames - 1) / max(sample_count - 1, 1))
        for i in range(sample_count)
    )

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sample_indices:
            frames.append(frame)
        frame_idx += 1
        if len(frames) >= sample_count:
            break

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
    return frames, fps
