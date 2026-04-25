"""
Frame extractor: samples frames evenly from a video file using OpenCV.
"""
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# For production tracking quality, keep every frame.
FRAME_STEP = 1  # Extract every Nth frame

# Temporal upsampling factor for analysis (1 = disabled, 2 = add one synthetic in-between frame).
# This improves motion continuity on short/fast ball flights.
INTERPOLATION_FACTOR = max(1, int(os.getenv("ANALYSIS_INTERPOLATION_FACTOR", "2")))


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """
    Extract frames from entire video at FRAME_STEP intervals and optionally
    interpolate intermediate frames to improve analysis continuity.

    Returns:
        frames: list of BGR numpy arrays (full video)
        fps: effective fps of returned frame sequence
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    adjusted_fps = original_fps / FRAME_STEP  # FPS of sampled original frames
    duration_sec = total_frames / original_fps

    logger.info(
        f"Video: {Path(video_path).name} | "
        f"Frames: {total_frames} | FPS: {original_fps:.1f} | "
        f"Duration: {duration_sec:.1f}s | Step: {FRAME_STEP} | Interp: x{INTERPOLATION_FACTOR}"
    )

    frames = []
    prev_kept = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_STEP == 0:
            if prev_kept is None:
                frames.append(frame)
                prev_kept = frame
            else:
                if INTERPOLATION_FACTOR > 1:
                    # Linear temporal interpolation: simple and fast for denser tracking inputs.
                    for k in range(1, INTERPOLATION_FACTOR):
                        alpha = float(k) / float(INTERPOLATION_FACTOR)
                        interp = cv2.addWeighted(prev_kept, 1.0 - alpha, frame, alpha, 0.0)
                        frames.append(interp)
                frames.append(frame)
                prev_kept = frame
        frame_idx += 1

    cap.release()
    effective_fps = adjusted_fps * INTERPOLATION_FACTOR
    logger.info(
        f"Extracted {len(frames)} frames (duration: {len(frames) / effective_fps:.1f}s) "
        f"from {Path(video_path).name}"
    )
    return frames, effective_fps
