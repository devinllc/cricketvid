"""
Video normalizer: uses FFmpeg to enhance low-quality cricket videos.
Falls back to OpenCV-based enhancement if FFmpeg is unavailable.
"""
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Check FFmpeg availability safely (don't crash if not installed)
def _check_ffmpeg() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


FFMPEG_AVAILABLE = _check_ffmpeg()
if FFMPEG_AVAILABLE:
    logger.info("FFmpeg available — using FFmpeg video enhancement")
else:
    logger.info("FFmpeg not found — using OpenCV enhancement fallback")


def enhance_video(input_path: str, output_dir: str) -> str:
    """
    Normalize a video file:
    - Preserve source resolution
    - Denoise
    - Stabilize brightness
    Returns the path to the enhanced video file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"norm_{input_path.stem}.mp4"

    if FFMPEG_AVAILABLE:
        logger.info(f"Enhancing video with FFmpeg: {input_path.name}")
        _ffmpeg_enhance(str(input_path), str(output_path))
    else:
        logger.warning("FFmpeg not found — falling back to OpenCV enhancement")
        _opencv_enhance(str(input_path), str(output_path))

    return str(output_path)


def _ffmpeg_enhance(input_path: str, output_path: str) -> None:
    """Run FFmpeg with mild denoise + brightness normalization, preserving resolution."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf",
        (
            "hqdn3d=0.8:0.8:3:3,"
            "eq=brightness=0.01:contrast=1.02"
        ),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "17",
        "-an",          # strip audio (not needed for pose)
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        logger.error(f"FFmpeg failed: {err[:500]}")
        raise RuntimeError(f"FFmpeg normalization failed: {err[:200]}")
    logger.info(f"FFmpeg enhancement complete → {output_path}")


def _opencv_enhance(input_path: str, output_path: str) -> None:
    """Fallback: OpenCV frame-level CLAHE while preserving original resolution."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # CLAHE on luminance channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # Init writer with actual frame size
        if out is None:
            h2, w2 = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (w2, h2))
        out.write(frame)

    cap.release()
    if out:
        out.release()
    logger.info(f"OpenCV enhancement complete → {output_path}")
