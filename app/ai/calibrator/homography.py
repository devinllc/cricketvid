"""Pitch calibration helpers.

Estimate a usable image->pitch homography from pitch-like line structure.
This is not a full calibration solver, but provides robust enough anchors for
AR lane and wicket alignment in many practice videos.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PitchCalibrationResult:
    homography: Optional[np.ndarray]
    inverse_homography: Optional[np.ndarray]
    src_corners: Optional[np.ndarray]
    dst_corners: Optional[np.ndarray]
    confidence: float


def _line_intersection(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float32)


def _pick_boundary_lines(lines: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    """Pick left/right side lines and top/bottom cross lines from Hough output."""
    vertical = []
    horizontal = []
    total_len = 0.0

    for l in lines:
        x1, y1, x2, y2 = l[0].astype(float)
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 70:
            continue
        total_len += length
        if abs(dy) > abs(dx) * 1.25:
            x_at_bottom = x1 + (x2 - x1) * max(0.0, (1080 - y1) / (dy + 1e-6))
            vertical.append((x_at_bottom, length, np.array([x1, y1, x2, y2], dtype=np.float32)))
        elif abs(dx) > abs(dy) * 1.6:
            y_mid = 0.5 * (y1 + y2)
            horizontal.append((y_mid, length, np.array([x1, y1, x2, y2], dtype=np.float32)))

    if len(vertical) < 2:
        return None, None, None, None, 0.0

    vertical.sort(key=lambda t: t[0])
    left = max(vertical[: max(1, len(vertical) // 2)], key=lambda t: t[1])[2]
    right = max(vertical[max(1, len(vertical) // 2):], key=lambda t: t[1])[2]

    top = None
    bottom = None
    if horizontal:
        horizontal.sort(key=lambda t: t[0])
        top = max(horizontal[: max(1, len(horizontal) // 2)], key=lambda t: t[1])[2]
        bottom = max(horizontal[max(1, len(horizontal) // 2):], key=lambda t: t[1])[2]

    strength = float(np.clip(total_len / 4000.0, 0.0, 1.0))
    return left, right, top, bottom, strength


def estimate_pitch_homography(frame: np.ndarray) -> PitchCalibrationResult:
    """Estimate image->pitch homography from detected line structure."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=75, minLineLength=max(60, w // 10), maxLineGap=20)

    if lines is None:
        return PitchCalibrationResult(
            homography=None,
            inverse_homography=None,
            src_corners=None,
            dst_corners=None,
            confidence=0.0,
        )

    left, right, top, bottom, strength = _pick_boundary_lines(lines)
    if left is None or right is None:
        return PitchCalibrationResult(
            homography=None,
            inverse_homography=None,
            src_corners=None,
            dst_corners=None,
            confidence=round(0.25 * strength, 3),
        )

    if top is not None and bottom is not None:
        tl = _line_intersection(left, top)
        tr = _line_intersection(right, top)
        bl = _line_intersection(left, bottom)
        br = _line_intersection(right, bottom)
    else:
        # Fallback if top/bottom lines are weak: intersect sides with frame bounds.
        top_line = np.array([0, 0, w - 1, 0], dtype=np.float32)
        bottom_line = np.array([0, h - 1, w - 1, h - 1], dtype=np.float32)
        tl = _line_intersection(left, top_line)
        tr = _line_intersection(right, top_line)
        bl = _line_intersection(left, bottom_line)
        br = _line_intersection(right, bottom_line)

    if tl is None or tr is None or br is None or bl is None:
        return PitchCalibrationResult(
            homography=None,
            inverse_homography=None,
            src_corners=None,
            dst_corners=None,
            confidence=round(0.35 * strength, 3),
        )

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [300.0, 0.0],
            [300.0, 1200.0],
            [0.0, 1200.0],
        ],
        dtype=np.float32,
    )

    # Basic geometric sanity checks.
    area = cv2.contourArea(src)
    if area < (w * h * 0.02):
        return PitchCalibrationResult(
            homography=None,
            inverse_homography=None,
            src_corners=src,
            dst_corners=dst,
            confidence=round(0.45 * strength, 3),
        )

    H = cv2.getPerspectiveTransform(src, dst)
    Hinv = None
    try:
        Hinv = np.linalg.inv(H)
    except Exception:
        Hinv = None

    geom_conf = float(np.clip(area / (w * h * 0.20), 0.0, 1.0))
    confidence = round(float(np.clip(0.55 * strength + 0.45 * geom_conf, 0.0, 1.0)), 3)
    return PitchCalibrationResult(
        homography=H,
        inverse_homography=Hinv,
        src_corners=src,
        dst_corners=dst,
        confidence=confidence,
    )
