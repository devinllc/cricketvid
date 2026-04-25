"""Renderer helpers for broadcast-style overlays."""
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _project_points(inv_h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(pts, inv_h)
    return proj.reshape(-1, 2)


def draw_calibrated_pitch_overlay(frame: np.ndarray, inverse_homography: Optional[np.ndarray]) -> np.ndarray:
    """Draw pitch lane and wicket guides anchored in calibrated pitch plane."""
    if inverse_homography is None:
        return frame

    # Canonical pitch coordinates in calibration plane (x: 0..300, y: 0..1200).
    lane = np.array(
        [
            [150 - 16, 120],
            [150 + 16, 120],
            [150 + 56, 1180],
            [150 - 56, 1180],
        ],
        dtype=np.float32,
    )

    try:
        lane_img = _project_points(inverse_homography, lane)
    except Exception:
        return frame

    lane_poly = lane_img.astype(np.int32)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, lane_poly, (190, 135, 60))
    frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0.0)

    # Draw wicket triplets near far and near ends.
    for y_base, color in [(170.0, (60, 210, 255)), (1110.0, (40, 160, 240))]:
        stumps = np.array(
            [
                [140.0, y_base], [140.0, y_base + 55.0],
                [150.0, y_base], [150.0, y_base + 55.0],
                [160.0, y_base], [160.0, y_base + 55.0],
            ],
            dtype=np.float32,
        )
        pts = _project_points(inverse_homography, stumps)
        for i in range(0, len(pts), 2):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[i + 1].astype(int))
            cv2.line(frame, p1, p2, color, 2, lineType=cv2.LINE_AA)

    return frame


def draw_broadcast_overlay(
    frame: np.ndarray,
    curve_points: List[Tuple[int, int]],
    draw_lane: bool = True,
) -> np.ndarray:
    """Draw red trajectory curve with glow for consistent broadcast look."""
    if len(curve_points) < 2:
        return frame

    for i in range(1, len(curve_points)):
        cv2.line(frame, curve_points[i - 1], curve_points[i], (25, 25, 25), 9, lineType=cv2.LINE_AA)
    for i in range(1, len(curve_points)):
        cv2.line(frame, curve_points[i - 1], curve_points[i], (0, 0, 255), 5, lineType=cv2.LINE_AA)

    # Bright highlight ridge for strong visibility in daylight backgrounds.
    for i in range(1, len(curve_points)):
        cv2.line(frame, curve_points[i - 1], curve_points[i], (70, 70, 255), 2, lineType=cv2.LINE_AA)

    if draw_lane:
        # Optional subtle lane underlay.
        p0 = curve_points[0]
        p1 = curve_points[-1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        n = max(1e-6, float(np.hypot(dx, dy)))
        nx, ny = -dy / n, dx / n
        lane = np.array(
            [
                [int(p0[0] - nx * 7), int(p0[1] - ny * 7)],
                [int(p0[0] + nx * 7), int(p0[1] + ny * 7)],
                [int(p1[0] + nx * 22), int(p1[1] + ny * 22)],
                [int(p1[0] - nx * 22), int(p1[1] - ny * 22)],
            ],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, lane, (180, 120, 40))
        frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0.0)

    return frame
