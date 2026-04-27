"""
Simple ball tracker and trajectory overlay (beta).

Best-effort detection of a cricket ball using color + circles/contours.
Fits a quadratic curve y = ax^2 + bx + c to detected (x,y) points and renders
an overlay video with the projected path.

Outputs are in pixel units (no camera calibration). Speed is reported as
px/sec (relative). Real-world speed requires calibration — not included here.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.ai.calibrator.homography import estimate_pitch_homography
from app.ai.tracker.kalman_tracker import KalmanBallTracker
from app.engine.physics import fit_projectile_like_curve
from app.engine.renderer import draw_broadcast_overlay, draw_calibrated_pitch_overlay
from app.utils.logger import get_logger
from app.services.normalizer import FFMPEG_AVAILABLE

logger = get_logger(__name__)


@dataclass
class TrajectoryResult:
    points: List[Optional[Tuple[int, int]]]  # centers per frame (or None)
    coeffs: Optional[Tuple[float, float, float]]  # a, b, c for y=ax^2+bx+c
    px_speed_est: Optional[float]  # rough px/sec using first/last valid points
    overlay_relpath: Optional[str]  # relative to app/processed
    model_used: Optional[str] = None
    quality_score: Optional[float] = None
    bounce_frame: Optional[int] = None
    impact_frame: Optional[int] = None
    shot_type: Optional[str] = None  # "ground" | "aerial"
    shot_confidence: Optional[float] = None
    tracking_confidence: Optional[float] = None
    bounce_confidence: Optional[float] = None
    impact_confidence: Optional[float] = None
    calibration_confidence: Optional[float] = None


@dataclass
class ShotSummaryResult:
    shots: List[Dict[str, Any]]
    wagon_wheel: Dict[str, Any]
    impact_frame: Optional[int] = None
    shot_type: Optional[str] = None
    shot_confidence: Optional[float] = None
    impact_point: Optional[Tuple[int, int]] = None
    landing_point: Optional[Tuple[int, int]] = None
    region: Optional[str] = None
    side: Optional[str] = None
    summary_text: Optional[str] = None


@dataclass
class _TrajectoryFit:
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    start_idx: int
    end_idx: int
    bounce_idx: Optional[int] = None
    pre_coeffs_x: Optional[np.ndarray] = None
    pre_coeffs_y: Optional[np.ndarray] = None
    post_coeffs_x: Optional[np.ndarray] = None
    post_coeffs_y: Optional[np.ndarray] = None


_YOLO_MODEL = None
_YOLO_TRIED = False


def _get_yolo_model():
    """Lazy-load YOLO model if ultralytics is available.

    Uses env var BALL_TRACKER_MODEL (default: yolov8s.pt).
    """
    global _YOLO_MODEL, _YOLO_TRIED
    if _YOLO_TRIED:
        return _YOLO_MODEL

    _YOLO_TRIED = True
    try:
        from ultralytics import YOLO  # type: ignore

        model_name = os.getenv("BALL_TRACKER_MODEL", "yolov8s.pt")
        _YOLO_MODEL = YOLO(model_name)
        logger.info(f"AI ball detector enabled: {model_name}")
    except Exception as e:
        logger.info(f"AI ball detector unavailable, using CV fallback: {e}")
        _YOLO_MODEL = None
    return _YOLO_MODEL


def _yolo_detect_ball_center(frame: np.ndarray, prev_center: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Detect sports ball using YOLO class 32 (sports ball)."""
    model = _get_yolo_model()
    if model is None:
        return None

    try:
        # COCO class 32 = sports ball.
        results = model.predict(frame, imgsz=640, conf=0.10, classes=[32], verbose=False)
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        best = None
        best_score = -1e9
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            area = max(1.0, (x2 - x1) * (y2 - y1))
            conf = float(box.conf[0].cpu().numpy())
            score = conf * 120.0 - area * 0.002
            if prev_center is not None:
                jump = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
                score -= max(0.0, jump - 12.0) * 1.2
            if score > best_score:
                best_score = score
                best = (cx, cy)
        return best
    except Exception:
        return None


def _detect_ball_centers(frames: List[np.ndarray]) -> Tuple[List[Optional[Tuple[int, int]]], str]:
    """
    Detect ball centers using color + motion + shape cues, then clean track outliers.
    """
    centers: List[Optional[Tuple[int, int]]] = []
    prev_frame_gray = None
    prev_center: Optional[Tuple[int, int]] = None
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=24, detectShadows=False)
    yolo_hits = 0
    cv_hits = 0

    for i, frame in enumerate(frames):
        # AI-first detection (if available). Falls back to CV if not found.
        yolo_center = _yolo_detect_ball_center(frame, prev_center)
        if yolo_center is not None:
            centers.append(yolo_center)
            prev_center = yolo_center
            yolo_hits += 1
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Support common practice-ball colors: red leather, yellow tennis, and white practice balls.
        lower_red1 = np.array([0, 80, 55]);   upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 55]); upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([18, 70, 70]); upper_yellow = np.array([42, 255, 255])
        lower_white = np.array([0, 0, 180]);   upper_white = np.array([180, 80, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        color_mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_yellow, mask_white))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Motion from both frame-difference and background subtraction.
        motion_mask = bg_subtractor.apply(frame)
        _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)
        if prev_frame_gray is not None:
            diff = cv2.absdiff(frame_gray, prev_frame_gray)
            diff_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)[1]
            motion_mask = cv2.bitwise_or(motion_mask, diff_mask)
        motion_mask = cv2.dilate(motion_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

        # Prefer color+motion intersection, but fallback to color-only if too sparse.
        mask = cv2.bitwise_and(color_mask, motion_mask)
        if cv2.countNonZero(mask) < 20:
            mask = color_mask
        mask = cv2.medianBlur(mask, 5)

        center = None
        h, w = frame.shape[:2]

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 14 or area > 1400:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-6))
            if circularity < 0.35:
                continue

            (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
            if radius < 2 or radius > 24:
                continue
            cx, cy = int(cx_f), int(cy_f)
            if not (8 < cx < w - 8 and 8 < cy < h - 8):
                continue

            score = float(area) + 35.0 * circularity
            if prev_center is not None:
                jump = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
                # Penalize implausible jumps heavily to keep one consistent track.
                score -= max(0.0, jump - 10.0) * 2.2

            if score > best_score:
                best_score = score
                center = (cx, cy)

        centers.append(center)
        if center is not None:
            prev_center = center
            cv_hits += 1
        prev_frame_gray = frame_gray

    cleaned = _clean_track(centers)
    detected = sum(1 for c in cleaned if c is not None)
    model_used = "AI+CV" if yolo_hits > 0 else "CV"
    logger.info(
        f"Ball centers detected in {detected}/{len(frames)} frames "
        f"(mode={model_used}, ai_hits={yolo_hits}, cv_hits={cv_hits})"
    )
    return cleaned, model_used


def _kalman_smooth_track(centers: List[Optional[Tuple[int, int]]]) -> Tuple[List[Optional[Tuple[int, int]]], float]:
    """Apply Kalman smoothing while preserving true misses.

    Confidence is derived from detection continuity.
    """
    if not centers:
        return centers, 0.0

    tracker = KalmanBallTracker()
    smoothed: List[Optional[Tuple[int, int]]] = []
    detected = 0

    for c in centers:
        pred = tracker.update(c)
        if c is not None:
            detected += 1
            smoothed.append(pred)
        else:
            smoothed.append(None)

    continuity = detected / max(1.0, float(len(centers)))
    conf = float(np.clip(0.55 + 0.45 * continuity, 0.0, 1.0))
    return smoothed, round(conf, 3)


def _poly_coeffs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fit degree-2 where possible; downgrade to linear when samples are limited."""
    if len(xs) >= 6:
        return np.polyfit(xs, ys, 2)
    if len(xs) >= 2:
        p1 = np.polyfit(xs, ys, 1)
        return np.array([0.0, p1[0], p1[1]], dtype=np.float64)
    return np.array([0.0, 0.0, float(ys[0])], dtype=np.float64)


def _clean_track(centers: List[Optional[Tuple[int, int]]]) -> List[Optional[Tuple[int, int]]]:
    """Reject outliers with robust fitting and fill short gaps using interpolation."""
    valid_idx = np.array([i for i, p in enumerate(centers) if p is not None], dtype=np.float64)
    if len(valid_idx) < 4:
        return centers

    pts = np.array([centers[int(i)] for i in valid_idx], dtype=np.float64)
    keep = np.ones(len(valid_idx), dtype=bool)

    for _ in range(3):
        idx_fit = valid_idx[keep]
        pts_fit = pts[keep]
        if len(idx_fit) < 4:
            break
        cx = _poly_coeffs(idx_fit, pts_fit[:, 0])
        cy = _poly_coeffs(idx_fit, pts_fit[:, 1])

        x_pred = np.polyval(cx, valid_idx)
        y_pred = np.polyval(cy, valid_idx)
        residual = np.hypot(pts[:, 0] - x_pred, pts[:, 1] - y_pred)
        mad = float(np.median(np.abs(residual - np.median(residual))) + 1e-6)
        threshold = max(8.0, 2.8 * mad)
        new_keep = residual <= threshold
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    clean: List[Optional[Tuple[int, int]]] = [None] * len(centers)
    kept_idx = valid_idx[keep].astype(int)
    kept_pts = pts[keep]
    for i, p in zip(kept_idx, kept_pts):
        clean[int(i)] = (int(round(p[0])), int(round(p[1])))

    if len(kept_idx) >= 4:
        cx = _poly_coeffs(kept_idx.astype(np.float64), kept_pts[:, 0])
        cy = _poly_coeffs(kept_idx.astype(np.float64), kept_pts[:, 1])

        # Fill only short gaps to keep track realistic and avoid over-painting.
        for a, b in zip(kept_idx[:-1], kept_idx[1:]):
            if 1 < b - a <= 3:
                for t in range(a + 1, b):
                    xi = int(np.clip(round(np.polyval(cx, t)), 0, 100000))
                    yi = int(np.clip(round(np.polyval(cy, t)), 0, 100000))
                    clean[t] = (xi, yi)

    return clean


def _fit_trajectory(
    centers: List[Optional[Tuple[int, int]]],
    start_idx: Optional[int] = None,
) -> Optional[_TrajectoryFit]:
    """Fit x(t) and y(t) polynomials over frame index for smooth 2D trajectory."""
    idx = np.array(
        [i for i, p in enumerate(centers) if p is not None and (start_idx is None or i >= start_idx)],
        dtype=np.float64,
    )
    if len(idx) < 5:
        return None

    pts = np.array([centers[int(i)] for i in idx], dtype=np.float64)
    try:
        coeffs_x = _poly_coeffs(idx, pts[:, 0])
        coeffs_y = _poly_coeffs(idx, pts[:, 1])
        fit = _TrajectoryFit(
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
            start_idx=int(idx.min()),
            end_idx=int(idx.max()),
        )

        # Detect bounce around the frame where vertical motion flips (down -> up).
        y_vals = pts[:, 1]
        if len(y_vals) >= 8:
            dy = np.diff(y_vals)
            candidate_local = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0]
            if len(candidate_local) > 0:
                # +1 because diff index maps between samples.
                bounce_local = int(candidate_local[np.argmax(y_vals[candidate_local + 1])] + 1)
            else:
                bounce_local = int(np.argmax(y_vals))

            # Keep bounce away from edges and require enough samples on both sides.
            if 3 <= bounce_local <= len(idx) - 4:
                pre_idx = idx[: bounce_local + 1]
                pre_pts = pts[: bounce_local + 1]
                post_idx = idx[bounce_local:]
                post_pts = pts[bounce_local:]
                fit.bounce_idx = int(idx[bounce_local])
                fit.pre_coeffs_x = _poly_coeffs(pre_idx, pre_pts[:, 0])
                fit.pre_coeffs_y = _poly_coeffs(pre_idx, pre_pts[:, 1])
                fit.post_coeffs_x = _poly_coeffs(post_idx, post_pts[:, 0])
                fit.post_coeffs_y = _poly_coeffs(post_idx, post_pts[:, 1])

        return fit
    except Exception:
        return None


def _sample_fit_point(fit: _TrajectoryFit, t: float, w: int, h: int) -> Tuple[int, int]:
    """Sample fitted point at frame-time t using piecewise bounce model when available."""
    if fit.bounce_idx is not None and fit.pre_coeffs_x is not None and fit.post_coeffs_x is not None:
        if t <= fit.bounce_idx:
            x = np.polyval(fit.pre_coeffs_x, t)
            y = np.polyval(fit.pre_coeffs_y, t)
        else:
            x = np.polyval(fit.post_coeffs_x, t)
            y = np.polyval(fit.post_coeffs_y, t)
    else:
        x = np.polyval(fit.coeffs_x, t)
        y = np.polyval(fit.coeffs_y, t)

    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return xi, yi


def _estimate_px_speed(points: List[Tuple[int, int]], fps: float) -> Optional[float]:
    if len(points) < 2 or fps <= 0:
        return None
    p0, p1 = points[0], points[-1]
    dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    duration = len(points) / fps
    if duration <= 0:
        return None
    return dist / duration  # px/sec


def _trajectory_quality(centers: List[Optional[Tuple[int, int]]], fit: Optional[_TrajectoryFit]) -> Optional[float]:
    """Compute a simple 0-100 trajectory quality score from continuity and fit residuals."""
    if fit is None:
        return None

    idx = np.array([i for i, p in enumerate(centers) if p is not None], dtype=np.float64)
    if len(idx) < 5:
        return None
    pts = np.array([centers[int(i)] for i in idx], dtype=np.float64)

    pred = np.array([_sample_fit_point(fit, float(t), 100000, 100000) for t in idx], dtype=np.float64)
    residual = np.hypot(pts[:, 0] - pred[:, 0], pts[:, 1] - pred[:, 1])
    rmse = float(np.sqrt(np.mean(np.square(residual))))

    detected = len(idx)
    continuity = detected / max(1.0, float(len(centers)))

    # Heuristic scale: <=4px RMSE excellent, >=20px poor.
    fit_score = float(np.clip(100.0 - (rmse - 4.0) * 5.5, 0.0, 100.0))
    cont_score = float(np.clip(continuity * 100.0, 0.0, 100.0))
    return round(0.65 * fit_score + 0.35 * cont_score, 2)


def _landmark_px(frame_shape: Tuple[int, int, int], frame_lm: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
    """Convert normalized landmark to pixel coordinates for a frame."""
    if frame_lm is None or name not in frame_lm:
        return None
    lm = frame_lm[name]
    if lm.get("visibility", 1.0) < 0.1:
        return None
    h, w = frame_shape[:2]
    return float(lm["x"] * w), float(lm["y"] * h)


def _detect_bat_impact_and_shot(
    frames: List[np.ndarray],
    centers: List[Optional[Tuple[int, int]]],
    landmark_sequence: Optional[List[Optional[Dict[str, Any]]]],
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """Estimate bat impact frame and classify shot as ground or aerial.

    Impact is inferred from minimum ball-to-wrist distance around striker zone.
    """
    if not landmark_sequence or len(landmark_sequence) != len(frames):
        return None, None, None

    candidates: List[Tuple[int, float]] = []
    for i, (c, lm) in enumerate(zip(centers, landmark_sequence)):
        if c is None or lm is None:
            continue
        rw = _landmark_px(frames[i].shape, lm, "RIGHT_WRIST")
        lw = _landmark_px(frames[i].shape, lm, "LEFT_WRIST")
        if rw is None and lw is None:
            continue

        bx, by = c
        dists = []
        if rw is not None:
            dists.append(float(np.hypot(bx - rw[0], by - rw[1])))
        if lw is not None:
            dists.append(float(np.hypot(bx - lw[0], by - lw[1])))
        min_dist = min(dists)

        # Keep plausible proximity around bat-hands region.
        if min_dist <= 95.0:
            candidates.append((i, min_dist))

    if not candidates:
        return None, None, None

    impact_idx, impact_dist = min(candidates, key=lambda x: x[1])

    # Classify shot from post-impact vertical trend.
    post_pts = [centers[j] for j in range(impact_idx, min(len(centers), impact_idx + 12)) if centers[j] is not None]
    if len(post_pts) < 3:
        confidence = float(np.clip(1.0 - impact_dist / 100.0, 0.25, 0.75))
        return impact_idx, None, round(confidence, 2)

    y0 = float(post_pts[0][1])
    min_after = float(min(p[1] for p in post_pts[1:]))
    rise_px = y0 - min_after  # positive rise means ball moved upward in image.

    shot_type = "aerial" if rise_px >= 22.0 else "ground"
    # Confidence blends wrist proximity and clear rise/no-rise signal.
    rise_strength = float(np.clip(abs(rise_px) / 36.0, 0.0, 1.0))
    proximity = float(np.clip(1.0 - impact_dist / 100.0, 0.0, 1.0))
    confidence = round(0.55 * proximity + 0.45 * rise_strength, 2)
    return impact_idx, shot_type, confidence


def _format_timestamp(frame_idx: int, fps: float) -> str:
    if fps <= 0:
        return "0:00"
    total_seconds = max(0.0, frame_idx / fps)
    minutes = int(total_seconds // 60)
    seconds = int(round(total_seconds % 60))
    return f"{minutes}:{seconds:02d}"


def _classify_shot_region(
    impact_point: Tuple[int, int],
    landing_point: Tuple[int, int],
    frame_shape: Tuple[int, int, int],
) -> Tuple[str, str]:
    """Map post-impact direction to a coarse batting region."""
    h, w = frame_shape[:2]
    dx = landing_point[0] - impact_point[0]
    dy = landing_point[1] - impact_point[1]

    horizontal_threshold = max(18, int(w * 0.07))
    vertical_threshold = max(10, int(h * 0.05))

    if abs(dx) <= horizontal_threshold:
        if dy >= vertical_threshold:
            return "Straight / Mid-on", "straight"
        return "Straight / Down the ground", "straight"

    if dx < 0:
        if dy >= -vertical_threshold:
            return "Cover / Extra Cover", "off_side"
        return "Backward Point", "off_side"

    if dy >= -vertical_threshold:
        return "Mid-wicket / Square Leg", "leg_side"
    return "Fine Leg / Third Man", "leg_side"


def _extract_landing_point(
    centers: List[Optional[Tuple[int, int]]],
    impact_idx: int,
) -> Optional[Tuple[int, int]]:
    post_pts = [p for i, p in enumerate(centers) if p is not None and i > impact_idx]
    if not post_pts:
        return None
    sample = post_pts[: min(6, len(post_pts))]
    xs = [p[0] for p in sample]
    ys = [p[1] for p in sample]
    return int(round(float(np.median(xs)))), int(round(float(np.median(ys))))


def _map_zone_to_wagon_wheel(region: str) -> str:
    """Map region string to wagon wheel zone code for counting."""
    zone_map = {
        "Cover / Extra Cover": "cover",
        "Mid-off": "mid_off",
        "Backward Point": "backward_point",
        "Fine Leg": "fine_leg",
        "Square Leg": "square_leg",
        "Mid-wicket": "mid_wicket",
        "Mid-on": "mid_on",
        "Straight / Mid-on": "mid_on",
        "Straight / Down the Ground": "straight",
        "Straight / Down the ground": "straight",
    }
    return zone_map.get(region, "unknown")


def _save_impact_frame(frame: np.ndarray, impact_point: Tuple[int, int], output_path: Path) -> None:
    """Save frame with impact point circle overlay."""
    f = frame.copy()
    ix, iy = impact_point
    cv2.circle(f, (ix, iy), 12, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(f, (ix, iy), 8, (0, 140, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(f, "BAT IMPACT", (max(8, ix - 80), max(24, iy - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(output_path), f)


def _build_wagon_wheel_summary(shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build wagon wheel zone counts from shots (8 zones)."""
    zones = {
        "cover": 0,
        "mid_off": 0,
        "backward_point": 0,
        "fine_leg": 0,
        "square_leg": 0,
        "mid_wicket": 0,
        "mid_on": 0,
        "straight": 0,
    }
    for shot in shots:
        zone = shot.get("zone", "unknown")
        if zone in zones:
            zones[zone] += 1

    off_side_total = zones["cover"] + zones["mid_off"] + zones["backward_point"]
    leg_side_total = zones["fine_leg"] + zones["square_leg"] + zones["mid_wicket"]
    straight_total = zones["mid_on"] + zones["straight"]

    dominant_side = None
    if off_side_total > leg_side_total and off_side_total > straight_total:
        dominant_side = "off_side"
    elif leg_side_total > off_side_total and leg_side_total > straight_total:
        dominant_side = "leg_side"
    elif straight_total > 0:
        dominant_side = "straight"

    return {
        "zones": zones,
        "off_side_total": off_side_total,
        "leg_side_total": leg_side_total,
        "straight_total": straight_total,
        "dominant_side": dominant_side,
        "summary": (
            f"Off-side: {off_side_total} shots | Leg-side: {leg_side_total} shots | Straight: {straight_total} shots"
            if shots
            else "No shot summary could be generated."
        ),
    }


def _contiguous_detection_runs(centers: List[Optional[Tuple[int, int]]], min_len: int = 8) -> List[Tuple[int, int]]:
    """Return [start, end) ranges for contiguous non-empty detections."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for idx, center in enumerate(centers):
        if center is not None and start is None:
            start = idx
        elif center is None and start is not None:
            if idx - start >= min_len:
                runs.append((start, idx))
            start = None

    if start is not None and len(centers) - start >= min_len:
        runs.append((start, len(centers)))

    return runs


def _draw_trajectory(
    frame: np.ndarray,
    centers: List[Optional[Tuple[int, int]]],
    fit: Optional[_TrajectoryFit],
    frame_idx: int,
    impact_idx: Optional[int] = None,
    shot_type: Optional[str] = None,
    physics_fit: Optional[Dict[str, object]] = None,
    inverse_homography: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw detected centers and the fitted trajectory only up to current frame.
    """
    h, w = frame.shape[:2]

    # Show only current ball point to avoid dot-clutter.
    current_center = centers[frame_idx] if 0 <= frame_idx < len(centers) else None
    if current_center is not None:
        cv2.circle(frame, current_center, 5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, current_center, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # Draw calibrated lane/wickets only when calibration is reliable.
    if inverse_homography is not None:
        frame = draw_calibrated_pitch_overlay(frame, inverse_homography)

    if fit is not None:
        start = fit.start_idx
        if impact_idx is not None:
            start = max(start, impact_idx - 1)
        end = min(frame_idx, fit.end_idx)
        if end > start:
            ts = np.linspace(start, end, num=max(25, (end - start) * 5))
            curve_pts: List[Tuple[int, int]] = []
            for t in ts:
                curve_pts.append(_sample_fit_point(fit, float(t), w, h))

            # If very short fitted segment, fallback to detected post-impact points for cleaner batting visuals.
            if len(curve_pts) < 5 and impact_idx is not None:
                curve_pts = [
                    centers[j]
                    for j in range(max(0, impact_idx - 1), min(frame_idx + 1, len(centers)))
                    if centers[j] is not None
                ]

            frame = draw_broadcast_overlay(frame, curve_pts, draw_lane=(inverse_homography is None))

            if fit.bounce_idx is not None and frame_idx >= fit.bounce_idx:
                bx, by = _sample_fit_point(fit, float(fit.bounce_idx), w, h)
                cv2.circle(frame, (bx, by), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    else:
        # Fallback: always show a batting trajectory path from detected centers.
        start_idx = max(0, impact_idx - 1) if impact_idx is not None else 0
        pts = [
            centers[j]
            for j in range(start_idx, min(frame_idx + 1, len(centers)))
            if centers[j] is not None
        ]
        if len(pts) >= 2:
            frame = draw_broadcast_overlay(frame, pts, draw_lane=(inverse_homography is None))

    if impact_idx is not None and frame_idx >= impact_idx:
        impact_pt = centers[impact_idx]
        if impact_pt is not None:
            ix, iy = impact_pt
            cv2.circle(frame, (ix, iy), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 5, (0, 140, 255), -1, lineType=cv2.LINE_AA)
            label = "BAT IMPACT"
            if shot_type == "aerial":
                label = "BAT IMPACT | AERIAL"
            elif shot_type == "ground":
                label = "BAT IMPACT | GROUND"
            cv2.putText(frame, label, (max(8, ix - 90), max(24, iy - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def _ffmpeg_reencode_to_h264(src_path: Path, dst_path: Path) -> None:
    """Re-encode to H.264 + yuv420p + faststart for broad browser support."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)


def analyze_and_overlay(
    frames: List[np.ndarray],
    fps: float,
    job_dir: Path,
    job_id: str,
    landmark_sequence: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> TrajectoryResult:
    """Detect ball per frame, fit quadratic, render overlay video. Full video duration."""
    job_dir = Path(job_dir)
    raw_out = job_dir / f"overlay_raw_{job_id}.mp4"

    calibration_conf = None
    calibration_inv_h = None
    try:
        calibration = estimate_pitch_homography(frames[0])
        calibration_conf = round(float(calibration.confidence), 3)
        # Use calibrated AR overlay only when confidence is high enough.
        calibration_inv_h = calibration.inverse_homography if calibration_conf >= 0.45 else None
    except Exception:
        calibration_conf = None
        calibration_inv_h = None

    centers_raw, model_used = _detect_ball_centers(frames)
    centers, tracking_conf = _kalman_smooth_track(centers_raw)
    valid_pts = [c for c in centers if c is not None]
    impact_frame, shot_type, shot_confidence = _detect_bat_impact_and_shot(frames, centers, landmark_sequence)
    # Batting-first trajectory: fit from impact onward (with tiny pre-roll) if impact exists.
    fit_start_idx = None
    if impact_frame is not None:
        post_valid = sum(1 for i, p in enumerate(centers) if i >= impact_frame and p is not None)
        if post_valid >= 8:
            fit_start_idx = max(0, impact_frame - 1)
    fit = _fit_trajectory(centers, fit_start_idx)
    physics_fit = fit_projectile_like_curve(valid_pts) if len(valid_pts) >= 4 else {"ok": False}
    coeffs = tuple(fit.coeffs_y.tolist()) if fit is not None else None
    # Report batting shot speed when possible (post-impact); fallback to full track speed.
    if impact_frame is not None:
        post_pts = [p for i, p in enumerate(centers) if p is not None and i >= impact_frame]
        px_speed = _estimate_px_speed(post_pts, fps) if post_pts else None
    else:
        px_speed = _estimate_px_speed(valid_pts, fps) if valid_pts else None
    quality_score = _trajectory_quality(centers, fit)
    bounce_conf = 0.0
    if fit is not None and fit.bounce_idx is not None:
        bounce_conf = 0.82 if len(valid_pts) >= 8 else 0.65

    try:
        if fps <= 0:
            fps = 24.0
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_out), fourcc, fps, (w, h))
        for i, frame in enumerate(frames):
            f = frame.copy()
            # Draw detected centers + fitted trajectory curve
            f = _draw_trajectory(
                f,
                centers,
                fit,
                i,
                impact_frame,
                shot_type,
                physics_fit,
                calibration_inv_h,
            )
            # HUD text
            if px_speed is not None:
                cv2.putText(f, f"Speed: {px_speed:.1f} px/s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if coeffs is not None:
                cv2.putText(f, f"Trajectory v5 Batting ({model_used}) | {len(valid_pts)} detections", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if fit is not None and fit.bounce_idx is not None:
                cv2.putText(f, "Bounce detected", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 255), 2)
            if quality_score is not None:
                cv2.putText(f, f"Trajectory quality: {quality_score:.0f}/100", (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2)
            if calibration_conf is not None:
                cal_text = f"Calibration confidence: {calibration_conf:.2f}"
                if calibration_inv_h is None:
                    cal_text += " (AR lane off)"
                cv2.putText(f, cal_text, (12, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 2)
            if shot_type is not None:
                cv2.putText(
                    f,
                    f"Shot type: {shot_type.title()} ({(shot_confidence or 0.0):.2f})",
                    (12, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 230, 180),
                    2,
                )
            writer.write(f)
        writer.release()

        # Prefer H.264 output for browser playback
        final_out = job_dir / f"overlay_{job_id}.mp4"
        try:
            if FFMPEG_AVAILABLE:
                _ffmpeg_reencode_to_h264(raw_out, final_out)
                raw_out.unlink(missing_ok=True)
                out_path = final_out
            else:
                out_path = raw_out
            overlay_rel = str(Path("processed") / job_id / out_path.name)
            duration = len(frames) / fps
            logger.info(f"Overlay video complete ({len(frames)} frames, {duration:.1f}s, {len(valid_pts)} ball detections) → {out_path}")
        except Exception as e:
            logger.warning(f"FFmpeg re-encode failed, keeping raw overlay: {e}")
            overlay_rel = str(Path("processed") / job_id / raw_out.name)
    except Exception as e:
        logger.warning(f"Overlay rendering failed: {e}")
        overlay_rel = None

    return TrajectoryResult(
        points=centers,
        coeffs=coeffs,
        px_speed_est=px_speed,
        overlay_relpath=overlay_rel,
        model_used=model_used,
        quality_score=quality_score,
        bounce_frame=(fit.bounce_idx if fit is not None else None),
        impact_frame=impact_frame,
        shot_type=shot_type,
        shot_confidence=shot_confidence,
        tracking_confidence=tracking_conf,
        bounce_confidence=(round(bounce_conf, 3) if bounce_conf > 0 else None),
        impact_confidence=shot_confidence,
        calibration_confidence=calibration_conf,
    )


def analyze_shot_summary(
    frames: List[np.ndarray],
    fps: float,
    job_dir: Path,
    landmark_sequence: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> ShotSummaryResult:
    """Generate a light-weight shot summary with frame captures at impact points."""
    job_dir = Path(job_dir)
    centers_raw, model_used = _detect_ball_centers(frames)
    centers, tracking_conf = _kalman_smooth_track(centers_raw)

    shots: List[Dict[str, Any]] = []
    impact_frame: Optional[int] = None
    shot_type: Optional[str] = None
    shot_confidence: Optional[float] = None
    impact_point: Optional[Tuple[int, int]] = None
    landing_point: Optional[Tuple[int, int]] = None
    region: Optional[str] = None
    side: Optional[str] = None
    summary_text: Optional[str] = None

    runs = _contiguous_detection_runs(centers)
    for shot_number, (start_idx, end_idx) in enumerate(runs, start=1):
        run_frames = frames[start_idx:end_idx]
        run_centers = centers[start_idx:end_idx]
        run_landmarks = landmark_sequence[start_idx:end_idx] if landmark_sequence and len(landmark_sequence) == len(frames) else None
        local_impact, local_shot_type, local_shot_conf = _detect_bat_impact_and_shot(run_frames, run_centers, run_landmarks)
        if local_impact is None:
            continue

        local_impact_point = run_centers[local_impact] if 0 <= local_impact < len(run_centers) else None
        local_landing_point = _extract_landing_point(run_centers, local_impact)
        if local_impact_point is None or local_landing_point is None:
            continue

        local_region, local_side = _classify_shot_region(local_impact_point, local_landing_point, run_frames[local_impact].shape)
        local_zone = _map_zone_to_wagon_wheel(local_region)
        shot_time = _format_timestamp(start_idx + local_impact, fps)

        if local_side == "off_side":
            shot_label = "Played to the off-side"
        elif local_side == "leg_side":
            shot_label = "Played to the leg-side"
        else:
            shot_label = "Played straight down the pitch"

        shot_summary = f"Shot {shot_number} ({shot_time}): {shot_label}, directed towards the {local_region} region."
        
        frame_image_path = None
        try:
            frame_name = f"shot_{shot_number}_impact.jpg"
            frame_path = job_dir / frame_name
            _save_impact_frame(run_frames[local_impact], local_impact_point, frame_path)
            frame_image_path = str(Path("processed") / job_dir.name / frame_name)
        except Exception as e:
            logger.warning(f"Failed to save impact frame for shot {shot_number}: {e}")

        shots.append(
            {
                "shot_number": shot_number,
                "timestamp": shot_time,
                "impact_frame": start_idx + local_impact,
                "impact_point": {"x": local_impact_point[0], "y": local_impact_point[1]},
                "landing_point": {"x": local_landing_point[0], "y": local_landing_point[1]},
                "region": local_region,
                "zone": local_zone,
                "side": local_side,
                "summary": shot_summary,
                "shot_type": local_shot_type,
                "shot_confidence": local_shot_conf,
                "tracking_confidence": tracking_conf,
                "model_used": model_used,
                "frame_image_path": frame_image_path,
            }
        )

        if impact_frame is None:
            impact_frame = start_idx + local_impact
            shot_type = local_shot_type
            shot_confidence = local_shot_conf
            impact_point = local_impact_point
            landing_point = local_landing_point
            region = local_region
            side = local_side

    wagon_wheel = _build_wagon_wheel_summary(shots)
    if shots:
        summary_text = "\n".join(shot["summary"] for shot in shots)
    else:
        summary_text = wagon_wheel["summary"]

    return ShotSummaryResult(
        shots=shots,
        wagon_wheel=wagon_wheel,
        impact_frame=impact_frame,
        shot_type=shot_type,
        shot_confidence=shot_confidence,
        impact_point=impact_point,
        landing_point=landing_point,
        region=region,
        side=side,
        summary_text=summary_text,
    )
