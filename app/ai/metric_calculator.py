"""
Metric calculator: computes biomechanical metrics from MediaPipe landmark sequences.
All calculations use normalized (0-1) coordinate space from MediaPipe.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_point(lm_frame: Dict, key: str) -> Optional[np.ndarray]:
    """Extract [x, y, z] array from a landmark frame. Returns None if missing."""
    if lm_frame is None or key not in lm_frame:
        return None
    d = lm_frame[key]
    # Be lenient: low-light or partial occlusion still yields useful relative geometry
    if d.get("visibility", 1.0) < 0.1:
        return None
    return np.array([d["x"], d["y"], d["z"]])


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point B formed by A-B-C (degrees)."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _valid_frames(landmark_sequence: List[Optional[Dict]]) -> List[Dict]:
    """Filter out None frames."""
    return [f for f in landmark_sequence if f is not None]


def compute_head_stability(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Measure head (nose) stability across frames.
    Lower variance in Y position → higher stability.
    Returns: stability score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 3:
        return 5.0

    nose_y = []
    for frame in valid:
        p = _get_point(frame, "NOSE")
        if p is not None:
            nose_y.append(p[1])

    if len(nose_y) < 3:
        return 5.0

    variance = float(np.var(nose_y))
    # variance typically 0.0 (perfect) to 0.05+ (very unstable)
    score = max(0.0, 10.0 - (variance * 400))
    return round(min(10.0, score), 2)


def compute_shoulder_rotation(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Measure shoulder rotation angle (left vs right shoulder delta in X).
    Target for batting: rotation between 30°–90°.
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 3:
        return 5.0

    angles = []
    for frame in valid:
        ls = _get_point(frame, "LEFT_SHOULDER")
        rs = _get_point(frame, "RIGHT_SHOULDER")
        if ls is None or rs is None:
            continue
        # Angle of shoulder line to horizontal
        dx = ls[0] - rs[0]
        dy = ls[1] - rs[1]
        angle = float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-8)))
        angles.append(angle)

    if not angles:
        return 5.0

    avg_angle = float(np.mean(angles))
    max_angle = float(np.max(angles))

    # Good rotation: shoulder line angle changes by 10°+ from start to peak
    rotation_range = max_angle - float(np.min(angles))

    if rotation_range >= 15:
        score = 9.0
    elif rotation_range >= 10:
        score = 7.5
    elif rotation_range >= 5:
        score = 6.0
    else:
        score = 4.0

    return round(min(10.0, score), 2)


def compute_footwork(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Measure front foot displacement (batting: forward stride, bowling: delivery stride).
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 5:
        return 5.0

    left_ankle_y = []
    right_ankle_y = []

    for frame in valid:
        la = _get_point(frame, "LEFT_ANKLE")
        ra = _get_point(frame, "RIGHT_ANKLE")
        if la is not None:
            left_ankle_y.append(la[1])
        if ra is not None:
            right_ankle_y.append(ra[1])

    if not left_ankle_y or not right_ankle_y:
        return 5.0

    # Movement: difference between first and minimum Y (stride into ball)
    la_arr = np.array(left_ankle_y)
    ra_arr = np.array(right_ankle_y)

    left_displacement = float(np.max(la_arr) - np.min(la_arr))
    right_displacement = float(np.max(ra_arr) - np.min(ra_arr))
    max_disp = max(left_displacement, right_displacement)

    # Good footwork: significant front foot movement in Y axis
    if max_disp >= 0.15:
        score = 9.0
    elif max_disp >= 0.10:
        score = 7.5
    elif max_disp >= 0.05:
        score = 6.0
    else:
        score = 4.0

    return round(min(10.0, score), 2)


def compute_balance(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Measure body balance via hip center deviation from vertical midline.
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 3:
        return 5.0

    hip_x = []
    for frame in valid:
        lh = _get_point(frame, "LEFT_HIP")
        rh = _get_point(frame, "RIGHT_HIP")
        if lh is not None and rh is not None:
            hip_x.append((lh[0] + rh[0]) / 2.0)

    if len(hip_x) < 3:
        return 5.0

    # Balance = consistency of hip center around its mean
    variance = float(np.var(hip_x))
    score = max(0.0, 10.0 - (variance * 500))
    return round(min(10.0, score), 2)


def compute_bat_path(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Approximate bat swing path via dominant wrist trajectory.
    Good batting: wrist moves in smooth downward arc (Y increases consistently).
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 5:
        return 5.0

    wrist_positions = []
    for frame in valid:
        # Use dominant-side wrist (try right first)
        rw = _get_point(frame, "RIGHT_WRIST")
        lw = _get_point(frame, "LEFT_WRIST")
        wrist = rw if rw is not None else lw
        if wrist is not None:
            wrist_positions.append([wrist[0], wrist[1]])

    if len(wrist_positions) < 5:
        return 5.0

    wp = np.array(wrist_positions)
    # Check smoothness: variance in direction changes
    diffs = np.diff(wp, axis=0)
    direction_changes = np.sum(np.abs(np.diff(np.sign(diffs[:, 1]))))

    # Fewer direction reversals = smoother swing path
    if direction_changes <= 2:
        score = 9.0
    elif direction_changes <= 4:
        score = 7.0
    elif direction_changes <= 7:
        score = 5.5
    else:
        score = 3.5

    # Also check range of motion
    y_range = float(np.max(wp[:, 1]) - np.min(wp[:, 1]))
    if y_range > 0.3:
        score = min(10.0, score + 1.0)

    return round(min(10.0, score), 2)


def compute_arm_angle(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Measure arm angle at elbow (for bowling: should be near straight ~160°+).
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 3:
        return 5.0

    angles = []
    for frame in valid:
        rs = _get_point(frame, "RIGHT_SHOULDER")
        re = _get_point(frame, "RIGHT_ELBOW")
        rw = _get_point(frame, "RIGHT_WRIST")
        if rs is not None and re is not None and rw is not None:
            angle = _angle_between(rs, re, rw)
            angles.append(angle)

    if not angles:
        return 5.0

    max_angle = float(np.max(angles))

    # Legal bowling: elbow should approach straight (>160°)
    # Batting: natural swing arc (120–160°)
    if max_angle >= 160:
        score = 9.5
    elif max_angle >= 140:
        score = 8.0
    elif max_angle >= 120:
        score = 6.5
    elif max_angle >= 90:
        score = 5.0
    else:
        score = 3.0

    return round(score, 2)


def compute_release_height(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Estimate release point height relative to shoulder — for bowling.
    Higher above shoulder = better power and angle.
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 3:
        return 5.0

    wrist_vs_shoulder = []
    for frame in valid:
        rw = _get_point(frame, "RIGHT_WRIST")
        rs = _get_point(frame, "RIGHT_SHOULDER")
        if rw is not None and rs is not None:
            # In MediaPipe Y increases downward, so lower Y = higher position
            # Wrist higher than shoulder = negative diff
            diff = rw[1] - rs[1]
            wrist_vs_shoulder.append(diff)

    if not wrist_vs_shoulder:
        return 5.0

    min_diff = float(np.min(wrist_vs_shoulder))

    # min_diff: -0.3 or lower = excellent release height (wrist well above shoulder)
    if min_diff <= -0.3:
        score = 9.5
    elif min_diff <= -0.2:
        score = 8.0
    elif min_diff <= -0.1:
        score = 6.5
    elif min_diff <= 0.0:
        score = 5.0
    else:
        score = 3.0

    return round(score, 2)


def compute_runup_rhythm(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Estimate run-up rhythm via hip height consistency across first 40% of frames.
    Smooth hip movement = good rhythm.
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 5:
        return 5.0

    # Only look at first 40% (run-up phase)
    runup_frames = valid[: max(3, len(valid) // 2)]

    hip_y = []
    for frame in runup_frames:
        lh = _get_point(frame, "LEFT_HIP")
        rh = _get_point(frame, "RIGHT_HIP")
        if lh is not None and rh is not None:
            hip_y.append((lh[1] + rh[1]) / 2.0)

    if len(hip_y) < 3:
        return 5.0

    variance = float(np.var(hip_y))
    score = max(0.0, 10.0 - (variance * 300))
    return round(min(10.0, score), 2)


def compute_followthrough(landmark_sequence: List[Optional[Dict]]) -> float:
    """
    Measure follow-through balance via hip/shoulder position in last 30% of frames.
    Returns: score 0–10
    """
    valid = _valid_frames(landmark_sequence)
    if len(valid) < 5:
        return 5.0

    followthrough_frames = valid[-(max(3, len(valid) // 3)):]

    hip_x = []
    for frame in followthrough_frames:
        lh = _get_point(frame, "LEFT_HIP")
        rh = _get_point(frame, "RIGHT_HIP")
        if lh is not None and rh is not None:
            hip_x.append((lh[0] + rh[0]) / 2.0)

    if len(hip_x) < 2:
        return 5.0

    variance = float(np.var(hip_x))
    score = max(0.0, 10.0 - (variance * 400))
    return round(min(10.0, score), 2)


def compute_all_metrics(
    landmark_sequence: List[Optional[Dict]], drill_type: str
) -> Dict[str, float]:
    """
    Compute all relevant metrics for the given drill type.
    Returns a dict of metric_name → score (0–10).
    """
    batting_drills = {"straight_drive", "cover_drive"}
    bowling_drills = {"bowling_action", "spin_bowling"}

    drill_key = drill_type.lower().replace(" ", "_")

    logger.info(f"Computing metrics for drill: {drill_type}")

    if drill_key in batting_drills:
        return {
            "head_stability": compute_head_stability(landmark_sequence),
            "footwork": compute_footwork(landmark_sequence),
            "shoulder_rotation": compute_shoulder_rotation(landmark_sequence),
            "balance": compute_balance(landmark_sequence),
            "bat_path": compute_bat_path(landmark_sequence),
        }
    elif drill_key in bowling_drills:
        return {
            "runup_rhythm": compute_runup_rhythm(landmark_sequence),
            "arm_angle": compute_arm_angle(landmark_sequence),
            "release_height": compute_release_height(landmark_sequence),
            "front_foot_landing": compute_footwork(landmark_sequence),
            "followthrough_balance": compute_followthrough(landmark_sequence),
        }
    else:
        # Generic fallback
        logger.warning(f"Unknown drill type '{drill_type}' — using generic metrics")
        return {
            "head_stability": compute_head_stability(landmark_sequence),
            "footwork": compute_footwork(landmark_sequence),
            "balance": compute_balance(landmark_sequence),
            "arm_angle": compute_arm_angle(landmark_sequence),
            "shoulder_rotation": compute_shoulder_rotation(landmark_sequence),
        }
