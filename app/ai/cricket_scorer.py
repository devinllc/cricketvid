"""
Cricket scorer: applies drill-specific rules to metric scores,
generates overall player score, issues, and recommendations.
"""
from typing import Any, Dict, List, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Drill display names
DRILL_DISPLAY_NAMES = {
    "straight_drive": "Straight Drive",
    "cover_drive": "Cover Drive",
    "bowling_action": "Bowling Action",
    "spin_bowling": "Spin Bowling",
}

# Metric weights per drill type
BATTING_WEIGHTS = {
    "head_stability": 0.25,
    "footwork": 0.20,
    "shoulder_rotation": 0.20,
    "balance": 0.20,
    "bat_path": 0.15,
}

BOWLING_WEIGHTS = {
    "runup_rhythm": 0.20,
    "arm_angle": 0.25,
    "release_height": 0.25,
    "front_foot_landing": 0.15,
    "followthrough_balance": 0.15,
}

# Issue thresholds: (metric, threshold, issue_text, recommendation_text)
BATTING_ISSUE_RULES: List[Tuple[str, float, str, str]] = [
    (
        "head_stability",
        6.0,
        "Head movement detected during stroke play",
        "Keep your head still and eyes level throughout the shot",
    ),
    (
        "footwork",
        5.5,
        "Front foot not moving early enough into position",
        "Step forward earlier — commit to the front foot drive as the ball pitches",
    ),
    (
        "shoulder_rotation",
        5.5,
        "Limited shoulder rotation through the shot",
        "Open your shoulders fully — rotate from the hips first, then follow with shoulders",
    ),
    (
        "balance",
        5.5,
        "Body balance is inconsistent at the crease",
        "Maintain a stable base — weight should transfer smoothly forward",
    ),
    (
        "bat_path",
        5.5,
        "Bat path appears slightly open or inconsistent",
        "Keep the bat straighter through impact — swing along the line of the ball",
    ),
]

BOWLING_ISSUE_RULES: List[Tuple[str, float, str, str]] = [
    (
        "runup_rhythm",
        5.5,
        "Run-up rhythm is inconsistent or choppy",
        "Maintain a smooth, rhythmic run-up — count your strides to build consistency",
    ),
    (
        "arm_angle",
        6.0,
        "Bowling arm angle may indicate elbow bend",
        "Extend your arm fully at delivery — practice shadow bowling in front of a mirror",
    ),
    (
        "release_height",
        5.5,
        "Release point appears low — losing carry and bounce",
        "Get taller at the crease — drive your bowling arm higher at release",
    ),
    (
        "front_foot_landing",
        5.5,
        "Front foot landing is not aligned correctly",
        "Aim to land your front foot close to the crease in line with the stumps",
    ),
    (
        "followthrough_balance",
        5.5,
        "Follow-through balance needs improvement",
        "Complete your follow-through — allow momentum to carry you towards fine leg",
    ),
]


def _get_issue_rules(drill_key: str) -> List[Tuple[str, float, str, str]]:
    batting = {"straight_drive", "cover_drive"}
    return BATTING_ISSUE_RULES if drill_key in batting else BOWLING_ISSUE_RULES


def _get_weights(drill_key: str) -> Dict[str, float]:
    batting = {"straight_drive", "cover_drive"}
    return BATTING_WEIGHTS if drill_key in batting else BOWLING_WEIGHTS


def score(
    metrics: Dict[str, float],
    drill_type: str,
    detection_rate: float = 1.0,
) -> Dict[str, Any]:
    """
    Apply cricket scoring logic to computed metrics.

    Args:
        metrics: dict of metric_name → raw score (0–10)
        drill_type: one of the supported drill type strings
        detection_rate: fraction of frames where pose was detected (0–1)

    Returns:
        dict with player_score, issues, recommendations
    """
    drill_key = drill_type.lower().replace(" ", "_")
    weights = _get_weights(drill_key)
    issue_rules = _get_issue_rules(drill_key)

    # Weighted player score (0–100)
    total_weight = 0.0
    weighted_sum = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            weighted_sum += metrics[metric] * weight * 10  # scale to 100
            total_weight += weight

    if total_weight > 0:
        player_score = weighted_sum / total_weight
    else:
        player_score = 50.0  # fallback neutral score

    # Penalize if detection was poor (< 50% frames detected)
    if detection_rate < 0.5:
        player_score *= 0.9
        logger.warning(
            f"Low pose detection rate ({detection_rate:.0%}) — score penalized"
        )

    player_score = round(min(100.0, max(0.0, player_score)), 1)

    # Generate issues and recommendations
    issues = []
    recommendations = []
    for metric, threshold, issue_text, rec_text in issue_rules:
        if metric in metrics and metrics[metric] < threshold:
            issues.append(issue_text)
            recommendations.append(rec_text)

    # Add detection quality note if needed
    if detection_rate < 0.4:
        issues.insert(
            0,
            "Low pose detection — video quality may be affecting analysis accuracy",
        )
        recommendations.insert(
            0,
            "Re-record with better lighting and ensure full body is visible in frame",
        )

    logger.info(
        f"Scoring complete for {drill_type}: score={player_score}, "
        f"issues={len(issues)}"
    )

    return {
        "player_score": player_score,
        "issues": issues,
        "recommendations": recommendations,
    }


def get_drill_display_name(drill_type: str) -> str:
    """Convert drill_type key to human-readable display name."""
    key = drill_type.lower().replace(" ", "_")
    return DRILL_DISPLAY_NAMES.get(key, drill_type.replace("_", " ").title())


SUPPORTED_DRILLS = list(DRILL_DISPLAY_NAMES.keys())
