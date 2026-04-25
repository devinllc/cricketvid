"""Physics-oriented fitting helpers."""
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline


def fit_projectile_like_curve(points: List[Tuple[int, int]]) -> Dict[str, object]:
    """Fit smooth x(t), y(t) splines for trajectory stabilization."""
    if len(points) < 4:
        return {"ok": False, "reason": "insufficient_points"}

    t = np.arange(len(points), dtype=np.float64)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)

    # Mild smoothing tuned for small noisy detections.
    sx = UnivariateSpline(t, xs, s=max(1.0, len(points) * 0.35))
    sy = UnivariateSpline(t, ys, s=max(1.0, len(points) * 0.55))

    return {
        "ok": True,
        "spline_x": sx,
        "spline_y": sy,
    }
