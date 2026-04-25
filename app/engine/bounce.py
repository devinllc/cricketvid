"""Bounce event detection utilities."""
from typing import List, Optional, Tuple


def detect_bounce_index(points: List[Tuple[int, int]]) -> Optional[int]:
    """Detect bounce as local maximum in image y (y increases downward)."""
    if len(points) < 6:
        return None

    ys = [p[1] for p in points]
    best_i = None
    best_y = -1
    for i in range(1, len(ys) - 1):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] > best_y:
            best_y = ys[i]
            best_i = i
    return best_i
