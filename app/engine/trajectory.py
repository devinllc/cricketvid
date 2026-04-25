"""Trajectory building primitives for physics-aware path generation."""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TrajectoryPoint:
    frame_idx: int
    x: float
    y: float


def build_trajectory(points: List[Tuple[int, int]]) -> List[TrajectoryPoint]:
    return [TrajectoryPoint(frame_idx=i, x=float(p[0]), y=float(p[1])) for i, p in enumerate(points)]
