"""Simple Kalman-based ball tracker used as a smoothing primitive."""
from typing import Optional, Tuple

import cv2
import numpy as np


class KalmanBallTracker:
    """2D constant-velocity Kalman tracker for ball center smoothing."""

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self._initialized = False

    def update(self, measurement: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        prediction = self.kf.predict()

        if measurement is not None:
            mx, my = measurement
            m = np.array([[np.float32(mx)], [np.float32(my)]])
            self.kf.correct(m)
            self._initialized = True
            return mx, my

        if not self._initialized:
            return 0, 0

        return int(prediction[0][0]), int(prediction[1][0])
