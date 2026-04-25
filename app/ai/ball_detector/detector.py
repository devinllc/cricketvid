"""Ball detector interface with optional YOLO backend.

This module is designed for incremental migration:
- Start with current tracker outputs
- Plug in custom-trained small-object detectors later
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BallDetection:
    x: int
    y: int
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


class BallDetector:
    """Detector wrapper ready for YOLOv8/RT-DETR style backends."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self._model = None
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(model_name)
        except Exception:
            self._model = None

    def detect(self, frame: np.ndarray) -> List[BallDetection]:
        """Return candidate ball detections for a frame.

        Class 32 in COCO corresponds to sports ball.
        """
        if self._model is None:
            return []

        out: List[BallDetection] = []
        try:
            results = self._model.predict(frame, imgsz=640, conf=0.10, classes=[32], verbose=False)
            if not results:
                return []
            r = results[0]
            if r.boxes is None:
                return []

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx = int(round((x1 + x2) / 2.0))
                cy = int(round((y1 + y2) / 2.0))
                conf = float(box.conf[0].cpu().numpy())
                out.append(
                    BallDetection(
                        x=cx,
                        y=cy,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
        except Exception:
            return []

        return out
