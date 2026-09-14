"""Color-segmentation detector for the synthetic demo video.

The demo-video generator renders objects as solid colored shapes. This
detector locates them via HSV thresholding + contour extraction so the full
pipeline (tracking, memory, query) can run end-to-end without requiring
real footage or a trained model to recognise abstract shapes.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..memory.models import BoundingBox, Detection

DEFAULT_PALETTE: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "notebook": ((0, 120, 120), (10, 255, 255)),
    "mug": ((100, 120, 80), (130, 255, 255)),
    "backpack": ((40, 120, 80), (80, 255, 255)),
    "person": ((140, 100, 100), (160, 255, 255)),
}


class ColorDetector:
    """Detect solid-color objects via HSV segmentation and contours."""

    name = "ColorDetector (synthetic)"

    def __init__(
        self,
        palette: (
            dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] | None
        ) = None,
        min_area: int = 400,
        confidence: float = 0.9,
    ) -> None:
        self.palette = palette or DEFAULT_PALETTE
        self.min_area = min_area
        self.confidence = confidence

    def detect(
        self, image: np.ndarray, frame_id: int, timestamp: float
    ) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for category, (lower, upper) in self.palette.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(
                    Detection(
                        category=category,
                        bounding_box=BoundingBox(
                            x_min=float(x),
                            y_min=float(y),
                            x_max=float(x + w),
                            y_max=float(y + h),
                        ),
                        confidence=self.confidence,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                )

        return detections
