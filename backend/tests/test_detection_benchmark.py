"""Detection benchmark metric logic."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.evaluation.detection import (
    average_precision,
    evaluate_detection,
)
from backend.app.memory.models import BoundingBox, Detection


class _StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image, frame_id, timestamp):
        return [
            Detection(
                category=label,
                bounding_box=BoundingBox(
                    x_min=x1, y_min=y1, x_max=x2, y_max=y2
                ),
                confidence=conf,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            for label, x1, y1, x2, y2, conf in self._detections
        ]


def _write_image(path):
    img = np.full((100, 100, 3), 255, np.uint8)
    cv2.imwrite(str(path), img)


def _item(path):
    return {
        "path": str(path),
        "boxes": [{"label": "person", "bbox": [10, 10, 50, 50]}],
    }


def test_perfect_detection_scores_one(tmp_path):
    image = tmp_path / "a.jpg"
    _write_image(image)
    detector = _StubDetector([("person", 10, 10, 50, 50, 0.9)])
    report = evaluate_detection(detector, [_item(image)])
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["map50"] == 1.0


def test_wrong_label_scores_zero(tmp_path):
    image = tmp_path / "a.jpg"
    _write_image(image)
    detector = _StubDetector([("bus", 10, 10, 50, 50, 0.9)])
    report = evaluate_detection(detector, [_item(image)])
    assert report["precision"] == 0.0
    assert report["recall"] == 0.0
    assert report["map50"] == 0.0


def test_average_precision_interpolates():
    # One class, two predictions: TP then FP, one GT.
    ap = average_precision([(0.9, True), (0.5, False)], total_gt=1)
    assert ap == 1.0
    ap_zero = average_precision([(0.9, False)], total_gt=1)
    assert ap_zero == 0.0
