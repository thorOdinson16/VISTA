"""Generic salient-region proposals (unclassified object fallback)."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.memory.models import BoundingBox, Detection
from backend.app.perception.saliency import (
    SaliencyConfirmer,
    SalientObjectDetector,
)


def test_propose_scales_to_source_and_labels_unclassified():
    mask = np.zeros((180, 320), np.uint8)
    cv2.rectangle(mask, (40, 40), (120, 120), 255, -1)  # ~80x80 region
    detector = SalientObjectDetector(min_area=100)
    detections = detector.propose(
        mask, frame_id=3, timestamp=1.5, frame_shape=(720, 1280)
    )
    assert len(detections) == 1
    detection = detections[0]
    assert detection.category == "object"
    assert detection.frame_id == 3
    assert detection.timestamp == 1.5
    box = detection.bounding_box
    # mask coords scale 4x to the 1280x720 source frame.
    assert 158 <= box.x_min <= 162
    assert 158 <= box.y_min <= 162
    assert 316 <= box.width <= 324


def test_propose_rejects_small_and_whole_frame_regions():
    small = np.zeros((180, 320), np.uint8)
    cv2.rectangle(small, (10, 10), (15, 15), 255, -1)
    # Scaled area is (5*4)*(5*4) = 400 source px, below the 1000 threshold.
    detector = SalientObjectDetector(min_area=1000)
    assert detector.propose(small, 0, 0.0, (720, 1280)) == []

    whole = np.full((180, 320), 255, np.uint8)
    assert detector.propose(whole, 0, 0.0, (720, 1280)) == []


def test_propose_caps_number_and_orders_by_area():
    mask = np.zeros((180, 320), np.uint8)
    for i in range(5):
        size = 20 + i * 5
        cv2.rectangle(
            mask, (10 + i * 30, 10), (10 + i * 30 + size, 10 + size), 255, -1
        )
    detector = SalientObjectDetector(min_area=100, max_objects=2)
    detections = detector.propose(mask, 0, 0.0, (720, 1280))
    assert len(detections) == 2


def test_propose_empty_mask():
    detector = SalientObjectDetector()
    assert detector.propose(None, 0, 0.0, (720, 1280)) == []
    assert detector.propose(np.zeros((180, 320), np.uint8), 0, 0.0, (720, 1280)) == []


def test_rejects_border_touching_region():
    mask = np.zeros((180, 320), np.uint8)
    cv2.rectangle(mask, (0, 40), (40, 120), 255, -1)  # touches left edge
    detector = SalientObjectDetector(min_area=100)
    assert detector.propose(mask, 0, 0.0, (720, 1280)) == []


def test_rejects_extreme_aspect_ratio():
    mask = np.zeros((180, 320), np.uint8)
    cv2.rectangle(mask, (100, 10), (110, 150), 255, -1)  # 11x141
    detector = SalientObjectDetector(min_area=100, max_aspect=4.0)
    assert detector.propose(mask, 0, 0.0, (720, 1280)) == []


def test_rejects_low_solidity_cross():
    mask = np.zeros((180, 320), np.uint8)
    # Plus/cross shape: low fill relative to its bounding box.
    cv2.rectangle(mask, (120, 30), (140, 150), 255, -1)
    cv2.rectangle(mask, (60, 80), (200, 100), 255, -1)
    detector = SalientObjectDetector(min_area=100, min_solidity=0.6)
    assert detector.propose(mask, 0, 0.0, (720, 1280)) == []


def _det(x_min, y_min, x_max, y_max):
    return Detection(
        category="object",
        bounding_box=BoundingBox(
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
        ),
        confidence=0.4,
        frame_id=0,
        timestamp=0.0,
    )


def test_confirmer_requires_two_frames():
    confirmer = SaliencyConfirmer(min_overlap=0.1)
    box = _det(10, 10, 50, 50)
    assert confirmer.confirm([box]) == []  # first frame: held back
    assert confirmer.confirm([box]) == [box]  # persists -> confirmed


def test_confirmer_rejects_non_overlapping():
    confirmer = SaliencyConfirmer(min_overlap=0.1)
    confirmer.confirm([_det(10, 10, 50, 50)])
    assert confirmer.confirm([_det(200, 200, 240, 240)]) == []


def test_confirmer_reset_clears_candidates():
    confirmer = SaliencyConfirmer()
    box = _det(10, 10, 50, 50)
    confirmer.confirm([box])
    confirmer.reset()
    assert confirmer.confirm([box]) == []
