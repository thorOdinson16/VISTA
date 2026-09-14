"""Pure YOLO output decoding (no model file required)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.perception.detector import decode_yolo

CLASSES = ["person", "cup"]


def _output(rows: list[list[float]]) -> np.ndarray:
    # (1, 4 + C, N)
    arr = np.asarray(rows, dtype=np.float32).T  # (N, 4+C) -> (4+C, N)
    return arr[None, :, :]


def test_decode_maps_classes_and_scales_boxes():
    output = _output(
        [
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.1],  # person
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.8],  # cup
        ]
    )
    detections = decode_yolo(
        output,
        input_size=100,
        ratio=1.0,
        pad=(0, 0),
        frame_shape=(100, 100),
        conf_threshold=0.25,
        iou_threshold=0.45,
        class_names=CLASSES,
    )
    assert len(detections) == 2
    person = next(d for d in detections if d.category == "person")
    assert person.bounding_box.x_min == pytest.approx(30.0)
    assert person.bounding_box.x_max == pytest.approx(70.0)
    assert person.confidence == pytest.approx(0.9)


def test_decode_is_class_aware_and_suppresses_same_class():
    output = _output(
        [
            [0.50, 0.50, 0.4, 0.4, 0.90, 0.05],  # person, high conf
            [0.52, 0.52, 0.4, 0.4, 0.80, 0.05],  # person, overlapping
            [0.50, 0.50, 0.4, 0.4, 0.10, 0.85],  # cup, same box, other class
            [0.10, 0.10, 0.1, 0.1, 0.05, 0.05],  # below threshold
        ]
    )
    detections = decode_yolo(
        output,
        input_size=100,
        ratio=1.0,
        pad=(0, 0),
        frame_shape=(100, 100),
        conf_threshold=0.25,
        iou_threshold=0.45,
        class_names=CLASSES,
    )
    categories = sorted(d.category for d in detections)
    # Same-class overlap suppressed; different-class overlap kept.
    assert categories == ["cup", "person"]


def test_decode_accepts_transposed_layout():
    rows = [[0.5, 0.5, 0.4, 0.4, 0.9, 0.1]]
    n_by_c = np.asarray(rows, dtype=np.float32)  # (N=1, 4+C=6)
    c_by_n = n_by_c.T  # (6, 1)
    for layout in (n_by_c, c_by_n, n_by_c[None, :, :], c_by_n[None, :, :]):
        detections = decode_yolo(
            layout, 100, 1.0, (0, 0), (100, 100), 0.25, 0.45, CLASSES
        )
        assert len(detections) == 1


def test_decode_wrong_class_count_raises():
    output = _output([[0.5, 0.5, 0.4, 0.4, 0.9, 0.1]])
    with pytest.raises(ValueError, match="Unsupported YOLO output shape"):
        decode_yolo(
            output, 100, 1.0, (0, 0), (100, 100), 0.25, 0.45, ["person"]
        )


def test_decode_non_2d_raises():
    with pytest.raises(ValueError, match="Unsupported YOLO output shape"):
        decode_yolo(
            np.zeros((5,), np.float32), 100, 1.0, (0, 0), (100, 100),
            0.25, 0.45, CLASSES,
        )
