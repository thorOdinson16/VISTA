"""Composite detector fallback and open-vocab configuration (no weights)."""

from __future__ import annotations

from backend.app.config import Settings
from backend.app.memory.models import BoundingBox, Detection
from backend.app.perception.openvocab import CompositeDetector


def _det(category: str) -> Detection:
    return Detection(
        category=category,
        bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
        confidence=0.9,
        frame_id=0,
        timestamp=0.0,
    )


class _Detector:
    def __init__(self, name, detections):
        self.name = name
        self._detections = detections
        self.calls = 0

    def detect(self, image, frame_id, timestamp):
        self.calls += 1
        return list(self._detections)


def test_composite_uses_primary_when_non_empty():
    primary = _Detector("primary", [_det("person")])
    secondary = _Detector("secondary", [_det("coin")])
    composite = CompositeDetector(primary, secondary)
    assert [d.category for d in composite.detect(None, 0, 0.0)] == ["person"]
    assert secondary.calls == 0
    assert "fallback" in composite.name


def test_composite_falls_back_when_primary_empty():
    primary = _Detector("primary", [])
    secondary = _Detector("secondary", [_det("coin")])
    composite = CompositeDetector(primary, secondary)
    assert [d.category for d in composite.detect(None, 0, 0.0)] == ["coin"]
    assert primary.calls == 1
    assert secondary.calls == 1


def test_composite_without_secondary():
    primary = _Detector("primary", [])
    composite = CompositeDetector(primary, None)
    assert composite.detect(None, 0, 0.0) == []
    assert "fallback" not in composite.name


def test_detector_label_list_parsing():
    settings = Settings(detector_labels="coin, shelf ,notebook")
    assert settings.detector_label_list == ["coin", "shelf", "notebook"]
