from __future__ import annotations

import numpy as np

from backend.app.perception.color_detector import ColorDetector
from backend.app.perception.features import FeatureExtractor
from backend.app.perception.motion import MotionAnalyzer


def _scene() -> np.ndarray:
    img = np.full((240, 320, 3), 255, np.uint8)
    img[50:150, 50:150] = (0, 0, 220)     # red notebook
    img[50:150, 200:250] = (220, 0, 0)    # blue mug
    img[80:220, 140:190] = (200, 0, 200)  # magenta person
    return img


def test_color_detector_finds_objects():
    det = ColorDetector().detect(_scene(), frame_id=0, timestamp=0.0)
    categories = {d.category for d in det}
    assert "notebook" in categories
    assert "mug" in categories
    assert "person" in categories

    notebook = next(d for d in det if d.category == "notebook")
    assert notebook.bounding_box.x_min <= 55
    assert notebook.bounding_box.y_min <= 55
    assert notebook.confidence > 0.0


def test_color_detector_ignores_plain_scene():
    det = ColorDetector().detect(np.full((240, 320, 3), 255, np.uint8), 0, 0.0)
    assert det == []


def test_feature_extractor_orb():
    extractor = FeatureExtractor(method="orb")
    img = np.random.randint(0, 255, (240, 320, 3), np.uint8)
    keypoints, descriptors = extractor.extract(img)
    assert descriptors is not None
    assert descriptors.ndim == 2


def test_motion_analyzer():
    analyzer = MotionAnalyzer()
    static = np.full((120, 160, 3), 128, np.uint8)
    assert analyzer.optical_flow(static)[0] == 0.0
    assert analyzer.optical_flow(static)[0] == 0.0

    mask, score = analyzer.foreground(np.full((120, 160, 3), 255, np.uint8))
    assert mask.shape == (120, 160)
    assert 0.0 <= score <= 1.0
