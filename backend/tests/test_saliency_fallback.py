"""Pipeline integration of the unknown-object fallback (deterministic stubs)."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.memory.models import BoundingBox, Detection
from backend.app.perception.color_detector import ColorDetector
from backend.app.pipeline.pipeline import Pipeline


class _NullDetector:
    def detect(self, image, frame_id, timestamp):
        return []


class _StubSaliency:
    """Always proposes one distinctive region, so gating is testable."""

    def __init__(self, category: str = "object") -> None:
        self.category = category

    def propose(self, mask, frame_id, timestamp, frame_shape):
        return [
            Detection(
                category=self.category,
                bounding_box=BoundingBox(
                    x_min=10, y_min=10, x_max=50, y_max=50
                ),
                confidence=0.4,
                frame_id=frame_id,
                timestamp=timestamp,
            )
        ]


def _write_video(path: str, color=None, frames: int = 30) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(frames):
        img = np.full((240, 320, 3), 255, np.uint8)
        if color is not None:
            img[100:160, 100:140] = color
        writer.write(img)
    writer.release()


def test_fallback_proposes_when_detector_empty(tmp_path):
    video = str(tmp_path / "v.mp4")
    _write_video(video)
    pipeline = Pipeline(
        detector=_NullDetector(),
        saliency=_StubSaliency(),
        scene_change_motion_gate=1e9,
    )
    result = pipeline.run(video)
    categories = {o.category for o in result.memory.object_memory.all()}
    assert "object" in categories


def test_fallback_disabled_or_not_used_yields_no_objects(tmp_path):
    video = str(tmp_path / "v.mp4")
    _write_video(video)
    pipeline = Pipeline(detector=_NullDetector(), saliency_fallback=False)
    result = pipeline.run(video)
    assert len(result.memory.object_memory) == 0


def test_fallback_not_used_when_detector_matches(tmp_path):
    video = str(tmp_path / "v.mp4")
    _write_video(video, color=(0, 0, 220))  # red notebook
    pipeline = Pipeline(
        detector=ColorDetector(),
        saliency=_StubSaliency(category="saliency_marker"),
    )
    result = pipeline.run(video)
    categories = {o.category for o in result.memory.object_memory.all()}
    assert "notebook" in categories
    assert "saliency_marker" not in categories
