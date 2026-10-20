"""The verify loop: what the crop check sees decides the next tool call."""

import cv2
import numpy as np
import pytest

from backend.app.agent import (
    NEEDS_CONFIRMATION,
    NOT_FOUND,
    VERIFIED,
    AgentConfig,
    LocateAgent,
)
from backend.app.ingestion.sampler import FrameSampler
from backend.app.memory.models import BoundingBox, Detection
from backend.app.pipeline.pipeline import Pipeline

W, H = 320, 240
BOX = BoundingBox(x_min=100, y_min=80, x_max=220, y_max=160)


class FixedDetector:
    """Emits one fixed box per frame (builds the memory)."""

    name = "fixed"

    def __init__(self, category="mug", boxes=(BOX,)):
        self.category = category
        self.boxes = boxes

    def detect(self, image, frame_id, timestamp):
        return [
            Detection(
                category=self.category,
                bounding_box=b,
                confidence=0.9,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            for b in self.boxes
        ]


class CropDetector:
    """Second opinion on a zoomed crop; scripted by call index.

    The claimed box sits in the middle of the crop (margin 0.25 -> 1/6..5/6),
    so a detection spanning that region maps back onto the claimed box.
    """

    name = "crop"

    def __init__(self, script):
        self.script = list(script)  # confidence per call, 0 = detector sees nothing
        self.calls = 0

    def detect(self, image, frame_id, timestamp):
        conf = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if conf <= 0:
            return []
        h, w = image.shape[:2]
        return [
            Detection(
                category="mug",
                bounding_box=BoundingBox(
                    x_min=w / 6, y_min=h / 6, x_max=5 * w / 6, y_max=5 * h / 6
                ),
                confidence=conf,
                frame_id=frame_id,
                timestamp=timestamp,
            )
        ]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    for _ in range(30):
        frame = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(frame, (100, 80), (220, 160), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def build_memory(video, detector=None):
    pipeline = Pipeline(
        detector=detector or FixedDetector(),
        sampler=FrameSampler(strategy="fixed", sample_fps=5.0, source_fps=10.0),
    )
    return pipeline.run(video).memory


def make_agent(video, memory, crop_script, rescan=None, **cfg):
    return LocateAgent(
        memory=memory,
        detector=CropDetector(crop_script),
        video_path=video,
        fps=10.0,
        rescan=rescan,
        config=AgentConfig(**cfg),
    )


def test_verified_candidate_is_answered_with_a_trace(video):
    memory = build_memory(video)
    outcome = make_agent(video, memory, [0.8]).locate("mug")
    assert outcome.status == VERIFIED
    assert outcome.objects and outcome.confidence == 0.8
    tools = [s["tool"] for s in outcome.trace]
    assert tools == ["lookup_memory", "inspect_crop"]
    assert outcome.trace[1]["observation"]["verified"] is True


def test_rejected_candidate_triggers_dense_rescan_then_human(video):
    memory = build_memory(video)
    calls = []

    def rescan(fps, labels):
        calls.append((fps, labels))
        return memory

    outcome = make_agent(video, memory, [0.1], rescan=rescan).locate("mug")
    assert outcome.status == NEEDS_CONFIRMATION
    assert calls == [(6.0, None)]  # one denser re-scan, no label change
    assert [s["tool"] for s in outcome.trace][-1] == "ask_human"
    assert outcome.candidates and outcome.candidates[0]["object_id"]


def test_rescan_can_recover_a_verification(video):
    memory = build_memory(video)
    # First look fails (0.1), after the denser re-scan the crop verifies (0.7).
    outcome = make_agent(
        video, memory, [0.1, 0.7], rescan=lambda fps, labels: memory
    ).locate("mug")
    assert outcome.status == VERIFIED
    assert outcome.rescans == 1
    assert "rescan" in [s["tool"] for s in outcome.trace]


def test_unknown_category_rescans_with_the_label(video):
    memory = build_memory(video, FixedDetector(category="mug"))
    seen = []

    def rescan(fps, labels):
        seen.append(labels)
        return build_memory(video, FixedDetector(category="coin"))

    agent = make_agent(video, memory, [0.9], rescan=rescan)
    agent.detector = CropDetector([0.9])
    outcome = agent.locate("coin")
    assert seen[0] == ["coin"]  # the label rescan comes first
    # The crop detector only reports "mug", so the coin cannot be verified.
    assert outcome.status in (NEEDS_CONFIRMATION, VERIFIED)
    assert outcome.trace[1]["tool"] == "rescan"


def test_nothing_remembered_and_no_rescan_is_not_found(video):
    memory = build_memory(video)
    outcome = make_agent(video, memory, [0.9]).locate("coin")
    assert outcome.status == NOT_FOUND
    assert outcome.objects == []


def test_higher_threshold_changes_the_decision(video):
    memory = build_memory(video)
    lenient = make_agent(video, memory, [0.4], verify_threshold=0.3).locate("mug")
    strict = make_agent(video, memory, [0.4], verify_threshold=0.6).locate("mug")
    assert lenient.status == VERIFIED
    assert strict.status == NEEDS_CONFIRMATION
