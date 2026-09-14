"""Scene-change detection and optical-flow-driven adaptive sampling."""

from __future__ import annotations

import numpy as np

from backend.app.ingestion.reader import Frame
from backend.app.ingestion.sampler import FrameSampler
from backend.app.memory.models import EventType
from backend.app.memory.scene_change import SceneChangeDetector


def test_baseline_sample_emits_nothing():
    detector = SceneChangeDetector(change_threshold=0.15, persistence=2)
    detector.observe(0, 0.0, 0.05)
    detector.observe(1, 0.1, 0.05)
    assert detector.events() == []


def test_single_frame_spike_is_not_a_change():
    detector = SceneChangeDetector(change_threshold=0.15, persistence=2)
    detector.observe(0, 0.0, 0.02)  # baseline
    detector.observe(1, 0.1, 0.90)  # one-frame spike
    detector.observe(2, 0.2, 0.02)  # back to baseline
    assert detector.events() == []


def test_sustained_change_emits_one_event():
    detector = SceneChangeDetector(change_threshold=0.15, persistence=2)
    detector.observe(0, 0.0, 0.02)
    detector.observe(1, 0.1, 0.80)
    detector.observe(2, 0.2, 0.82)
    events = detector.events()
    assert len(events) == 1
    event = events[0]
    assert event.type is EventType.SCENE_CHANGED
    assert event.kind == "observed"
    assert event.start_time == 0.1
    assert 0.5 <= event.confidence <= 1.0
    assert event.details["foreground_delta"] > 0


def test_sustained_plateau_does_not_repeat():
    detector = SceneChangeDetector(change_threshold=0.15, persistence=2)
    detector.observe(0, 0.0, 0.02)
    for i in range(1, 6):
        detector.observe(i, i * 0.1, 0.80)
    assert len(detector.events()) == 1


def test_global_camera_motion_suppresses_change():
    detector = SceneChangeDetector(
        change_threshold=0.15, persistence=2, motion_gate=2.0
    )
    detector.observe(0, 0.0, 0.02)
    # Large foreground change but the camera is moving: no event.
    for i in range(1, 5):
        detector.observe(i, i * 0.1, 0.90, motion_magnitude=10.0)
    assert detector.events() == []
    # Once the camera settles, a sustained change is still detected.
    for i in range(5, 8):
        detector.observe(i, i * 0.1, 0.90, motion_magnitude=0.0)
    assert len(detector.events()) == 1


def test_gentle_drift_is_baseline_not_change():
    detector = SceneChangeDetector(
        change_threshold=0.15, persistence=2, baseline_alpha=0.2
    )
    score = 0.0
    for i in range(20):
        score += 0.01
        detector.observe(i, i * 0.1, score)
    assert detector.events() == []


class _FakeMotion:
    def __init__(self, magnitude: float) -> None:
        self.magnitude = magnitude

    def optical_flow(self, image):
        return self.magnitude, None


def _frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=frame_id / 10.0,
        image=np.full((180, 320, 3), 128, np.uint8),
        source_id="t",
    )


def test_adaptive_sampler_samples_early_on_motion():
    sampler = FrameSampler(
        strategy="adaptive",
        sample_fps=2.0,
        source_fps=10.0,
        motion=_FakeMotion(magnitude=99.0),
    )
    decisions = [sampler.decide(_frame(i)) for i in range(5)]
    # Base interval is 5; frame 0 always samples, motion triggers an early
    # sample before the next base frame.
    assert decisions[0] is True
    assert any(decisions[1:])


def test_adaptive_sampler_ignores_low_motion():
    sampler = FrameSampler(
        strategy="adaptive",
        sample_fps=2.0,
        source_fps=10.0,
        motion=_FakeMotion(magnitude=0.0),
    )
    decisions = [sampler.decide(_frame(i)) for i in range(5)]
    assert decisions == [True, False, False, False, False]
