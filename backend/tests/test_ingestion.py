from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.app.ingestion.reader import Frame, VideoReader
from backend.app.ingestion.sampler import FrameSampler


def make_video(path: str, n_frames: int = 30, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(n_frames):
        img = np.full((64, 64, 3), i * 5 % 255, np.uint8)
        writer.write(img)
    writer.release()


def make_frame(frame_id: int, value: int = 0, fps: float = 10.0) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=frame_id / fps,
        image=np.full((180, 320, 3), value, np.uint8),
        source_id="test",
    )


def test_reader_info_and_frames(tmp_path):
    path = str(tmp_path / "v.mp4")
    make_video(path, n_frames=30, fps=10.0)
    with VideoReader(path) as reader:
        info = reader.info
        assert info.width == 64
        assert info.height == 64
        assert info.frame_count == 30
        assert abs(info.fps - 10.0) < 0.5

        frames = list(reader.frames())
        assert len(frames) == 30
        assert frames[0].frame_id == 0
        assert abs(frames[5].timestamp - 0.5) < 1e-6


def test_reader_rejects_bad_file(tmp_path):
    with pytest.raises(ValueError):
        VideoReader(str(tmp_path / "missing.mp4"))


def test_sampler_all():
    sampler = FrameSampler(strategy="all")
    assert sampler.decide(make_frame(0))
    assert sampler.decide(make_frame(1))


def test_sampler_fixed():
    sampler = FrameSampler(strategy="fixed", sample_fps=2.0, source_fps=10.0)
    decisions = [sampler.decide(make_frame(i)) for i in range(20)]
    assert sum(decisions) == 4  # interval == 5 frames


def test_sampler_adaptive_detects_change():
    sampler = FrameSampler(
        strategy="adaptive",
        sample_fps=10.0,
        source_fps=10.0,
        change_threshold=50.0,
    )
    assert sampler.decide(make_frame(0, value=0))  # interval == 1, always samples
    assert sampler.decide(make_frame(1, value=0))
    assert sampler.decide(make_frame(2, value=255))
