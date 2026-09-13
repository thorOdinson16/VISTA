"""Configurable frame sampling to avoid redundant processing (FR-002)."""

from __future__ import annotations

import cv2
import numpy as np

from ..perception.motion import MotionAnalyzer
from .reader import Frame


class FrameSampler:
    """Decides which frames to process downstream.

    Strategies:
    - ``all``: process every frame.
    - ``fixed``: process one frame every ``interval`` frames.
    - ``adaptive``: process at a fixed base interval, but additionally sample
      early when the frame differs enough from the last sampled frame
      (pixel change) *or* when OpenCV optical flow measures significant
      inter-frame motion, so important moments are never skipped.
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        sample_fps: float = 2.0,
        source_fps: float = 30.0,
        change_threshold: float = 8.0,
        motion_threshold: float = 1.5,
        motion: MotionAnalyzer | None = None,
    ) -> None:
        self.strategy = strategy
        self.interval = (
            max(1, int(round(source_fps / sample_fps)))
            if sample_fps > 0
            else 1
        )
        self.change_threshold = change_threshold
        self.motion_threshold = motion_threshold
        self.motion = motion or MotionAnalyzer()
        self._last_sampled_gray: np.ndarray | None = None
        self._since_sample = 0

    @staticmethod
    def _gray(frame: Frame) -> np.ndarray:
        small = cv2.resize(frame.image, (320, 180))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _sample(self, frame: Frame) -> bool:
        self._last_sampled_gray = self._gray(frame)
        self._since_sample = 0
        return True

    def decide(self, frame: Frame) -> bool:
        self._since_sample += 1

        if self.strategy == "all":
            return self._sample(frame)

        if self.strategy == "fixed":
            return self._sample(frame) if frame.frame_id % self.interval == 0 else False

        # Optical flow must be observed on every frame to stay continuous;
        # its magnitude drives early sampling under fast motion.
        flow_magnitude, _ = self.motion.optical_flow(frame.image)

        if frame.frame_id % self.interval == 0:
            return self._sample(frame)

        if (
            self._last_sampled_gray is not None
            and self._since_sample >= self.interval // 2
        ):
            diff = float(cv2.absdiff(self._gray(frame), self._last_sampled_gray).mean())
            if diff >= self.change_threshold or flow_magnitude >= self.motion_threshold:
                return self._sample(frame)

        return False
