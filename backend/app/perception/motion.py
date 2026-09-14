"""Motion analysis via optical flow and background subtraction."""

from __future__ import annotations

import cv2
import numpy as np

# Motion cues are computed on a downscaled frame: optical flow and MOG2
# do not need full resolution, and this keeps per-frame cost independent of
# the source resolution (PR-001/PR-002).
_MOTION_WIDTH = 320


class MotionAnalyzer:
    """Estimate motion between consecutive frames and detect scene change.

    MOG2 reports the entire frame as foreground on its first call because
    the background model is still empty. ``foreground`` therefore returns a
    score of 0.0 for the first ``warmup_frames`` calls (while still training
    the model) so a cold start can never be mistaken for a scene change.
    """

    def __init__(
        self, width: int = _MOTION_WIDTH, warmup_frames: int = 3
    ) -> None:
        self.width = width
        self.warmup_frames = max(1, warmup_frames)
        self.last_flow_magnitude = 0.0
        self._prev_gray: np.ndarray | None = None
        self._foreground_calls = 0
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=False
        )

    def reset(self) -> None:
        self._prev_gray = None
        self.last_flow_magnitude = 0.0
        self._foreground_calls = 0
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=False
        )

    def _small(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w <= self.width:
            return image
        scale = self.width / float(w)
        return cv2.resize(
            image, (self.width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def _gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self._small(image), cv2.COLOR_BGR2GRAY)

    def optical_flow(self, image: np.ndarray) -> tuple[float, np.ndarray | None]:
        gray = self._gray(image)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, None
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Mean per-pixel flow magnitude: scale-independent (unlike the L2
        # norm over all pixels), so a single threshold works across
        # resolutions.
        magnitude = float(np.mean(np.linalg.norm(flow, axis=2)))
        self.last_flow_magnitude = magnitude
        self._prev_gray = gray
        return magnitude, flow

    def foreground(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        mask = self._bg_subtractor.apply(self._small(image))
        self._foreground_calls += 1
        if self._foreground_calls <= self.warmup_frames:
            # Cold-start frames are not a scene change.
            return mask, 0.0
        change_score = float(np.count_nonzero(mask)) / float(mask.size)
        return mask, change_score
