"""Re-identification of objects after track interruption (FR-007).

Combines appearance similarity (color histogram + named color + local ORB
features), spatial continuity, and temporal constraints into a single
association score with confidence and a per-cue breakdown.

A high score from color alone is *not* claimed as a successful
re-identification: the breakdown exposes how much of the score came from
local visual features versus color/spatial/temporal priors.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..memory.models import BoundingBox
from ..perception.appearance import (
    AppearanceProfile,
    histogram_similarity,
    profile_similarity,
)


def crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, int(bbox.x_min))
    y1 = max(0, int(bbox.y_min))
    x2 = min(w, int(bbox.x_max))
    y2 = min(h, int(bbox.y_max))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), np.uint8)
    return image[y1:y2, x1:x2]


def color_histogram(image_crop: np.ndarray | None) -> np.ndarray | None:
    if image_crop is None or image_crop.size == 0:
        return None
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


class ReIdentifier:
    """Score whether a new observation is a previously-lost object."""

    def __init__(
        self,
        threshold: float = 0.6,
        w_appearance: float = 0.5,
        w_spatial: float = 0.3,
        w_temporal: float = 0.2,
        spatial_scale: float = 200.0,
        temporal_scale: float = 10.0,
    ) -> None:
        self.threshold = threshold
        self.w_appearance = w_appearance
        self.w_spatial = w_spatial
        self.w_temporal = w_temporal
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale

    def match_score(
        self,
        appearance_a: np.ndarray | AppearanceProfile | None,
        appearance_b: np.ndarray | AppearanceProfile | None,
        last_bbox: BoundingBox,
        first_bbox: BoundingBox,
        gap_seconds: float,
    ) -> float:
        score, _ = self.match_with_breakdown(
            appearance_a, appearance_b, last_bbox, first_bbox, gap_seconds
        )
        return score

    def match_with_breakdown(
        self,
        appearance_a: np.ndarray | AppearanceProfile | None,
        appearance_b: np.ndarray | AppearanceProfile | None,
        last_bbox: BoundingBox,
        first_bbox: BoundingBox,
        gap_seconds: float,
    ) -> tuple[float, dict[str, Any]]:
        """Score a re-identification hypothesis with per-cue breakdown."""
        prof_a = _as_profile(appearance_a)
        prof_b = _as_profile(appearance_b)
        if prof_a is not None and prof_b is not None:
            sim = profile_similarity(prof_a, prof_b)
            app = sim["score"]
            appearance_detail: dict[str, Any] = dict(sim)
        else:
            app = self._appearance_similarity(
                prof_a.descriptor if prof_a else None,
                prof_b.descriptor if prof_b else None,
            )
            appearance_detail = {"score": app, "histogram": app, "local": 0.0}
        spatial = math.exp(-self._distance(last_bbox, first_bbox) / self.spatial_scale)
        temporal = math.exp(-max(0.0, gap_seconds) / self.temporal_scale)
        score = (
            self.w_appearance * app
            + self.w_spatial * spatial
            + self.w_temporal * temporal
        )
        return float(score), {
            "score": float(score),
            "appearance": appearance_detail,
            "spatial": float(spatial),
            "temporal": float(temporal),
        }

    @staticmethod
    def _distance(a: BoundingBox, b: BoundingBox) -> float:
        (ax, ay) = a.center
        (bx, by) = b.center
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def _appearance_similarity(
        a: np.ndarray | None, b: np.ndarray | None
    ) -> float:
        if a is None or b is None:
            return 0.5
        try:
            ha = np.asarray(a, dtype=np.float32).reshape(32, 32)
            hb = np.asarray(b, dtype=np.float32).reshape(32, 32)
        except ValueError:
            return 0.5
        return histogram_similarity(ha, hb)


def _as_profile(
    value: np.ndarray | AppearanceProfile | None,
) -> AppearanceProfile | None:
    if value is None:
        return None
    if isinstance(value, AppearanceProfile):
        return value
    arr = np.asarray(value, dtype=np.float32)
    return AppearanceProfile(descriptor=arr)
