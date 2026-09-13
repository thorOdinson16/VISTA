"""Local feature extraction and matching (ORB / SIFT) via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class FeatureExtractor:
    """Extract and match local features for tracking and re-identification."""

    def __init__(self, method: str = "orb", max_features: int = 500) -> None:
        self.method = method
        if method == "sift":
            self._detector = cv2.SIFT_create(nfeatures=max_features)
            self._norm = cv2.NORM_L2
        else:
            self._detector = cv2.ORB_create(nfeatures=max_features)
            self._norm = cv2.NORM_HAMMING

    def extract(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match(
        self, descriptors_a: np.ndarray, descriptors_b: np.ndarray
    ) -> list[cv2.DMatch]:
        if descriptors_a is None or descriptors_b is None:
            return []
        if len(descriptors_a) == 0 or len(descriptors_b) == 0:
            return []
        matcher = cv2.BFMatcher(self._norm, crossCheck=True)
        matches = matcher.match(descriptors_a, descriptors_b)
        return sorted(matches, key=lambda m: m.distance)
