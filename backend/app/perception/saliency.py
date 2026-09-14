"""Generic salient-region proposals for when no configured detector matches.

Real footage frequently contains objects the configured detector has no
class for (e.g. a coin against a COCO-only model). Rather than silently
producing an empty memory, the pipeline can fall back to the OpenCV MOG2
foreground mask already computed for motion analysis and propose
object-like regions as *unclassified* objects.

This fallback is deliberately conservative because MOG2 is a
**static-camera** background subtractor: on handheld footage it flags
residual background. Proposals are therefore required to be:

* inside the frame (not clinging to a border),
* plausibly shaped (minimum solidity, bounded aspect ratio),
* of sane size (min/max area), and
* persistent across at least two sampled frames (see ``SaliencyConfirmer``).

It is disabled by default (``VISTA_SALIENCY_FALLBACK=false``); when off,
unmatched footage yields an honest "no configured objects detected" rather
than a fabricated object.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..memory.models import BoundingBox, Detection

UNCLASSIFIED_CATEGORY = "object"


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class SalientObjectDetector:
    def __init__(
        self,
        min_area: int = 600,
        max_objects: int = 8,
        max_area_fraction: float = 0.8,
        min_solidity: float = 0.35,
        max_aspect: float = 4.0,
        border_margin: float = 0.03,
        category: str = UNCLASSIFIED_CATEGORY,
        base_confidence: float = 0.3,
        max_confidence: float = 0.6,
    ) -> None:
        self.min_area = min_area
        self.max_objects = max_objects
        self.max_area_fraction = max_area_fraction
        self.min_solidity = min_solidity
        self.max_aspect = max_aspect
        self.border_margin = border_margin
        self.category = category
        self.base_confidence = base_confidence
        self.max_confidence = max_confidence

    def _is_border_touching(
        self, x: float, y: float, w: float, h: float, fw: int, fh: int
    ) -> bool:
        mx = self.border_margin * fw
        my = self.border_margin * fh
        return x <= mx or y <= my or (x + w) >= fw - mx or (y + h) >= fh - my

    def propose(
        self,
        mask: np.ndarray | None,
        frame_id: int,
        timestamp: float,
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        """Propose unclassified regions from a foreground mask.

        ``frame_shape`` is the source ``(height, width)``; mask coordinates
        are scaled back up so boxes are in source-frame pixels.
        """
        if mask is None or mask.size == 0:
            return []
        mh, mw = mask.shape[:2]
        fh, fw = frame_shape[:2]
        if mh == 0 or mw == 0 or fh == 0 or fw == 0:
            return []
        sx = fw / float(mw)
        sy = fh / float(mh)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = float(fh * fw)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = cv2.contourArea(contour) * sx * sy
            if area < self.min_area:
                continue
            if area > self.max_area_fraction * frame_area:
                continue
            # Shape plausibility. Solidity is scale-invariant; aspect uses
            # source pixels.
            bbox_area = float(w * h)
            solidity = (area / (sx * sy)) / bbox_area if bbox_area > 0 else 0.0
            if solidity < self.min_solidity:
                continue
            aspect = max(w, h) / float(min(w, h))
            if aspect > self.max_aspect:
                continue
            if self._is_border_touching(x, y, w, h, mw, mh):
                continue
            candidates.append((area, x, y, w, h))
        candidates.sort(key=lambda c: c[0], reverse=True)

        detections: list[Detection] = []
        for area, x, y, w, h in candidates[: self.max_objects]:
            fraction = min(1.0, area / frame_area)
            confidence = min(
                self.max_confidence,
                self.base_confidence + (self.max_confidence - self.base_confidence)
                * fraction
                * 5.0,
            )
            detections.append(
                Detection(
                    category=self.category,
                    bounding_box=BoundingBox(
                        x_min=x * sx,
                        y_min=y * sy,
                        x_max=(x + w) * sx,
                        y_max=(y + h) * sy,
                    ),
                    confidence=confidence,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
        return detections


class SaliencyConfirmer:
    """Require a saliency region to persist across two sampled frames.

    A single-frame MOG2 blob is usually noise. A proposal is only forwarded
    to the tracker when it overlaps (IoU) a proposal from the previous
    sampled frame; the first observation is held back as the candidate.
    """

    def __init__(self, min_overlap: float = 0.1) -> None:
        self.min_overlap = min_overlap
        self._previous: list[Detection] = []

    def reset(self) -> None:
        self._previous = []

    def confirm(self, proposals: list[Detection]) -> list[Detection]:
        confirmed = []
        for proposal in proposals:
            if any(
                _iou(proposal.bounding_box, prev.bounding_box) >= self.min_overlap
                for prev in self._previous
            ):
                confirmed.append(proposal)
        self._previous = list(proposals)
        return confirmed
