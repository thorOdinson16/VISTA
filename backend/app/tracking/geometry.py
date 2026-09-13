"""Geometric helpers for tracking association."""

from __future__ import annotations

import math

from ..memory.models import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: BoundingBox, b: BoundingBox) -> float:
    (ax, ay) = a.center
    (bx, by) = b.center
    return math.hypot(ax - bx, ay - by)
