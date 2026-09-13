"""Appearance representation for persistent object identity.

Combines three OpenCV-based cues into one profile:

1. **Named color attribute** -- dominant HSV hue mapped to a small
   vocabulary (``red``, ``blue``, ...). Used for attribute matching
   (e.g. "the red notebook") and stored on observations/objects.
   A missing/``None`` color means "unknown" -- never silently assumed.
2. **Color-histogram descriptor** -- normalized HSV 2-D histogram,
   flattened into a fixed-length float vector. Stored per observation
   (``appearance_features``) and averaged into the object's stable
   ``appearance_embedding`` via EMA so it reflects the object's history,
   not just the latest frame.
3. **Local ORB descriptors** -- for textured objects, matched with a
   ratio-test score. This is the cue that lets the system distinguish
   two same-category/same-color objects (``red notebook A`` vs
   ``red notebook B``) *when visual evidence allows it*.

Known limitation (documented, not hidden): two objects that are pixel-
identical (e.g. two solid-color synthetic rectangles of the same hue)
carry no discriminative visual evidence. In that case the local-feature
score is ~0 and the similarity falls back to color + spatiotemporal
cues, with the association confidence reflecting the ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import FeatureExtractor

HIST_BINS = (32, 32)
ORB_FEATURES = 500
# Minimum fraction of agreeing observation color labels required before the
# object-level attribute is set. Below this the color stays unknown.
COLOR_SUPPORT_FRACTION = 0.6
# Minimum mean saturation for a hue to be trustworthy.
MIN_MEAN_SATURATION = 30.0


@dataclass
class AppearanceProfile:
    """Appearance of one object crop."""

    # Fixed-length float descriptor (flattened normalized HSV histogram).
    descriptor: np.ndarray | None = None
    # Named color label, or None when it cannot be established confidently.
    color_label: str | None = None
    # ORB descriptors for local-feature matching (None when unavailable).
    local_descriptors: np.ndarray | None = None
    # Extra named attributes established at encode time.
    attributes: dict = field(default_factory=dict)


def encode_crop(crop: np.ndarray | None) -> AppearanceProfile:
    """Encode an object image crop into an AppearanceProfile."""
    if crop is None or crop.size == 0:
        return AppearanceProfile()
    if crop.ndim != 3 or crop.shape[2] != 3:
        return AppearanceProfile()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    descriptor = np.asarray(hist, dtype=np.float32).flatten()

    color_label = dominant_color_label(hsv)

    local_descriptors: np.ndarray | None = None
    try:
        _, local_descriptors = FeatureExtractor(
            method="orb", max_features=ORB_FEATURES
        ).extract(crop)
    except cv2.error:
        local_descriptors = None

    attributes: dict = {}
    if color_label is not None:
        attributes["color"] = color_label
    return AppearanceProfile(
        descriptor=descriptor,
        color_label=color_label,
        local_descriptors=local_descriptors,
        attributes=attributes,
    )


def dominant_color_label(hsv: np.ndarray) -> str | None:
    """Map dominant HSV content to a named color, or None if not confident."""
    if hsv.size == 0:
        return None
    h_full = hsv.shape[0]
    w_full = hsv.shape[1]
    # Inset from the borders: bounding boxes include background and codec
    # edge artifacts; the interior carries the trustworthy hue.
    inset_h, inset_w = max(1, h_full // 8), max(1, w_full // 8)
    inner = hsv[inset_h : h_full - inset_h or None, inset_w : w_full - inset_w or None]
    if inner.size == 0:
        inner = hsv
    h = inner[:, :, 0].astype(np.float32)
    s = inner[:, :, 1].astype(np.float32)
    v = inner[:, :, 2].astype(np.float32)
    mean_s = float(s.mean())
    mean_v = float(v.mean())

    # Achromatic cases first (low saturation carries no hue information).
    if mean_s < MIN_MEAN_SATURATION:
        if mean_v < 50:
            return "black"
        if mean_v > 200:
            return "white"
        return "gray"

    # Saturation-weighted circular mean of hue (hue is circular, so a
    # plain average breaks at the red wrap-around; binning quantizes
    # boundaries like 120 (blue) into the next bucket).
    mask = s >= MIN_MEAN_SATURATION
    hues = h[mask]
    weights = s[mask]
    if hues.size == 0 or weights.sum() <= 0:
        return None
    radians = np.deg2rad(hues.astype(np.float64) * 2.0)
    w = weights.astype(np.float64)
    sin_sum = float(np.sum(w * np.sin(radians)))
    cos_sum = float(np.sum(w * np.cos(radians)))
    resultant = float(np.hypot(sin_sum, cos_sum)) / float(w.sum())
    if resultant < 0.25:
        # No dominant hue -- do not guess.
        return None
    mean_hue = (float(np.rad2deg(np.arctan2(sin_sum, cos_sum))) / 2.0) % 180.0
    return _hue_to_name(mean_hue)


def _hue_to_name(hue: float) -> str:
    if hue < 10 or hue >= 160:
        return "red"
    if hue < 20:
        return "orange"
    if hue < 33:
        return "yellow"
    if hue < 78:
        return "green"
    if hue < 95:
        return "cyan"
    if hue < 125:
        return "blue"
    if hue < 145:
        return "purple"
    return "magenta"


def histogram_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Correlation similarity in [0, 1]; 0.5 when either side is missing."""
    if a is None or b is None:
        return 0.5
    try:
        ha = np.asarray(a, dtype=np.float32).reshape(-1)
        hb = np.asarray(b, dtype=np.float32).reshape(-1)
        if ha.shape != hb.shape:
            return 0.5
        ha = ha.reshape(HIST_BINS)
        hb = hb.reshape(HIST_BINS)
        return max(0.0, float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))
    except cv2.error:
        return 0.5


def local_feature_similarity(
    a: np.ndarray | None,
    b: np.ndarray | None,
    ratio: float = 0.75,
) -> float:
    """Lowe-ratio-test match score in [0, 1].

    Returns 0.0 when either side has no descriptors (no evidence of visual
    similarity), which is the honest signal for textureless/synthetic crops.
    """
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    try:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(a, b, k=2)
    except cv2.error:
        return 0.0
    good = 0
    total = 0
    for m_n in pairs:
        if len(m_n) != 2:
            continue
        total += 1
        m, n = m_n
        if m.distance < ratio * n.distance:
            good += 1
    if total == 0:
        return 0.0
    return good / total


def profile_similarity(a: AppearanceProfile, b: AppearanceProfile) -> dict:
    """Combined appearance similarity with per-cue breakdown.

    Returns {"score", "histogram", "local", "color_match"} so callers can
    see *why* two observations were judged similar instead of trusting a
    single opaque number.
    """
    hist = histogram_similarity(a.descriptor, b.descriptor)
    local = local_feature_similarity(a.local_descriptors, b.local_descriptors)
    if a.color_label is None or b.color_label is None:
        color_match = 0.5
    else:
        color_match = 1.0 if a.color_label == b.color_label else 0.0
    # Local features only count when they exist; otherwise fall back to
    # color-histogram evidence with the ambiguity made explicit by the
    # "local" component being 0.
    local_weight = 0.35 if local > 0.0 else 0.0
    hist_weight = 0.65 if local > 0.0 else 1.0
    score = hist_weight * hist + local_weight * local
    return {
        "score": float(max(0.0, min(1.0, score))),
        "histogram": float(hist),
        "local": float(local),
        "color_match": float(color_match),
    }


def ema_update(
    current: list[float] | np.ndarray | None,
    sample: np.ndarray | None,
    alpha: float = 0.3,
) -> list[float] | None:
    """Update a stable object descriptor with a new observation (EMA).

    ``alpha`` controls how much the newest observation moves the running
    descriptor; history is never discarded outright.
    """
    if sample is None:
        if current is None:
            return None
        return list(np.asarray(current, dtype=np.float32))
    sample = np.asarray(sample, dtype=np.float32).flatten()
    if current is None:
        return sample.tolist()
    current_arr = np.asarray(current, dtype=np.float32).flatten()
    if current_arr.shape != sample.shape:
        return sample.tolist()
    updated = (1.0 - alpha) * current_arr + alpha * sample
    norm = float(np.linalg.norm(updated))
    if norm > 0:
        updated = updated / norm
    return updated.tolist()


def majority_color(colors: list[str | None]) -> str | None:
    """Object-level color: majority vote with minimum support fraction."""
    votes = [c for c in colors if c is not None]
    if not votes:
        return None
    top = max(set(votes), key=votes.count)
    if votes.count(top) / len(colors) < COLOR_SUPPORT_FRACTION:
        return None
    return top
