"""Canonical object vocabulary and detector-label mapping.

There are three vocabularies in VISTA: the detector's raw labels, the
categories stored in memory, and the words users type. Historically these
disagreed (COCO says ``book``/``dining table``; the demo/query vocabulary
says ``notebook``/``desk``), so detections were invisible to queries.

This module defines one canonical vocabulary plus:

* ``CANONICAL_SYNONYMS`` -- user-facing synonyms per canonical category,
  used to build the query-parser vocabulary.
* ``COCO_TO_CANONICAL`` -- raw detector label -> canonical category
  (unmapped labels pass through unchanged, so an unknown detection is never
  silently renamed).
* ``LabelMapper`` / ``MappedDetector`` -- rewrite detector output to the
  canonical vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..memory.models import Detection

CANONICAL_SYNONYMS: dict[str, list[str]] = {
    "person": ["person", "man", "woman", "people", "someone", "child"],
    "notebook": ["notebook", "book", "notepad", "journal"],
    "mug": ["mug", "cup", "coffee cup"],
    "backpack": ["backpack", "bag", "handbag", "rucksack"],
    "laptop": ["laptop", "computer"],
    "phone": ["phone", "cell phone", "mobile", "smartphone"],
    "chair": ["chair", "seat"],
    "bottle": ["bottle"],
    "tool": ["tool", "screwdriver", "wrench", "hammer"],
    "desk": ["desk", "table", "dining table", "workbench"],
    "shelf": ["shelf", "bookshelf", "bookcase"],
    "coin": ["coin"],
    "tablet": ["tablet"],
}

# Raw detector labels that should be stored under a canonical category.
# Anything not listed here passes through unchanged.
COCO_TO_CANONICAL: dict[str, str] = {
    "person": "person",
    "book": "notebook",
    "dining table": "desk",
    "cup": "mug",
    "cell phone": "phone",
    "handbag": "backpack",
    "backpack": "backpack",
    "laptop": "laptop",
    "chair": "chair",
    "bottle": "bottle",
}


def default_label_map() -> dict[str, str]:
    return dict(COCO_TO_CANONICAL)


def load_label_map(path: str | Path | None) -> dict[str, str]:
    """Load a raw->canonical JSON map, falling back to the built-in map."""
    if not path or not Path(path).exists():
        return default_label_map()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k).lower(): str(v) for k, v in data.items()}


def query_terms(
    mapping: dict[str, str] | None = None,
    extra_labels: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(term, canonical)`` pairs, longest terms first.

    Includes canonical categories, their synonyms, and every raw label in
    the detector mapping (so asking for "book" or "dining table" resolves
    to the canonical objects). ``extra_labels`` are configured open-vocab
    labels (e.g. ``wallet``) that map to themselves, so anything the
    detector can find is also something a query can ask for.
    """
    terms: dict[str, str] = {}
    for canonical, synonyms in CANONICAL_SYNONYMS.items():
        terms[canonical] = canonical
        for synonym in synonyms:
            terms[synonym] = canonical
    for raw, canonical in (mapping or {}).items():
        terms.setdefault(raw, canonical)
    for label in extra_labels or []:
        key = label.strip().lower()
        if key:
            terms.setdefault(key, key)
    return sorted(terms.items(), key=lambda pair: -len(pair[0]))


class LabelMapper:
    def __init__(
        self, mapping: dict[str, str] | None = None, drop_unmapped: bool = False
    ) -> None:
        self.mapping = {
            str(k).lower(): str(v) for k, v in (mapping or {}).items()
        }
        self.drop_unmapped = drop_unmapped

    def map(self, label: str) -> str | None:
        key = label.lower()
        if key in self.mapping:
            return self.mapping[key]
        return None if self.drop_unmapped else label

    def map_detections(self, detections: list[Detection]) -> list[Detection]:
        out: list[Detection] = []
        for detection in detections:
            category = self.map(detection.category)
            if category is None:
                continue
            if category != detection.category:
                detection = detection.model_copy(update={"category": category})
            out.append(detection)
        return out


class MappedDetector:
    """Wrap a detector so its output uses the canonical vocabulary."""

    def __init__(self, detector, mapper: LabelMapper) -> None:
        self.detector = detector
        self.mapper = mapper
        self.name = f"{getattr(detector, 'name', type(detector).__name__)} +labels"

    def detect(self, image, frame_id, timestamp):
        return self.mapper.map_detections(
            self.detector.detect(image, frame_id, timestamp)
        )
