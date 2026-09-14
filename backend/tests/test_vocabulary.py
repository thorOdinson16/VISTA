"""Canonical vocabulary and detector label mapping."""

from __future__ import annotations

from backend.app.memory.models import BoundingBox, Detection
from backend.app.perception.vocabulary import (
    LabelMapper,
    MappedDetector,
    default_label_map,
    query_terms,
)


def _det(category: str) -> Detection:
    return Detection(
        category=category,
        bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
        confidence=0.9,
        frame_id=0,
        timestamp=0.0,
    )


class _StubDetector:
    name = "stub"

    def detect(self, image, frame_id, timestamp):
        return [_det("book"), _det("dining table"), _det("frisbee")]


def test_mapper_rewrites_known_labels():
    mapper = LabelMapper(default_label_map())
    assert mapper.map("book") == "notebook"
    assert mapper.map("dining table") == "desk"
    assert mapper.map("cup") == "mug"
    # Unknown labels pass through unchanged: never silently renamed.
    assert mapper.map("frisbee") == "frisbee"


def test_mapped_detector_rewrites_detections():
    detector = MappedDetector(_StubDetector(), LabelMapper(default_label_map()))
    categories = [d.category for d in detector.detect(None, 0, 0.0)]
    assert categories == ["notebook", "desk", "frisbee"]
    assert "labels" in detector.name


def test_mapper_can_drop_unmapped():
    mapper = LabelMapper({"book": "notebook"}, drop_unmapped=True)
    assert mapper.map("frisbee") is None
    mapped = mapper.map_detections([_det("book"), _det("frisbee")])
    assert [d.category for d in mapped] == ["notebook"]


def test_query_terms_map_synonyms_to_canonical():
    terms = dict(query_terms(default_label_map()))
    assert terms["book"] == "notebook"
    assert terms["dining table"] == "desk"
    assert terms["cup"] == "mug"
    assert terms["shelf"] == "shelf"


def test_query_terms_are_longest_first():
    terms = [term for term, _ in query_terms(default_label_map())]
    assert terms.index("cell phone") < terms.index("phone")
