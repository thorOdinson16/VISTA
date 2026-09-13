"""Scene memory: snapshots of the observed environment (FR-009)."""

from __future__ import annotations

import math

from .models import BoundingBox, LocationRecord, SceneState, Track


class SceneMemory:
    # Proximity (px) under which two visible entities count as NEAR.
    NEAR_DISTANCE = 120.0
    # IoU above which an object overlapping a person counts as held.
    HOLD_IOU = 0.2
    # Max center distance (px) for a held-object hypothesis.
    HOLD_DISTANCE = 50.0
    # Max vertical gap (px) between a person's top edge and an object
    # bottom edge (with horizontal overlap) for a held/carried hypothesis.
    HOLD_ABOVE_GAP = 80.0

    def __init__(self) -> None:
        self._states: list[SceneState] = []

    def snapshot(
        self, frame_id: int, timestamp: float, tracks: list[Track]
    ) -> SceneState:
        visible_objects: list[str] = []
        object_locations: dict[str, LocationRecord] = {}
        active_people: list[str] = []
        boxes: dict[str, BoundingBox | None] = {}

        for track in tracks:
            visible_objects.append(track.track_id)
            if track.trajectory:
                p = track.trajectory[-1]
                object_locations[track.track_id] = LocationRecord(
                    frame_id=p.frame_id, timestamp=p.timestamp, x=p.x, y=p.y
                )
            if track.object_class == "person":
                active_people.append(track.track_id)
            boxes[track.track_id] = self._last_bbox(track)

        state = SceneState(
            frame_id=frame_id,
            timestamp=timestamp,
            visible_objects=visible_objects,
            object_locations=object_locations,
            active_people=active_people,
            spatial_relationships=self._relationships(tracks, boxes),
        )
        self._states.append(state)
        return state

    @staticmethod
    def _last_bbox(track: Track) -> BoundingBox | None:
        for obs in reversed(track.observations):
            if obs.bounding_box is not None:
                return obs.bounding_box
        return None

    def _relationships(
        self, tracks: list[Track], boxes: dict[str, BoundingBox | None]
    ) -> list[str]:
        rels: list[str] = []
        for i, a in enumerate(tracks):
            for b in tracks[i + 1 :]:
                rel = self._pair_relationship(a, b, boxes)
                if rel is not None:
                    rels.append(rel)
        return rels

    def _pair_relationship(
        self, a: Track, b: Track, boxes: dict[str, BoundingBox | None]
    ) -> str | None:
        person, other = None, None
        if a.object_class == "person" and b.object_class != "person":
            person, other = a, b
        elif b.object_class == "person" and a.object_class != "person":
            person, other = b, a
        if person is not None and other is not None:
            return self._person_object(person, other, boxes)

        box_a, box_b = boxes.get(a.track_id), boxes.get(b.track_id)
        dist = _center_distance(a, b, box_a, box_b)
        if box_a is not None and box_b is not None:
            small, big = (a, b) if box_a.area <= box_b.area else (b, a)
            sbox = box_a if small is a else box_b
            lbox = box_b if small is a else box_a
            if _contains(lbox, sbox):
                return f"{small.track_id} ON {big.track_id}"
        if dist <= self.NEAR_DISTANCE:
            return f"{a.track_id} NEAR {b.track_id}"
        return None

    def _person_object(
        self, person: Track, other: Track, boxes: dict[str, BoundingBox | None]
    ) -> str | None:
        pbox, obox = boxes.get(person.track_id), boxes.get(other.track_id)
        if pbox is not None and obox is not None:
            if _iou(pbox, obox) >= self.HOLD_IOU or (
                _center_dist_boxes(pbox, obox) <= self.HOLD_DISTANCE
            ):
                return f"{other.track_id} IN_HAND_OF {person.track_id}"
            if _held_above(pbox, obox, self.HOLD_ABOVE_GAP):
                return f"{other.track_id} IN_HAND_OF {person.track_id}"
        dist = _center_distance(person, other, pbox, obox)
        if dist <= self.NEAR_DISTANCE:
            return f"{person.track_id} NEAR {other.track_id}"
        return None

    def all(self) -> list[SceneState]:
        return list(self._states)

    def extend(self, states: list[SceneState]) -> None:
        self._states.extend(states)

    def latest(self) -> SceneState | None:
        return self._states[-1] if self._states else None

    def at(self, timestamp: float) -> SceneState | None:
        best: SceneState | None = None
        for state in self._states:
            if state.timestamp <= timestamp:
                best = state
            else:
                break
        return best


def _center(track: Track, box: BoundingBox | None) -> tuple[float, float] | None:
    if box is not None:
        return box.center
    if track.trajectory:
        p = track.trajectory[-1]
        return (p.x, p.y)
    return None


def _center_distance(
    a: Track, b: Track, box_a: BoundingBox | None, box_b: BoundingBox | None
) -> float:
    ca, cb = _center(a, box_a), _center(b, box_b)
    if ca is None or cb is None:
        return math.inf
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def _center_dist_boxes(a: BoundingBox, b: BoundingBox) -> float:
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    cx, cy = inner.center
    return outer.x_min <= cx <= outer.x_max and outer.y_min <= cy <= outer.y_max


def _held_above(person: BoundingBox, obj: BoundingBox, max_gap: float) -> bool:
    """Object hovering just above the person's top edge with horizontal overlap."""
    horizontal_overlap = min(person.x_max, obj.x_max) - max(person.x_min, obj.x_min)
    if horizontal_overlap <= 0:
        return False
    vertical_gap = person.y_min - obj.y_max
    return 0 <= vertical_gap <= max_gap
