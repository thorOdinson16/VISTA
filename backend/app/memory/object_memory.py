"""Persistent object memory: consolidate tracks into object records."""

from __future__ import annotations

from typing import Any

from ..perception.appearance import ema_update, majority_color
from .models import LocationRecord, ObjectRecord, Track


class ObjectMemory:
    def __init__(self) -> None:
        self._objects: dict[str, ObjectRecord] = {}

    def upsert(self, track: Track) -> ObjectRecord:
        obj = self._from_track(track)
        self._objects[track.track_id] = obj
        return obj

    @staticmethod
    def _from_track(track: Track) -> ObjectRecord:
        observations = sorted(track.observations, key=lambda o: o.frame_id)
        trajectory = sorted(track.trajectory, key=lambda p: p.frame_id)
        if observations:
            confidence = sum(o.confidence for o in observations) / len(observations)
        else:
            confidence = 0.0
        # Stable appearance descriptor: EMA over the observation history so
        # it reflects the object's appearance over time, not the last frame.
        embedding: list[float] | None = None
        for obs in observations:
            if obs.appearance_features:
                embedding = ema_update(embedding, obs.appearance_features)
        attributes: dict[str, Any] = {}
        color = majority_color([o.attributes.get("color") for o in observations])
        if color is not None:
            attributes["color"] = color
        location_history = [
            LocationRecord(
                frame_id=p.frame_id, timestamp=p.timestamp, x=p.x, y=p.y
            )
            for p in trajectory
        ]
        return ObjectRecord(
            object_id=track.track_id,
            category=track.object_class,
            appearance_embedding=embedding,
            attributes=attributes,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            observations=observations,
            trajectory=trajectory,
            location_history=location_history,
            confidence=confidence,
        )

    def get(self, object_id: str) -> ObjectRecord | None:
        return self._objects.get(object_id)

    def upsert_from_object(self, obj: ObjectRecord) -> None:
        self._objects[obj.object_id] = obj

    def all(self) -> list[ObjectRecord]:
        return list(self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects.values())
