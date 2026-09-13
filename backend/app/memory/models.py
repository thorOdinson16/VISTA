"""VISTA domain data models.

These are the canonical in-memory representations shared across the
perception, tracking, memory, retrieval and reasoning layers. Persistence
models (SQLAlchemy) will map to/from these.

Field names track the records specified in the requirements document
(section 7) and architecture document (section 5).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    """Extensible event vocabulary (FR-010)."""

    OBJECT_APPEARED = "object_appeared"
    OBJECT_DISAPPEARED = "object_disappeared"
    PERSON_ENTERED = "person_entered"
    PERSON_EXITED = "person_exited"
    OBJECT_PICKED_UP = "object_picked_up"
    OBJECT_PLACED = "object_placed"
    OBJECT_MOVED = "object_moved"
    OBJECT_CARRIED = "object_carried"
    OBJECT_APPROACHED = "object_approached"
    SCENE_CHANGED = "scene_changed"


class TrackStatus(str, Enum):
    ACTIVE = "active"
    OCCLUDED = "occluded"
    LOST = "lost"
    INACTIVE = "inactive"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )


class Detection(BaseModel):
    """A single-frame detection produced by the detector (FR-004)."""

    category: str
    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    frame_id: int
    timestamp: float


class Observation(BaseModel):
    """A single observation of an object (requirements section 7.2)."""

    observation_id: str
    object_id: str
    frame_id: int
    timestamp: float
    bounding_box: BoundingBox
    appearance_features: list[float] | None = None
    # Named visual attributes established at observation time, e.g. color.
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    association_confidence: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class TrajectoryPoint(BaseModel):
    """A sampled position of an object over time."""

    frame_id: int
    timestamp: float
    x: float
    y: float


class LocationRecord(BaseModel):
    """A position with an optional semantic region label."""

    frame_id: int
    timestamp: float
    x: float
    y: float
    region: str | None = None


class ObjectRecord(BaseModel):
    """Persistent object memory (FR-008, requirements section 7.1)."""

    object_id: str
    category: str
    appearance_embedding: list[float] | None = None
    # Named visual attributes (e.g. {"color": "red"}) with established values.
    # Never silently assumed: a missing key means "unknown".
    attributes: dict[str, Any] = Field(default_factory=dict)
    first_seen: float | None = None
    last_seen: float | None = None
    observations: list[Observation] = Field(default_factory=list)
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    location_history: list[LocationRecord] = Field(default_factory=list)
    event_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EventRecord(BaseModel):
    """A temporal event (FR-011, requirements section 7.3)."""

    event_id: str = Field(default="")
    type: EventType
    start_time: float
    end_time: float | None = None
    subject_refs: list[str] = Field(default_factory=list)
    actor_refs: list[str] = Field(default_factory=list)
    source_state: dict[str, Any] | None = None
    destination_state: dict[str, Any] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    # FR-022: distinguish directly observed facts from higher-level inference
    kind: str = Field(default="inferred")
    # Supporting detail for inferences (e.g. pickup hypothesis component
    # scores). None for plain observed events.
    details: dict[str, Any] | None = None


class EvidenceRecord(BaseModel):
    """A reference to source evidence (requirements section 7.4)."""

    evidence_id: str
    source_video: str
    frame_start: int
    frame_end: int | None = None
    timestamp_start: float
    timestamp_end: float | None = None
    object_refs: list[str] = Field(default_factory=list)
    event_refs: list[str] = Field(default_factory=list)


class SceneState(BaseModel):
    """A snapshot of the observed environment (FR-009, architecture 5.6)."""

    frame_id: int
    timestamp: float
    visible_objects: list[str] = Field(default_factory=list)
    object_locations: dict[str, LocationRecord] = Field(default_factory=dict)
    active_people: list[str] = Field(default_factory=list)
    spatial_relationships: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """A directed relationship between entities (FR-012)."""

    subject: str
    predicate: str
    object: str
    timestamp: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)


class Track(BaseModel):
    """A persistent trajectory under construction (architecture 5.4)."""

    track_id: str
    object_class: str
    observations: list[Observation] = Field(default_factory=list)
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None
    status: TrackStatus = TrackStatus.ACTIVE


__all__ = [
    "BoundingBox",
    "Detection",
    "EventRecord",
    "EventType",
    "EvidenceRecord",
    "LocationRecord",
    "ObjectRecord",
    "Observation",
    "Relationship",
    "SceneState",
    "Track",
    "TrackStatus",
    "TrajectoryPoint",
]
