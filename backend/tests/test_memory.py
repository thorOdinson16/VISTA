from __future__ import annotations

from backend.app.memory.event_memory import EventExtractor, EventMemory
from backend.app.memory.graph import RelationshipGraph
from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import (
    EventRecord,
    EventType,
    ObjectRecord,
    Track,
    TrackStatus,
    TrajectoryPoint,
)
from backend.app.memory.object_memory import ObjectMemory
from backend.app.memory.scene_memory import SceneMemory


def make_track(
    track_id: str,
    category: str,
    points: list[tuple[float, float, float]],
    confidence: float = 0.9,
) -> Track:
    return Track(
        track_id=track_id,
        object_class=category,
        trajectory=[
            TrajectoryPoint(frame_id=i, timestamp=t, x=x, y=y)
            for i, (t, x, y) in enumerate(points)
        ],
        first_seen=points[0][0],
        last_seen=points[-1][0],
        observations=[],
        status=TrackStatus.ACTIVE,
    )


def make_object(
    object_id: str, category: str, points: list[tuple[float, float, float]]
) -> ObjectRecord:
    return ObjectRecord(
        object_id=object_id,
        category=category,
        first_seen=points[0][0],
        last_seen=points[-1][0],
        trajectory=[
            TrajectoryPoint(frame_id=i, timestamp=t, x=x, y=y)
            for i, (t, x, y) in enumerate(points)
        ],
        confidence=0.9,
    )


def test_object_memory_upsert():
    memory = ObjectMemory()
    track = make_track("obj_0", "notebook", [(0.0, 10, 10), (1.0, 20, 20)])
    obj = memory.upsert(track)
    assert obj.object_id == "obj_0"
    assert obj.category == "notebook"
    assert obj.first_seen == 0.0
    assert len(obj.trajectory) == 2
    assert len(obj.location_history) == 2
    assert len(memory) == 1


def test_scene_memory_snapshot():
    memory = SceneMemory()
    person = make_track("p1", "person", [(0.0, 0, 0)])
    notebook = make_track("n1", "notebook", [(0.0, 10, 10)])
    state = memory.snapshot(0, 0.0, [person, notebook])
    assert set(state.visible_objects) == {"p1", "n1"}
    assert state.active_people == ["p1"]
    assert "n1" in state.object_locations


def test_graph_queries():
    graph = RelationshipGraph()
    graph.add("p1", "picked_up", "n1", timestamp=5.0, confidence=0.9)
    assert graph.has_edge("p1", "picked_up", "n1")
    assert graph.by_subject("p1")[0].predicate == "picked_up"
    assert graph.by_object("n1")[0].subject == "p1"


def test_event_memory_queries():
    memory = EventMemory()
    memory.add(
        EventRecord(
            type=EventType.OBJECT_PICKED_UP,
            start_time=5.0,
            subject_refs=["n1"],
            actor_refs=["p1"],
        )
    )
    assert memory.by_type(EventType.OBJECT_PICKED_UP)[0].event_id == "evt_0"
    assert memory.by_subject("n1")[0].type is EventType.OBJECT_PICKED_UP
    assert memory.by_actor("p1")[0].event_id == "evt_0"


def test_extractor_appearance_and_movement():
    extractor = EventExtractor(video_duration=100.0)
    obj = make_object("n1", "notebook", [(0.0, 0, 0), (10.0, 200, 0)])
    person = make_object("p1", "person", [(0.0, 500, 500), (5.0, 500, 500)])
    events = extractor.extract([obj, person])
    types = {e.type for e in events}
    assert EventType.OBJECT_APPEARED in types
    assert EventType.OBJECT_MOVED in types
    assert EventType.PERSON_ENTERED in types
    assert EventType.PERSON_EXITED in types


def test_extractor_interactions():
    extractor = EventExtractor(video_duration=100.0)
    person = make_object(
        "p1", "person", [(0.0, 100, 0), (4.0, 0, 0), (10.0, 100, 0)]
    )
    notebook = make_object(
        "n1", "notebook", [(0.0, 0, 0), (4.0, 0, 0), (10.0, 100, 0)]
    )
    events = extractor.extract([person, notebook])
    picked = [e for e in events if e.type is EventType.OBJECT_PICKED_UP]
    carried = [e for e in events if e.type is EventType.OBJECT_CARRIED]
    placed = [e for e in events if e.type is EventType.OBJECT_PLACED]
    assert picked and picked[0].actor_refs == ["p1"]
    assert carried and carried[0].subject_refs == ["n1"]
    assert placed


def test_visual_memory_finalize_and_save_load(tmp_path):
    memory = VisualMemory(video_duration=100.0, store_path=tmp_path / "mem.db")
    person = make_track(
        "p1", "person", [(0.0, 100, 0), (4.0, 0, 0), (10.0, 100, 0)]
    )
    notebook = make_track(
        "n1", "notebook", [(0.0, 0, 0), (4.0, 0, 0), (10.0, 100, 0)]
    )
    memory.update(0, 0.0, [person, notebook])
    memory.update(4, 4.0, [person, notebook])
    memory.update(10, 10.0, [person, notebook])
    events = memory.finalize()

    assert any(e.type is EventType.OBJECT_PICKED_UP for e in events)
    assert memory.graph.has_edge("p1", "picked_up", "n1")

    memory.save()

    loaded = VisualMemory.load(tmp_path / "mem.db")
    assert len(loaded.object_memory) == 2
    assert loaded.graph.has_edge("p1", "picked_up", "n1")
    assert any(e.type is EventType.OBJECT_PICKED_UP for e in loaded.event_memory.all())


def test_object_appearance_embedding_and_attributes():
    from backend.app.memory.models import BoundingBox, Observation

    observations = []
    for i in range(4):
        observations.append(
            Observation(
                observation_id=f"n1:{i}",
                object_id="n1",
                frame_id=i,
                timestamp=float(i),
                bounding_box=BoundingBox(
                    x_min=0, y_min=0, x_max=40, y_max=40
                ),
                appearance_features=[0.1 * i + 0.5, 0.2, 0.3],
                attributes={"color": "red"},
                confidence=0.9,
            )
        )
    track = Track(
        track_id="n1",
        object_class="notebook",
        observations=observations,
        trajectory=[
            TrajectoryPoint(frame_id=i, timestamp=float(i), x=10.0, y=10.0)
            for i in range(4)
        ],
        first_seen=0.0,
        last_seen=3.0,
        status=TrackStatus.ACTIVE,
    )
    memory = ObjectMemory()
    obj = memory.upsert(track)
    # Stable descriptor aggregated over history, not just the last frame.
    assert obj.appearance_embedding
    assert len(obj.appearance_embedding) == 3
    assert obj.appearance_embedding != [0.8, 0.2, 0.3]
    # Named attribute established by majority vote.
    assert obj.attributes.get("color") == "red"


def test_scene_snapshot_spatial_relationships():
    from backend.app.memory.models import BoundingBox, Observation

    def tracked(tid, cat, cx, cy, w=40.0, h=40.0):
        return Track(
            track_id=tid,
            object_class=cat,
            observations=[
                Observation(
                    observation_id=f"{tid}:0",
                    object_id=tid,
                    frame_id=0,
                    timestamp=0.0,
                    bounding_box=BoundingBox(
                        x_min=cx - w / 2, y_min=cy - h / 2,
                        x_max=cx + w / 2, y_max=cy + h / 2,
                    ),
                    confidence=0.9,
                )
            ],
            trajectory=[TrajectoryPoint(frame_id=0, timestamp=0.0, x=cx, y=cy)],
            first_seen=0.0,
            last_seen=0.0,
            status=TrackStatus.ACTIVE,
        )

    memory = SceneMemory()
    state = memory.snapshot(
        0,
        0.0,
        [
            tracked("p1", "person", 0, 0),
            tracked("n1", "notebook", 10, 10),  # overlapping person -> held
            tracked("m1", "mug", 300, 300),  # far away -> no relation
        ],
    )
    assert "n1 IN_HAND_OF p1" in state.spatial_relationships
    assert not any("m1" in r for r in state.spatial_relationships)


def test_scene_on_and_near_relationships():
    from backend.app.memory.models import BoundingBox, Observation

    def tracked(tid, cat, cx, cy, w=40.0, h=40.0):
        return Track(
            track_id=tid,
            object_class=cat,
            observations=[
                Observation(
                    observation_id=f"{tid}:0",
                    object_id=tid,
                    frame_id=0,
                    timestamp=0.0,
                    bounding_box=BoundingBox(
                        x_min=cx - w / 2, y_min=cy - h / 2,
                        x_max=cx + w / 2, y_max=cy + h / 2,
                    ),
                    confidence=0.9,
                )
            ],
            trajectory=[TrajectoryPoint(frame_id=0, timestamp=0.0, x=cx, y=cy)],
            first_seen=0.0,
            last_seen=0.0,
            status=TrackStatus.ACTIVE,
        )

    memory = SceneMemory()
    state = memory.snapshot(
        0, 0.0, [tracked("desk", "desk", 200, 200, w=200, h=100),
                 tracked("n1", "notebook", 200, 200)]
    )
    assert "n1 ON desk" in state.spatial_relationships


def test_persistence_round_trip_scenes_and_counters(tmp_path):
    memory = VisualMemory(video_duration=100.0, store_path=tmp_path / "mem.db")
    person = make_track(
        "p1", "person", [(0.0, 100, 0), (4.0, 0, 0), (10.0, 100, 0)]
    )
    notebook = make_track(
        "n1", "notebook", [(0.0, 0, 0), (4.0, 0, 0), (10.0, 100, 0)]
    )
    memory.update(0, 0.0, [person, notebook])
    memory.finalize()
    assert memory.scene_memory.all()
    assert memory.scene_memory.latest().spatial_relationships
    memory.save()

    loaded = VisualMemory.load(tmp_path / "mem.db")
    assert len(loaded.scene_memory.all()) == len(memory.scene_memory.all())
    assert loaded.scene_memory.latest().spatial_relationships
    assert len(loaded.evidence_memory) == len(memory.evidence_memory)
    # Event IDs keep increasing after load (no collisions).
    before = {e.event_id for e in loaded.event_memory.all()}
    loaded.event_memory.add(
        EventRecord(type=EventType.OBJECT_MOVED, start_time=99.0)
    )
    new_ids = {e.event_id for e in loaded.event_memory.all()} - before
    assert len(new_ids) == 1 and not (new_ids & before)
