"""Event extraction: observed vs inferred, movement segments, pickup gating."""

from __future__ import annotations

from backend.app.memory.event_memory import EventExtractor
from backend.app.memory.models import EventType, ObjectRecord, TrajectoryPoint


def make_object(object_id, category, points, confidence=0.9):
    return ObjectRecord(
        object_id=object_id,
        category=category,
        first_seen=points[0][0],
        last_seen=points[-1][0],
        trajectory=[
            TrajectoryPoint(frame_id=i, timestamp=t, x=x, y=y)
            for i, (t, x, y) in enumerate(points)
        ],
        confidence=confidence,
    )


def dense(obj_id, cat, start, end, pos, step=0.5):
    pts = []
    t = start
    while t <= end + 1e-9:
        pts.append((round(t, 3), pos[0], pos[1]))
        t += step
    return make_object(obj_id, cat, pts)


def moving(obj_id, cat, segments, step=0.5):
    """segments: list of (t0, t1, x0, y0, x1, y1) linearly interpolated."""
    pts = []
    for t0, t1, x0, y0, x1, y1 in segments:
        t = t0
        while t <= t1 + 1e-9:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            pts.append((round(t, 3), x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
            t += step
    return make_object(obj_id, cat, pts)


def test_appearance_events_are_observed():
    ex = EventExtractor(video_duration=100.0)
    obj = dense("n1", "notebook", 0.0, 5.0, (10, 10))
    events = ex.extract([obj])
    appeared = [e for e in events if e.type is EventType.OBJECT_APPEARED]
    disappeared = [e for e in events if e.type is EventType.OBJECT_DISAPPEARED]
    assert appeared and appeared[0].kind == "observed"
    assert disappeared and disappeared[0].kind == "observed"


def test_movement_segmented_stationary_moving_stationary():
    ex = EventExtractor(video_duration=100.0)
    obj = moving(
        "n1",
        "notebook",
        [
            (0.0, 4.0, 0, 0, 0, 0),      # stationary
            (4.0, 8.0, 0, 0, 200, 0),    # moving
            (8.0, 12.0, 200, 0, 200, 0),  # stationary
        ],
    )
    events = [e for e in ex.extract([obj]) if e.type is EventType.OBJECT_MOVED]
    assert len(events) == 1
    seg = events[0]
    assert seg.start_time >= 3.5
    assert seg.end_time is not None and seg.end_time <= 8.5
    assert seg.source_state["x"] < 50
    assert seg.destination_state["x"] > 150


def test_movement_jitter_produces_no_event():
    import math

    pts = [
        (float(i), 10.0 * math.sin(i), 5.0 * math.cos(i)) for i in range(20)
    ]
    obj = make_object("n1", "notebook", pts)
    ex = EventExtractor(video_duration=100.0)
    moved = [e for e in ex.extract([obj]) if e.type is EventType.OBJECT_MOVED]
    assert moved == []


def test_pickup_carry_place_inferred_with_details():
    ex = EventExtractor(video_duration=100.0)
    person = moving(
        "p1",
        "person",
        [
            (0.0, 2.0, 120, 0, 60, 0),   # approaching
            (2.0, 3.0, 60, 0, 10, 0),    # arriving
            (3.0, 8.0, 10, 0, 110, 0),   # carrying (joint motion)
            (8.0, 10.0, 110, 0, 160, 0),  # leaving
        ],
    )
    notebook = moving(
        "n1",
        "notebook",
        [
            (0.0, 3.0, 0, 0, 0, 0),      # stationary at origin
            (3.0, 8.0, 0, 0, 100, 0),    # moving with the person
            (8.0, 10.0, 100, 0, 100, 0),  # stationary after placement
        ],
    )
    events = ex.extract([person, notebook])
    picked = [e for e in events if e.type is EventType.OBJECT_PICKED_UP]
    carried = [e for e in events if e.type is EventType.OBJECT_CARRIED]
    placed = [e for e in events if e.type is EventType.OBJECT_PLACED]
    assert picked and picked[0].actor_refs == ["p1"]
    assert picked[0].kind == "inferred"
    assert picked[0].details
    assert picked[0].details["joint_motion"] > 0
    assert carried and carried[0].end_time is not None
    assert placed
    # Observed vicinity accompanies the inference as supporting evidence.
    approached = [e for e in events if e.type is EventType.OBJECT_APPROACHED]
    assert approached and all(e.kind == "observed" for e in approached)


def test_walk_past_proximity_alone_is_not_pickup():
    ex = EventExtractor(video_duration=100.0)
    person = moving(
        "p1",
        "person",
        [(0.0, 5.0, 100, 0, -20, 0), (5.0, 10.0, -20, 0, -100, 0)],
    )
    notebook = dense("n1", "notebook", 0.0, 10.0, (0, 0))
    events = ex.extract([person, notebook])
    types = {e.type for e in events}
    assert EventType.OBJECT_APPROACHED in types
    assert EventType.OBJECT_PICKED_UP not in types
    assert EventType.OBJECT_CARRIED not in types
    assert EventType.OBJECT_PLACED not in types


def test_static_person_near_moving_object_is_not_pickup():
    ex = EventExtractor(video_duration=100.0)
    person = dense("p1", "person", 0.0, 10.0, (0, 50))
    obj = moving("n1", "notebook", [(0.0, 10.0, 0, 0, 120, 0)])
    events = ex.extract([person, obj])
    types = {e.type for e in events}
    assert EventType.OBJECT_PICKED_UP not in types
    assert EventType.OBJECT_CARRIED not in types


def test_object_moving_before_person_arrives_is_not_pickup():
    ex = EventExtractor(video_duration=100.0)
    # Object already moving before the person gets near: the
    # stationary-before gate must reject the pickup hypothesis.
    person = moving("p1", "person", [(0.0, 10.0, 200, 0, 60, 0)])
    obj = moving("n1", "notebook", [(0.0, 10.0, 0, 0, 120, 0)])
    events = ex.extract([person, obj])
    types = {e.type for e in events}
    assert EventType.OBJECT_PICKED_UP not in types
