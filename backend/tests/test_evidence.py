"""Evidence as a first-class flow: stored at finalize, resolved by lookup."""

from __future__ import annotations

from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import Track, TrackStatus, TrajectoryPoint


def make_track(track_id, category, points):
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


def build_memory(**kwargs) -> VisualMemory:
    memory = VisualMemory(video_duration=100.0, **kwargs)
    person = make_track("p1", "person", [(0.0, 100, 0), (4.0, 0, 0), (10.0, 100, 0)])
    notebook = make_track("n1", "notebook", [(0.0, 0, 0), (4.0, 0, 0), (10.0, 100, 0)])
    for t in (0.0, 4.0, 10.0):
        memory.update(int(t), t, [person, notebook])
    memory.finalize()
    return memory


def test_every_event_has_stored_evidence():
    memory = build_memory(source_video="demo.mp4", fps=10.0)
    assert memory.event_memory.all()
    for event in memory.event_memory.all():
        assert event.evidence_refs, f"{event.event_id} has no evidence refs"
        for eid in event.evidence_refs:
            rec = memory.evidence_memory.get(eid)
            assert rec is not None
            assert event.event_id in rec.event_refs


def test_evidence_carries_source_location():
    memory = build_memory(source_video="demo.mp4", fps=10.0)
    for rec in memory.evidence_memory.all():
        assert rec.source_video == "demo.mp4"
        assert rec.frame_start == int(round(rec.timestamp_start * 10.0))


def test_evidence_uses_actual_fps_not_hardcoded():
    memory = build_memory(source_video="demo.mp4", fps=10.0)
    recs = memory.evidence_memory.by_object("n1")
    assert recs
    # At 10 fps, t=4.0s is frame 40 -- not frame 120 (30 fps hardcode).
    assert any(r.frame_start == 40 for r in recs)


def test_observation_evidence_resolves_to_source_frame():
    from backend.app.retrieval.evidence import EvidenceResolver

    memory = build_memory(source_video="demo.mp4", fps=10.0)
    obj = memory.object_memory.get("n1")
    assert obj is not None
    for obs in obj.observations:
        assert obs.evidence_refs
    resolver = EvidenceResolver(memory=memory)
    recs = resolver.for_object(obj)
    assert recs
    last = resolver.for_object_last(obj)
    assert last is not None
    assert last.timestamp_start == obj.last_seen


def test_object_event_refs_linked():
    memory = build_memory()
    obj = memory.object_memory.get("n1")
    assert obj is not None and obj.event_refs
    for eid in obj.event_refs:
        assert any(e.event_id == eid for e in memory.event_memory.all())


def test_evidence_survives_persistence(tmp_path):
    memory = build_memory(
        source_video="demo.mp4", fps=10.0, store_path=tmp_path / "mem.db"
    )
    memory.save()
    loaded = VisualMemory.load(tmp_path / "mem.db")
    assert len(loaded.evidence_memory) == len(memory.evidence_memory)
    for event in loaded.event_memory.all():
        assert event.evidence_refs
        for eid in event.evidence_refs:
            assert loaded.evidence_memory.get(eid) is not None
    # Counters reseeded: new records must not collide with loaded IDs.
    rec = loaded.evidence_memory.record(0, 0, 0.0, 0.0, ["x"], [])
    assert loaded.evidence_memory.get(rec.evidence_id) is rec
    assert len(loaded.evidence_memory) == len(memory.evidence_memory) + 1
