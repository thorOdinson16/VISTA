from __future__ import annotations

from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import (
    EventRecord,
    EventType,
    ObjectRecord,
    Track,
    TrackStatus,
    TrajectoryPoint,
)
from backend.app.reasoning.answerer import Answerer
from backend.app.retrieval.evidence import EvidenceResolver
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever
from backend.app.retrieval.temporal import after, before, first, last
from backend.app.retrieval.vector import VectorStore


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


def build_memory() -> VisualMemory:
    memory = VisualMemory(video_duration=100.0)
    person = make_track("p1", "person", [(0.0, 100, 0), (4.0, 0, 0), (10.0, 100, 0)])
    notebook = make_track("n1", "notebook", [(0.0, 0, 0), (4.0, 0, 0), (10.0, 100, 0)])
    for t in (0.0, 4.0, 10.0):
        memory.update(int(t), t, [person, notebook])
    memory.finalize()
    return memory


def test_query_parser_rules():
    parser = QueryParser()
    plan = parser.parse("Who moved the mug?")
    assert plan.query_type == "actor"
    assert plan.target_category == "mug"
    assert plan.relationship == "moved"

    plan = parser.parse("Where was the red notebook last seen?")
    assert plan.query_type == "locate"
    assert plan.target_category == "notebook"
    assert plan.attributes == ["red"]

    plan = parser.parse("Show me the history of the notebook.")
    assert plan.query_type == "history"

    plan = parser.parse("What changed while I was away?")
    assert plan.query_type == "change"


def test_temporal_operators():
    from backend.app.memory.models import EventRecord, EventType

    events = [
        EventRecord(type=EventType.OBJECT_MOVED, start_time=5.0),
        EventRecord(type=EventType.OBJECT_MOVED, start_time=10.0),
        EventRecord(type=EventType.OBJECT_MOVED, start_time=15.0),
    ]
    assert len(before(events, 10.0)) == 1
    assert len(after(events, 10.0)) == 1
    assert first(events).start_time == 5.0
    assert last(events).start_time == 15.0


def test_vector_store():
    store = VectorStore(dim=3)
    store.add("a", [1, 0, 0])
    store.add("b", [0, 1, 0])
    results = store.search([1, 0, 0], k=2)
    assert results[0][0] == "a"
    assert results[0][1] > results[1][1]


def test_evidence_resolver():
    from backend.app.memory.memory import VisualMemory
    from backend.app.memory.models import EventRecord, EventType

    memory = VisualMemory(source_video="demo.mp4", fps=30.0)
    event = EventRecord(
        event_id="e1", type=EventType.OBJECT_MOVED, start_time=2.0, end_time=4.0,
        subject_refs=["n1"],
    )
    rec = memory.evidence_memory.record_for_event(event)
    event.evidence_refs.append(rec.evidence_id)

    resolver = EvidenceResolver(memory=memory)
    ev = resolver.for_event(event)
    assert ev is not None
    assert ev.frame_start == 60
    assert ev.frame_end == 120
    assert ev.source_video == "demo.mp4"
    assert ev.event_refs == ["e1"]
    assert ev.object_refs == ["n1"]


def test_retrieve_actor_and_answer():
    memory = build_memory()
    parser = QueryParser()
    retriever = Retriever(memory, source_video="demo.mp4", fps=30.0)
    answerer = Answerer(memory=memory)

    plan = parser.parse("Who moved the notebook?")
    result = retriever.retrieve(plan)
    answer = answerer.answer("Who moved the notebook?", plan, result)

    assert result.events
    assert "Person A" in answer.text
    assert answer.inferred
    assert answer.evidence_refs


def test_retrieve_locate():
    memory = build_memory()
    parser = QueryParser()
    retriever = Retriever(memory, source_video="demo.mp4", fps=30.0)
    answerer = Answerer(memory=memory)

    plan = parser.parse("Where is the notebook?")
    result = retriever.retrieve(plan)
    answer = answerer.answer("Where is the notebook?", plan, result)

    assert result.objects
    assert "notebook" in answer.text
    assert answer.evidence_refs


def test_retrieve_history_timeline():
    memory = build_memory()
    parser = QueryParser()
    retriever = Retriever(memory, source_video="demo.mp4", fps=30.0)
    answerer = Answerer(memory=memory)

    plan = parser.parse("Show me the history of the notebook.")
    result = retriever.retrieve(plan)
    answer = answerer.answer("Show me the history of the notebook.", plan, result)

    assert answer.timeline
    assert any(e.start_time <= answer.timeline[-1].time for e in result.events)


def _traj(points):
    return [
        TrajectoryPoint(frame_id=i, timestamp=t, x=x, y=y)
        for i, (t, x, y) in enumerate(points)
    ]


def build_attr_memory() -> VisualMemory:
    """Two notebooks (red disappears LAST) for attribute/temporal tests."""
    memory = VisualMemory(video_duration=100.0, source_video="demo.mp4", fps=30.0)
    red = ObjectRecord(
        object_id="n1", category="notebook", attributes={"color": "red"},
        first_seen=0.0, last_seen=60.0,
        trajectory=_traj([(0.0, 0, 0), (50.0, 100, 0), (60.0, 100, 0)]),
        confidence=0.9,
    )
    blue = ObjectRecord(
        object_id="n2", category="notebook", attributes={"color": "blue"},
        first_seen=0.0, last_seen=20.0,
        trajectory=_traj([(0.0, 200, 200), (10.0, 250, 200), (20.0, 250, 200)]),
        confidence=0.9,
    )
    memory.object_memory.upsert_from_object(red)
    memory.object_memory.upsert_from_object(blue)
    specs = [
        (EventType.OBJECT_APPEARED, 0.0, ["n1"], "observed"),
        (EventType.OBJECT_MOVED, 50.0, ["n1"], "inferred"),
        (EventType.OBJECT_DISAPPEARED, 60.0, ["n1"], "observed"),
        (EventType.OBJECT_APPEARED, 0.0, ["n2"], "observed"),
        (EventType.OBJECT_MOVED, 10.0, ["n2"], "inferred"),
        (EventType.OBJECT_DISAPPEARED, 20.0, ["n2"], "observed"),
    ]
    for i, (etype, t, subjects, kind) in enumerate(specs):
        event = EventRecord(
            event_id=f"e{i}", type=etype, start_time=t,
            subject_refs=subjects, confidence=0.9, kind=kind,
        )
        memory.event_memory.add(event)
        rec = memory.evidence_memory.record_for_event(event)
        event.evidence_refs.append(rec.evidence_id)
    return memory


def test_attribute_matching_red_notebook():
    memory = build_attr_memory()
    plan = QueryParser().parse("Where was the red notebook last seen?")
    assert plan.attributes == ["red"]
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    assert [o.object_id for o in result.objects] == ["n1"]
    answer = Answerer(memory=memory).answer(
        "Where was the red notebook last seen?", plan, result
    )
    assert answer.evidence_refs


def test_attribute_unknown_color_matches_nothing():
    memory = build_attr_memory()
    plan = QueryParser().parse("Where was the green notebook last seen?")
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    assert result.objects == []
    answer = Answerer(memory=memory).answer(
        "Where was the green notebook last seen?", plan, result
    )
    assert "cannot determine" in answer.text


def test_temporal_before_resolves_entity_first():
    # Red disappears at 60, blue at 20. The anchor must be RED's
    # disappearance (60), not the first disappearance globally (20).
    memory = build_attr_memory()
    question = "What happened before the red notebook disappeared?"
    plan = QueryParser().parse(question)
    assert plan.temporal is not None and plan.temporal.op == "before"
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    assert result.events
    assert all(e.start_time < 60.0 for e in result.events)
    assert {s for e in result.events for s in e.subject_refs} == {"n1"}


def test_temporal_after_excludes_anchor():
    memory = build_attr_memory()
    question = "What happened after the blue notebook disappeared?"
    plan = QueryParser().parse(question)
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    assert all(e.start_time > 20.0 for e in result.events)
    assert all("n2" not in e.subject_refs or e.start_time > 20.0 for e in result.events)


def test_temporal_between_pickup_and_place():
    memory = build_memory()
    question = "What happened between the notebook being picked up and placed?"
    plan = QueryParser().parse(question)
    assert plan.temporal is not None and plan.temporal.op == "between"
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    types = {e.type for e in result.events}
    assert EventType.OBJECT_CARRIED in types


def test_temporal_first_and_last():
    memory = build_attr_memory()
    retriever = Retriever(memory, source_video="demo.mp4", fps=30.0)
    first_plan = QueryParser().parse("What happened first to the red notebook?")
    first = retriever.retrieve(first_plan)
    assert len(first.events) == 1 and first.events[0].start_time == 0.0
    last_plan = QueryParser().parse("What happened last to the red notebook?")
    last = retriever.retrieve(last_plan)
    assert len(last.events) == 1 and last.events[0].start_time == 60.0
