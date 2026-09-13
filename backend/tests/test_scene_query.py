"""Scene-state queries (FR-009/FR-017) over stored spatial snapshots."""

from __future__ import annotations

from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import (
    BoundingBox,
    Observation,
    Track,
    TrackStatus,
    TrajectoryPoint,
)
from backend.app.reasoning.answerer import Answerer
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever


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


def build_memory() -> VisualMemory:
    memory = VisualMemory(video_duration=100.0, source_video="demo.mp4", fps=10.0)
    desk = tracked("desk", "desk", 200, 200, w=300, h=200)
    notebook = tracked("n1", "notebook", 200, 200)
    person = tracked("p1", "person", 205, 200)
    memory.update(0, 0.0, [desk, notebook, person])
    memory.finalize()
    return memory


def ask(question):
    memory = build_memory()
    plan = QueryParser().parse(question)
    result = Retriever(memory, source_video="demo.mp4", fps=10.0).retrieve(plan)
    answer = Answerer(memory=memory).answer(question, plan, result)
    return plan, result, answer


def test_on_relationship_query():
    plan, result, answer = ask("What was on the desk?")
    assert plan.query_type == "state"
    assert any(f["predicate"] == "ON" for f in result.facts)
    assert "was on" in answer.text
    assert answer.evidence_refs


def test_holding_query():
    plan, result, answer = ask("Who was holding the notebook?")
    assert plan.query_type == "state"
    assert any(f["predicate"] == "IN_HAND_OF" for f in result.facts)
    assert "held by" in answer.text


def test_empty_scene_is_explicit_uncertainty():
    memory = VisualMemory(video_duration=10.0, source_video="demo.mp4", fps=10.0)
    plan = QueryParser().parse("What was on the desk?")
    result = Retriever(memory, source_video="demo.mp4", fps=10.0).retrieve(plan)
    answer = Answerer(memory=memory).answer("What was on the desk?", plan, result)
    assert result.facts == []
    assert "cannot determine" in answer.text
    assert answer.confidence == 0.0
