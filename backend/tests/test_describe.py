"""Describe / "what is shown" queries and the honest no-objects branch."""

from __future__ import annotations

from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import ObjectRecord, TrajectoryPoint
from backend.app.reasoning.answerer import Answerer
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever


def _obj(object_id, category, first, last, color=None):
    attributes = {"color": color} if color else {}
    return ObjectRecord(
        object_id=object_id,
        category=category,
        attributes=attributes,
        first_seen=first,
        last_seen=last,
        trajectory=[
            TrajectoryPoint(frame_id=0, timestamp=first, x=10, y=10),
            TrajectoryPoint(frame_id=1, timestamp=last, x=20, y=20),
        ],
        confidence=0.9,
    )


def build_memory() -> VisualMemory:
    memory = VisualMemory(video_duration=100.0, source_video="demo.mp4", fps=10.0)
    memory.object_memory.upsert_from_object(_obj("n1", "notebook", 0.0, 60.0, "red"))
    memory.object_memory.upsert_from_object(_obj("m1", "mug", 0.0, 60.0, "blue"))
    memory.finalize()
    return memory


def test_parser_routes_describe():
    parser = QueryParser()
    for question in (
        "what is shown in the video?",
        "what's shown?",
        "what do you see?",
        "describe the video",
        "what's in the video",
    ):
        assert parser.parse(question).query_type == "describe", question


def test_describe_lists_objects_and_poster():
    memory = build_memory()
    plan = QueryParser().parse("what is shown in the video?")
    result = Retriever(memory, source_video="demo.mp4", fps=10.0).retrieve(plan)
    answer = Answerer(memory=memory).answer(
        "what is shown in the video?", plan, result
    )
    assert "notebook" in answer.text
    assert "mug" in answer.text
    assert answer.poster and answer.poster.endswith("scene.jpg")
    assert answer.evidence_refs


def test_describe_without_objects_is_honest():
    memory = VisualMemory(video_duration=10.0, source_video="demo.mp4", fps=10.0)
    plan = QueryParser().parse("what is shown in the video?")
    result = Retriever(memory, source_video="demo.mp4", fps=10.0).retrieve(plan)
    answer = Answerer(memory=memory).answer(
        "what is shown in the video?", plan, result
    )
    assert "No configured objects" in answer.text
    assert answer.confidence == 0.0
    assert answer.poster and answer.poster.endswith("scene.jpg")


def test_general_no_longer_uses_history_refusal():
    memory = VisualMemory(video_duration=10.0, source_video="demo.mp4", fps=10.0)
    plan = QueryParser().parse("hello there")
    assert plan.query_type == "general"
    result = Retriever(memory, source_video="demo.mp4", fps=10.0).retrieve(plan)
    answer = Answerer(memory=memory).answer("hello there", plan, result)
    assert "what happened to the object" not in answer.text
