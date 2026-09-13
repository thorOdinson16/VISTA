"""Answer grounding: supported vs unsupported vs uncertain."""

from __future__ import annotations

from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import Track, TrackStatus, TrajectoryPoint
from backend.app.reasoning.answerer import Answer, Answerer
from backend.app.reasoning.grounding import ensure_grounded
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever


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
    memory = VisualMemory(video_duration=100.0, source_video="demo.mp4", fps=30.0)
    person = make_track("p1", "person", [(0.0, 100, 0), (4.0, 0, 0), (10.0, 100, 0)])
    notebook = make_track("n1", "notebook", [(0.0, 0, 0), (4.0, 0, 0), (10.0, 100, 0)])
    for t in (0.0, 4.0, 10.0):
        memory.update(int(t), t, [person, notebook])
    memory.finalize()
    return memory


def ask(question):
    memory = build_memory()
    plan = QueryParser().parse(question)
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    return Answerer(memory=memory).answer(question, plan, result), result


def test_supported_answer_keeps_evidence():
    answer, result = ask("Who moved the notebook?")
    assert result.events
    assert answer.evidence_refs
    assert answer.inferred
    assert "Person A" in answer.text


def test_unsupported_actor_is_explicit_uncertainty():
    answer, _ = ask("Who moved the mug?")
    assert "cannot determine" in answer.text
    assert answer.confidence == 0.0
    assert answer.inferred == []


def test_unknown_object_is_explicit_uncertainty():
    answer, result = ask("Where is the laptop?")
    assert result.objects == []
    assert "cannot determine" in answer.text


def test_invented_evidence_refs_are_dropped():
    answer, result = ask("Show me the history of the notebook.")
    assert answer.evidence_refs
    answer.evidence_refs.append("evd_invented_999")
    grounded = ensure_grounded(answer, result)
    assert "evd_invented_999" not in grounded.evidence_refs
    assert grounded.evidence_refs


def test_claims_without_evidence_get_flagged():
    from backend.app.retrieval.retriever import RetrievalResult

    answer = Answer(
        text="The notebook moved.",
        observed=["notebook moved"],
        confidence=0.9,
        evidence_refs=[],
    )
    grounded = ensure_grounded(answer, RetrievalResult())
    assert grounded.text == (
        "I cannot determine this from the available visual evidence."
    )
    assert grounded.confidence == 0.0


def test_observed_vs_inferred_split():
    memory = build_memory()
    plan = QueryParser().parse("Show me the history of the notebook.")
    result = Retriever(memory, source_video="demo.mp4", fps=30.0).retrieve(plan)
    kinds = {e.kind for e in result.events}
    assert "observed" in kinds  # appeared / approached are observed
    assert "inferred" in kinds  # pickup / carry / place are inferred
    answer = Answerer(memory=memory).answer(
        "Show me the history of the notebook.", plan, result
    )
    assert answer.observed
    assert answer.inferred
