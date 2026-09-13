"""Semantic object resolution and the OpenCV appearance vector index."""

from __future__ import annotations

from backend.app.memory.memory import VisualMemory
from backend.app.memory.models import ObjectRecord, TrajectoryPoint
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever


class FakeEmbedder:
    name = "fake"

    def complete(self, prompt, system=None, temperature=0.0):
        return ""

    def embed(self, texts):
        vectors = []
        for text in texts:
            low = text.lower()
            vec = [0.0, 0.0, 0.0]
            if "notebook" in low or "book" in low:
                vec[0] = 1.0
            if "mug" in low:
                vec[1] = 1.0
            if "backpack" in low:
                vec[2] = 1.0
            if all(v == 0.0 for v in vec):
                vec = [0.4, 0.4, 0.2]
            vectors.append(vec)
        return vectors


def build_memory() -> VisualMemory:
    memory = VisualMemory(video_duration=100.0, source_video="demo.mp4", fps=10.0)
    notebook = ObjectRecord(
        object_id="n1",
        category="notebook",
        attributes={"color": "red"},
        first_seen=0.0,
        last_seen=10.0,
        trajectory=_traj([(0.0, 0, 0), (10.0, 0, 0)]),
        confidence=0.9,
    )
    memory.object_memory.upsert_from_object(notebook)
    return memory


def _traj(points):
    return [
        TrajectoryPoint(frame_id=i, timestamp=t, x=x, y=y)
        for i, (t, x, y) in enumerate(points)
    ]


def test_semantic_fallback_resolves_synonym_category():
    memory = build_memory()
    plan = QueryParser().parse("Where is the book?")
    assert plan.target_category == "book"
    # Structured matching finds no "book" object...
    structured = Retriever(memory).retrieve(plan)
    assert structured.objects == []
    # ...but embeddings resolve it to the notebook.
    retriever = Retriever(memory, llm=FakeEmbedder(), semantic_threshold=0.85)
    result = retriever.retrieve(plan)
    assert [o.object_id for o in result.objects] == ["n1"]


def test_explicit_attribute_filter_is_not_overridden():
    memory = build_memory()
    plan = QueryParser().parse("Where is the green book?")
    assert plan.attributes == ["green"]
    retriever = Retriever(memory, llm=FakeEmbedder(), semantic_threshold=0.85)
    result = retriever.retrieve(plan)
    assert result.objects == []


def test_no_embedding_provider_stays_structured():
    memory = build_memory()
    plan = QueryParser().parse("Where is the book?")
    result = Retriever(memory).retrieve(plan)
    assert result.objects == []


def test_visual_index_returns_similar_objects():
    memory = VisualMemory(video_duration=100.0)
    for oid, vec in (
        ("n1", [1.0, 0.0, 0.0]),
        ("n2", [0.95, 0.05, 0.0]),
        ("m1", [0.0, 1.0, 0.0]),
    ):
        memory.object_memory.upsert_from_object(
            ObjectRecord(
                object_id=oid,
                category="notebook",
                appearance_embedding=vec,
                confidence=0.9,
            )
        )
    memory._build_visual_index()
    results = memory.similar_objects("n1", k=2)
    assert results
    assert results[0][0] == "n2"
    assert all(oid != "n1" for oid, _ in results)


def test_visual_index_persists(tmp_path):
    memory = VisualMemory(video_duration=100.0, store_path=tmp_path / "mem.db")
    memory.object_memory.upsert_from_object(
        ObjectRecord(
            object_id="n1",
            category="notebook",
            appearance_embedding=[1.0, 0.0, 0.0],
            confidence=0.9,
        )
    )
    memory._build_visual_index()
    memory.save()

    loaded = VisualMemory.load(tmp_path / "mem.db")
    assert loaded.visual_index is not None
    assert len(loaded.visual_index) == 1
