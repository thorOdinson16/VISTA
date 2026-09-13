"""Persistence integration: ingest -> save -> destroy -> load -> query."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.memory.memory import VisualMemory
from backend.app.perception.color_detector import ColorDetector
from backend.app.pipeline.pipeline import Pipeline
from backend.app.reasoning.answerer import Answerer
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever


def make_video(path: str, fps: float = 10.0, frames: int = 60) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(frames):
        img = np.full((240, 320, 3), 255, np.uint8)
        x = int(20 + i * 3)
        img[100:160, x : x + 40] = (0, 0, 220)  # red notebook, moving
        img[100:160, 200:250] = (220, 0, 0)  # blue mug, static
        writer.write(img)
    writer.release()


def query(memory: VisualMemory, question: str):
    plan = QueryParser().parse(question)
    result = Retriever(memory, source_video="v.mp4", fps=10.0).retrieve(plan)
    return Answerer(memory=memory).answer(question, plan, result), result


def test_ingest_save_load_query(tmp_path):
    video = str(tmp_path / "v.mp4")
    make_video(video)
    pipeline = Pipeline(
        detector=ColorDetector(), store_path=tmp_path / "mem.db"
    )
    result = pipeline.run(video)
    assert len(result.memory.object_memory) >= 1

    before, _ = query(result.memory, "Where is the notebook?")
    assert before.text

    # Destroy in-memory state and reload from SQLite.
    del result
    loaded = VisualMemory.load(tmp_path / "mem.db")

    after, after_result = query(loaded, "Where is the notebook?")
    assert "notebook" in after.text
    assert after.evidence_refs
    assert len(loaded.evidence_memory) > 0
    assert loaded.scene_memory.all()

    history, hist_result = query(loaded, "Show me the history of the notebook.")
    assert hist_result.events
    assert history.timeline
