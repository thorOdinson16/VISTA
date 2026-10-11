"""FastAPI application for VISTA (ingest, query, evidence, timeline)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import settings
from ..llm.factory import create_llm_from_settings
from ..perception.vocabulary import load_label_map, query_terms
from ..pipeline.pipeline import Pipeline, build_default_detector, detector_name
from ..reasoning.answerer import Answerer
from ..retrieval.query_parser import QueryParser
from ..retrieval.retriever import Retriever
from ..services.session import VistaSession
from .schemas import (
    IngestResult,
    QueryRequest,
    QueryResponse,
    StatusResponse,
    TimelineResponse,
)

app = FastAPI(title="VISTA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-session MVP state, isolated behind the service boundary.
SESSION = VistaSession()
settings.ensure_dirs()
_UPLOAD_DIR = settings.uploads_dir
_EVIDENCE_DIR = settings.evidence_dir


def _run_pipeline(video_path: Path, labels: list[str]):
    """Detect + track + remember ``video_path`` with an open-vocab ``labels`` list."""
    detector = build_default_detector(
        model_path=settings.yolo_model_path,
        mode=settings.detector_mode,
        models_dir=settings.models_dir,
        conf_threshold=settings.detection_confidence_threshold,
        class_names_path=settings.yolo_class_names_path,
        label_map=load_label_map(settings.detector_label_map),
        labels=labels,
        openvocab_model=settings.openvocab_model,
        owlv2_model=settings.owlv2_model,
    )
    store_path = _UPLOAD_DIR / f"{video_path.stem}.db"
    pipeline = Pipeline(
        detector=detector,
        evidence_dir=_EVIDENCE_DIR,
        store_path=store_path,
        saliency_fallback=settings.saliency_fallback,
        saliency_min_area=settings.saliency_min_area,
        saliency_min_overlap=settings.saliency_min_persistence_overlap,
        scene_change_threshold=settings.scene_change_threshold,
        scene_change_motion_gate=settings.scene_change_motion_gate,
    )
    return pipeline.run(video_path), detector


def _known_categories() -> set[str]:
    memory = SESSION.memory
    if memory is None:
        return set()
    return {str(o.category).lower() for o in memory.object_memory.all()}



@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    memory = SESSION.memory
    count = len(memory.object_memory) if memory else 0
    events = len(memory.event_memory.all()) if memory else 0
    return StatusResponse(
        video=SESSION.source_video,
        objects=count,
        events=events,
        fps=SESSION.fps if SESSION.ready else None,
        duration=SESSION.duration,
        detector_mode=SESSION.detector_mode or settings.detector_mode,
        detector_name=SESSION.detector_name,
        saliency_fallback=settings.saliency_fallback,
    )


@app.post("/api/ingest", response_model=IngestResult)
async def ingest(file: Annotated[UploadFile, File()]) -> IngestResult:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = _UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with video_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    SESSION.extra_labels = []
    try:
        result, detector = _run_pipeline(video_path, settings.detector_label_list)
    except ValueError as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    SESSION.detector_name = detector_name(detector)
    SESSION.detector_mode = settings.detector_mode
    SESSION.attach(result.memory, str(video_path), result.video_info)

    return IngestResult(
        video=str(video_path),
        objects=len(result.memory.object_memory),
        events=len(result.memory.event_memory.all()),
        frames_processed=result.frames_processed,
        frames_sampled=result.frames_sampled,
        evidence_refs=result.evidence_refs,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    memory = SESSION.memory
    if memory is None:
        raise HTTPException(status_code=400, detail="No video ingested yet")

    llm = create_llm_from_settings(settings)
    plan = QueryParser(
        llm=llm,
        vocabulary=query_terms(
            load_label_map(settings.detector_label_map),
            extra_labels=settings.detector_label_list + SESSION.extra_labels,
        ),
    ).parse(request.question)

    searched = False
    target = (plan.target_category or "").strip().lower()
    if (
        target
        and settings.detector_mode in ("openvocab", "owlv2", "hybrid")
        and SESSION.source_video
        and target not in _known_categories()
        and target not in SESSION.extra_labels
        and target not in settings.detector_label_list
    ):
        # The user asked for something the detector was never prompted with:
        # add it to the vocabulary and re-detect, instead of answering "not found".
        SESSION.extra_labels.append(target)
        try:
            result, detector = _run_pipeline(
                Path(SESSION.source_video),
                settings.detector_label_list + SESSION.extra_labels,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        SESSION.detector_name = detector_name(detector)
        SESSION.attach(result.memory, SESSION.source_video, result.video_info)
        memory = SESSION.memory
        searched = True
    retriever = Retriever(
        memory,
        source_video=SESSION.source_video or "",
        fps=SESSION.fps,
        llm=llm,
        semantic_threshold=settings.semantic_retrieval_threshold,
    )
    result = retriever.retrieve(plan)
    answer = Answerer(memory=memory, llm=llm).answer(
        request.question, plan, result
    )

    return QueryResponse(
        question=request.question,
        query_type=plan.query_type,
        target_category=plan.target_category,
        answer=answer,
        evidence=[e.model_dump(mode="json") for e in result.evidence],
        searched_on_demand=searched,
    )


@app.get("/api/timeline", response_model=TimelineResponse)
def timeline() -> TimelineResponse:
    memory = SESSION.memory
    if memory is None:
        raise HTTPException(status_code=400, detail="No video ingested yet")
    events = [
        e.model_dump(mode="json")
        for e in sorted(memory.event_memory.all(), key=lambda e: e.start_time)
    ]
    return TimelineResponse(events=events, duration=SESSION.duration)


@app.get("/api/similar")
def similar(object_id: str, k: int = 5) -> dict:
    """Visually similar objects (OpenCV appearance descriptor search)."""
    memory = SESSION.memory
    if memory is None:
        raise HTTPException(status_code=400, detail="No video ingested yet")
    results = memory.similar_objects(object_id, k=k)
    return {
        "object_id": object_id,
        "similar": [{"object_id": oid, "score": score} for oid, score in results],
    }


@app.get("/api/scene")
def scene(t: float | None = None) -> dict:
    memory = SESSION.memory
    if memory is None:
        raise HTTPException(status_code=400, detail="No video ingested yet")
    state = (
        memory.scene_memory.at(t) if t is not None else memory.scene_memory.latest()
    )
    if state is None:
        return {"state": None}
    return {"state": state.model_dump(mode="json")}


@app.get("/api/objects")
def objects() -> dict:
    memory = SESSION.memory
    if memory is None:
        raise HTTPException(status_code=400, detail="No video ingested yet")
    return {
        "objects": [o.model_dump(mode="json") for o in memory.object_memory.all()]
    }


@app.get("/video")
def video() -> FileResponse:
    video_path = SESSION.source_video
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=400, detail="No video ingested yet")
    return FileResponse(video_path, media_type="video/mp4")


_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/evidence", StaticFiles(directory=str(_EVIDENCE_DIR)), name="evidence"
)
