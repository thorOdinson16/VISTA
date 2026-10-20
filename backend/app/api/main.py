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

from ..agent import NEEDS_CONFIRMATION, NOT_FOUND, VERIFIED, LocateAgent
from ..config import settings
from ..llm.factory import create_llm_from_settings
from ..perception.vocabulary import load_label_map, query_terms
from ..pipeline.pipeline import Pipeline, build_default_detector, detector_name
from ..reasoning.answerer import Answerer
from ..retrieval.query_parser import QueryParser
from ..retrieval.retriever import Retriever
from ..services.session import VistaSession
from ..storage import create_object_store
from .schemas import (
    ConfirmRequest,
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
STORE = create_object_store(
    settings.storage, settings.s3_bucket, settings.s3_prefix, settings.aws_region
)


def _run_pipeline(video_path: Path, labels: list[str], sample_fps: float | None = None):
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
        sample_fps=sample_fps or 2.0,
    )
    return pipeline.run(video_path), detector


def _mirror_to_store(video_path: Path) -> None:
    """Copy the video, memory DB and evidence frames to durable object storage."""
    session_id = video_path.stem
    base = f"sessions/{session_id}"
    STORE.delete_prefix(f"{base}/evidence")
    STORE.put_file(video_path, f"{base}/{video_path.name}")
    db = _UPLOAD_DIR / f"{session_id}.db"
    if db.exists():
        STORE.put_file(db, f"{base}/memory.db")
    STORE.put_dir(_EVIDENCE_DIR, f"{base}/evidence", "*.jpg")


def _compute_device() -> str | None:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _clear_evidence() -> None:
    """Drop evidence images from the previous video.

    Evidence files are keyed ``evt_N`` / ``obj_N`` / ``scene``, so a new video
    would otherwise be shown mixed with stale frames from an earlier one.
    """
    for path in _EVIDENCE_DIR.glob("*.jpg"):
        path.unlink(missing_ok=True)


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
        storage=STORE.name,
        device=_compute_device(),
    )


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/ingest", response_model=IngestResult)
async def ingest(file: Annotated[UploadFile, File()]) -> IngestResult:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = _UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with video_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    SESSION.extra_labels = []
    _clear_evidence()
    try:
        result, detector = _run_pipeline(video_path, settings.detector_label_list)
    except ValueError as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    SESSION.detector = detector
    SESSION.pending = []
    SESSION.detector_name = detector_name(detector)
    SESSION.detector_mode = settings.detector_mode
    SESSION.attach(result.memory, str(video_path), result.video_info)
    _mirror_to_store(video_path)

    return IngestResult(
        video=str(video_path),
        objects=len(result.memory.object_memory),
        events=len(result.memory.event_memory.all()),
        frames_processed=result.frames_processed,
        frames_sampled=result.frames_sampled,
        evidence_refs=result.evidence_refs,
    )


def _open_vocab_mode() -> bool:
    return settings.detector_mode in ("openvocab", "owlv2", "hybrid")


def _agent_rescan(fps: float, labels: list[str] | None):
    """Agent tool: re-run detection (new vocabulary and/or denser sampling)."""
    if labels:
        if not _open_vocab_mode():
            return SESSION.memory  # a closed-set detector cannot learn new labels
        for label in labels:
            if label not in SESSION.extra_labels and label not in settings.detector_label_list:
                SESSION.extra_labels.append(label)
    result, detector = _run_pipeline(
        Path(SESSION.source_video),
        settings.detector_label_list + SESSION.extra_labels,
        sample_fps=fps or None,
    )
    SESSION.detector = detector
    SESSION.detector_name = detector_name(detector)
    SESSION.attach(result.memory, SESSION.source_video, result.video_info)
    return SESSION.memory


def _answer_for(objects, plan, question, retriever, llm, confidence):
    """Answer a locate question from an explicit, ordered list of objects."""
    from ..retrieval.retriever import RetrievalResult

    evidence = [e for o in objects if (e := retriever.evidence.for_object_last(o)) is not None]
    result = RetrievalResult(objects=objects, evidence=evidence, confidence=confidence)
    answer = Answerer(memory=SESSION.memory, llm=llm).answer(question, plan, result)
    return answer, result


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

    use_agent = (
        settings.agent_enabled if request.agent is None else request.agent
    ) and plan.query_type == "locate" and bool(plan.target_category) and SESSION.detector is not None
    target = (plan.target_category or "").strip().lower()

    agent_status: str | None = None
    trace: list[dict] = []
    candidates: list[dict] = []
    searched = False
    outcome = None

    if use_agent:
        outcome = LocateAgent(
            memory=memory,
            detector=SESSION.detector,
            video_path=SESSION.source_video or "",
            fps=SESSION.fps,
            rescan=_agent_rescan,
        ).locate(target, plan.attributes)
        memory = SESSION.memory
        agent_status = outcome.status
        trace = outcome.trace
        candidates = outcome.candidates
        searched = any(
            s["tool"] == "rescan" and s["args"].get("labels") for s in trace
        )
    elif (
        target
        and _open_vocab_mode()
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
        SESSION.detector = detector
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
    if outcome is not None and outcome.status == VERIFIED:
        answer, result = _answer_for(
            outcome.objects, plan, request.question, retriever, llm,
            max(outcome.confidence, 0.0),
        )
        answer.text += " (verified on a zoomed crop)"
    elif outcome is not None and outcome.status == NEEDS_CONFIRMATION:
        top = outcome.objects[:2]
        SESSION.pending = [o.object_id for o in outcome.objects[: settings.agent_max_candidates]]
        answer, result = _answer_for(top, plan, request.question, retriever, llm, 0.0)
        answer.text = (
            f"I found possible sightings of the {target} but could not verify them. "
            "Please confirm which one is correct."
        )
        answer.confidence = 0.0
    else:
        result = retriever.retrieve(plan)
        answer = Answerer(memory=memory, llm=llm).answer(request.question, plan, result)

    return QueryResponse(
        question=request.question,
        query_type=plan.query_type,
        target_category=plan.target_category,
        answer=answer,
        evidence=[e.model_dump(mode="json") for e in result.evidence],
        searched_on_demand=searched,
        agent_status=agent_status,
        agent_trace=trace,
        needs_confirmation=bool(outcome and outcome.status == NEEDS_CONFIRMATION),
        candidates=candidates,
    )


@app.post("/api/confirm", response_model=QueryResponse)
def confirm(request: ConfirmRequest) -> QueryResponse:
    """Human decision on a sighting the agent could not verify."""
    memory = SESSION.memory
    if memory is None:
        raise HTTPException(status_code=400, detail="No video ingested yet")
    obj = memory.object_memory.get(request.object_id)
    if obj is None or request.object_id not in SESSION.pending:
        raise HTTPException(status_code=404, detail="No such pending sighting")
    from ..retrieval.query_parser import QueryPlan

    plan = QueryPlan(query_type="locate", target_category=obj.category)
    retriever = Retriever(memory, source_video=SESSION.source_video or "", fps=SESSION.fps)
    trace = [
        {
            "step": 1,
            "tool": "human_confirmation",
            "args": {"object": request.object_id},
            "observation": {"accepted": request.accepted},
            "decision": "use the sighting" if request.accepted else "discard the sighting",
        }
    ]
    if request.accepted:
        SESSION.pending = []
        answer, result = _answer_for(
            [obj], plan, f"Where is the {obj.category}?", retriever, None, 1.0
        )
        answer.text += " (confirmed by you)"
        status, remaining = "confirmed_by_human", []
    else:
        SESSION.pending = [i for i in SESSION.pending if i != request.object_id]
        nxt = [memory.object_memory.get(i) for i in SESSION.pending]
        nxt = [o for o in nxt if o is not None]
        if nxt:
            answer, result = _answer_for(
                nxt[:1], plan, f"Where is the {obj.category}?", retriever, None, 0.0
            )
            answer.text = "Understood. Is this sighting the right one?"
            answer.confidence = 0.0
            status, remaining = NEEDS_CONFIRMATION, [
                {"object_id": o.object_id, "last_seen": o.last_seen} for o in nxt
            ]
        else:
            answer, result = _answer_for([], plan, f"Where is the {obj.category}?", retriever, None, 0.0)
            status, remaining = NOT_FOUND, []
    return QueryResponse(
        question=f"Where is the {obj.category}?",
        query_type="locate",
        target_category=obj.category,
        answer=answer,
        evidence=[e.model_dump(mode="json") for e in result.evidence],
        agent_status=status,
        agent_trace=trace,
        needs_confirmation=status == NEEDS_CONFIRMATION,
        candidates=remaining,
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


# Serve the built frontend from the API so one container hosts everything.
# Registered last so /api, /video and /evidence take precedence.
if settings.frontend_dir.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(settings.frontend_dir), html=True),
        name="frontend",
    )
