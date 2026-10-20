"""Evaluate the verify loop (perception -> decision -> action) on Objectron.

    python scripts/eval_agent.py --classes cup bottle book laptop chair \\
        --per-class 4 --cache data/objectron --out docs/eval/agent_yolo.json

Per clip the pipeline builds memory once; then, for "Where is the X?":

* baseline - the retriever's lead sighting, no checking;
* agent    - ``LocateAgent`` (crop verification, denser re-scan, ask the human).

Scoring is against Objectron's projected 3D box (IoU >= 0.3 on the last
sighting's frame). When the agent defers to the human, "assisted" success
means the correct sighting was among the options shown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from backend.app.agent import (
    NEEDS_CONFIRMATION,
    NOT_FOUND,
    VERIFIED,
    LocateAgent,
)
from backend.app.config import settings
from backend.app.evaluation.mot_metrics import iou as box_iou
from backend.app.evaluation.objectron import load_geometry
from backend.app.perception.vocabulary import load_label_map
from backend.app.pipeline.pipeline import Pipeline, build_default_detector, detector_name
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever
from scripts.eval_objectron import IOU_HIT, fetch_clip, select_clips  # noqa: E402


def last_box(obj):
    last = max(obj.observations, key=lambda o: o.frame_id)
    b = last.bounding_box
    return last.frame_id, (b.x_min, b.y_min, b.x_max, b.y_max)


def hit(obj, gt) -> bool:
    frame, box = last_box(obj)
    g = gt.get(frame)
    return bool(g and box_iou(box, g) >= IOU_HIT)


def evaluate_clip(clip, cls, cache, detector, args) -> dict:
    video, ann = fetch_clip(clip, cache)
    first = cv2.VideoCapture(str(video))
    ok, frame = first.read()
    first.release()
    h, w = frame.shape[:2]
    gt = {
        f.frame_id: (
            f.objects[0].box_norm()[0] * w,
            f.objects[0].box_norm()[1] * h,
            f.objects[0].box_norm()[2] * w,
            f.objects[0].box_norm()[3] * h,
        )
        for f in load_geometry(ann)
        if f.objects
    }

    result = Pipeline(detector=detector, sample_fps=args.sample_fps).run(video)
    memory = result.memory
    fps = result.video_info.fps if result.video_info else 30.0
    plan = QueryParser().parse(f"Where is the {cls}?")
    target = plan.target_category

    # Baseline: the retriever's lead sighting.
    retrieved = Retriever(memory, source_video=str(video), fps=fps).retrieve(plan)
    base_hit = bool(retrieved.objects) and hit(retrieved.objects[0], gt)

    rescan_calls = []

    def rescan(sample_fps, labels):
        rescan_calls.append(sample_fps)
        if not sample_fps:
            return memory
        return Pipeline(detector=detector, sample_fps=sample_fps).run(video).memory

    start = time.perf_counter()
    outcome = LocateAgent(
        memory=memory,
        detector=detector,
        video_path=str(video),
        fps=fps,
        rescan=rescan,
    ).locate(target, plan.attributes)
    elapsed = time.perf_counter() - start

    # The agent may have re-scanned; its objects live in outcome.memory.
    shown = outcome.objects[:2]
    chosen_hit = bool(outcome.objects) and outcome.status == VERIFIED and hit(outcome.objects[0], gt)
    assisted = outcome.status == NEEDS_CONFIRMATION and any(hit(o, gt) for o in shown)
    # Rescans re-index objects, so the gt check above uses the memory that
    # produced the candidates (outcome.objects belong to outcome.memory).
    return {
        "clip": clip,
        "class": cls,
        "target": target,
        "candidates_in_memory": len(retrieved.objects),
        "baseline_hit": base_hit,
        "status": outcome.status,
        "agent_auto_hit": chosen_hit,
        "agent_assisted_hit": assisted,
        "tool_calls": outcome.tool_calls,
        "rescans": outcome.rescans,
        "agent_seconds": round(elapsed, 2),
        "trace": outcome.trace,
    }


def summarize(rows) -> dict:
    n = len(rows) or 1
    verified = [r for r in rows if r["status"] == VERIFIED]
    deferred = [r for r in rows if r["status"] == NEEDS_CONFIRMATION]
    not_found = [r for r in rows if r["status"] == NOT_FOUND]
    auto_hits = sum(r["agent_auto_hit"] for r in verified)
    assisted = sum(r["agent_assisted_hit"] for r in deferred)
    return {
        "clips": len(rows),
        "baseline_lead_hit": round(sum(r["baseline_hit"] for r in rows) / n, 3),
        "agent_verified": len(verified),
        "agent_verified_precision": round(auto_hits / len(verified), 3) if verified else None,
        "agent_deferred_to_human": len(deferred),
        "agent_not_found": len(not_found),
        "agent_auto_success": round(auto_hits / n, 3),
        "agent_success_with_human": round((auto_hits + assisted) / n, 3),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
        "mean_rescans": round(sum(r["rescans"] for r in rows) / n, 2),
        "mean_agent_seconds": round(sum(r["agent_seconds"] for r in rows) / n, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["cup", "bottle", "book", "laptop", "chair"])
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=settings.detection_confidence_threshold)
    ap.add_argument("--cache", default="data/objectron")
    ap.add_argument("--out", default="docs/eval/agent.json")
    args = ap.parse_args()

    detector = build_default_detector(
        model_path=settings.yolo_model_path,
        mode=settings.detector_mode,
        models_dir=settings.models_dir,
        conf_threshold=args.conf,
        class_names_path=settings.yolo_class_names_path,
        label_map=load_label_map(settings.detector_label_map),
        labels=settings.detector_label_list,
    )
    cache = Path(args.cache)
    rows = []
    for cls in args.classes:
        for clip in select_clips(cls, args.per_class, args.seed, cache):
            row = evaluate_clip(clip, cls, cache, detector, args)
            print(f"{cls:7s} {clip:18s} base={row['baseline_hit']!s:5s} agent={row['status']:19s} auto={row['agent_auto_hit']!s:5s} assisted={row['agent_assisted_hit']}", flush=True)
            rows.append(row)
    report = {
        "detector": detector_name(detector),
        "detector_mode": settings.detector_mode,
        "sample_fps": args.sample_fps,
        "overall": summarize(rows),
        "per_class": {c: summarize([r for r in rows if r["class"] == c]) for c in args.classes},
        "clips": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
