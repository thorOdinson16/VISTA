"""Quantitative evaluation of the VISTA pipeline (requirements section 13).

Runs the end-to-end pipeline over the deterministic demo scene and scores
it against the generated ground truth:

* event precision / recall / F1 (type + subject/actor category + time),
* tracking integrity (track fragments per ground-truth object),
* query-answer accuracy against the acceptance-criteria questions,
* processing throughput and query latency.

Ground truth annotates a subset of events, so precision is computed over
predicted events whose type is annotated in the ground truth; the extra
observed event families (moved, approached, ...) are not penalised.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..memory.memory import VisualMemory
from ..perception.color_detector import ColorDetector
from ..pipeline.pipeline import Pipeline
from ..reasoning.answerer import Answerer
from ..retrieval.query_parser import QueryParser
from ..retrieval.retriever import Retriever

# Minimum scores for a passing run (used by --strict and the test suite).
EVENT_F1_MIN = 0.5
QUERY_ACCURACY_MIN = 0.8
EVENT_TIME_TOLERANCE = 6.0

# Acceptance-criteria questions; an answer passes when any accepted
# substring appears (case-insensitive).
QUERY_SUITE: list[tuple[str, list[str]]] = [
    ("Where was the red notebook last seen?", ["notebook"]),
    ("Show me the history of the notebook.", ["notebook"]),
    ("Who moved the notebook?", ["person a"]),
    ("What changed while I was away?", ["change"]),
    ("Where is the mug?", ["mug"]),
]


def _category_map(memory: VisualMemory) -> dict[str, str]:
    return {o.object_id: o.category for o in memory.object_memory.all()}


def _predicted_signatures(memory: VisualMemory) -> list[dict]:
    catmap = _category_map(memory)
    sigs = []
    for event in memory.event_memory.all():
        subject = (
            catmap.get(event.subject_refs[0], event.subject_refs[0])
            if event.subject_refs
            else None
        )
        actor = (
            catmap.get(event.actor_refs[0], event.actor_refs[0])
            if event.actor_refs
            else None
        )
        sigs.append(
            {
                "type": event.type.value,
                "subject": subject,
                "actor": actor,
                "start": event.start_time,
                "end": event.end_time,
            }
        )
    return sigs


def _gt_signatures(gt: dict) -> list[dict]:
    return [
        {
            "type": e["type"],
            "subject": e.get("subject"),
            "actor": e.get("actor"),
            "start": e["start_time"],
            "end": e.get("end_time"),
        }
        for e in gt.get("events", [])
    ]


def match_events(
    preds: list[dict], gts: list[dict], tolerance: float = EVENT_TIME_TOLERANCE
) -> dict:
    used: set[int] = set()
    true_positives = 0
    for gt in gts:
        for i, pred in enumerate(preds):
            if i in used:
                continue
            if pred["type"] != gt["type"]:
                continue
            if pred["subject"] != gt["subject"]:
                continue
            if gt["actor"] is not None and pred["actor"] != gt["actor"]:
                continue
            if abs(pred["start"] - gt["start"]) > tolerance:
                continue
            used.add(i)
            true_positives += 1
            break
    false_positives = len(preds) - true_positives
    false_negatives = len(gts) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def evaluate_events(memory: VisualMemory, gt: dict) -> dict:
    gt_sigs = _gt_signatures(gt)
    annotated_types = {e["type"] for e in gt_sigs}
    pred_sigs = [
        s for s in _predicted_signatures(memory) if s["type"] in annotated_types
    ]
    return match_events(pred_sigs, gt_sigs)


def evaluate_tracking(memory: VisualMemory, gt: dict) -> dict:
    expected = gt.get("objects", {})
    by_category: dict[str, set[str]] = {}
    for obj in memory.object_memory.all():
        by_category.setdefault(obj.category, set()).add(obj.object_id)
    fragments = 0
    per_object = {}
    for category in expected:
        count = len(by_category.get(category, set()))
        per_object[category] = count
        fragments += max(0, count - 1)
    return {"track_fragments": fragments, "tracks_per_object": per_object}


def evaluate_queries(memory: VisualMemory, video: str, fps: float) -> dict:
    results = []
    latencies = []
    passed = 0
    for question, accepted in QUERY_SUITE:
        start = time.perf_counter()
        plan = QueryParser().parse(question)
        result = Retriever(memory, source_video=video, fps=fps).retrieve(plan)
        answer = Answerer(memory=memory).answer(question, plan, result)
        latencies.append(time.perf_counter() - start)
        text = answer.text.lower()
        ok = any(token.lower() in text for token in accepted)
        passed += int(ok)
        results.append(
            {"question": question, "answer": answer.text, "passed": ok}
        )
    total = len(QUERY_SUITE) or 1
    return {
        "accuracy": round(passed / total, 3),
        "passed": passed,
        "total": len(QUERY_SUITE),
        "mean_latency_s": round(sum(latencies) / len(latencies), 4),
        "results": results,
    }


def run_evaluation(
    video: str | Path = "data/demo.mp4",
    gt_path: str | Path = "data/demo_gt.json",
    detector=None,
) -> dict:
    gt = json.loads(Path(gt_path).read_text())
    detector = detector or ColorDetector()

    start = time.perf_counter()
    result = Pipeline(detector=detector).run(video)
    elapsed = time.perf_counter() - start

    events = evaluate_events(result.memory, gt)
    tracking = evaluate_tracking(result.memory, gt)
    fps = result.video_info.fps if result.video_info else 30.0
    queries = evaluate_queries(result.memory, str(video), fps)

    throughput = (
        result.frames_processed / elapsed if elapsed > 0 else 0.0
    )
    return {
        "video": str(video),
        "frames_processed": result.frames_processed,
        "frames_sampled": result.frames_sampled,
        "processing_seconds": round(elapsed, 3),
        "throughput_fps": round(throughput, 2),
        "events": events,
        "tracking": tracking,
        "queries": queries,
        "passed": (
            events["f1"] >= EVENT_F1_MIN
            and queries["accuracy"] >= QUERY_ACCURACY_MIN
        ),
    }


def format_report(report: dict) -> str:
    ev = report["events"]
    tr = report["tracking"]
    qr = report["queries"]
    lines = [
        f"Video: {report['video']}",
        f"Frames: {report['frames_processed']} processed, "
        f"{report['frames_sampled']} sampled",
        f"Throughput: {report['throughput_fps']} fps "
        f"({report['processing_seconds']}s)",
        "",
        "Events (annotated types only):",
        f"  precision={ev['precision']}  recall={ev['recall']}  f1={ev['f1']}",
        f"  TP={ev['true_positives']} FP={ev['false_positives']} "
        f"FN={ev['false_negatives']}",
        "",
        f"Tracking: {tr['track_fragments']} fragment(s) "
        f"{tr['tracks_per_object']}",
        "",
        f"Queries: {qr['passed']}/{qr['total']} "
        f"(accuracy={qr['accuracy']}, mean latency "
        f"{qr['mean_latency_s']}s)",
    ]
    for item in qr["results"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"  [{mark}] {item['question']} -> {item['answer']}")
    lines.append("")
    lines.append("RESULT: " + ("PASS" if report["passed"] else "FAIL"))
    return "\n".join(lines)
