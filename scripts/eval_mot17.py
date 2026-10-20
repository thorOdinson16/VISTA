"""Evaluate VISTA's tracking + memory on MOT17 sequences.

    python scripts/eval_mot17.py --root /data/MOT17/train \\
        --seqs MOT17-09-FRCNN MOT17-02-FRCNN MOT17-05-FRCNN \\
        --sample-fps 2 --out data/eval/mot17_2fps.json

Reports per sequence: MOTA / IDF1 / ID switches / precision / recall on the
frames the pipeline sampled, the grounding accuracy of the "where is the
person" answer, throughput, and the natural-language answer + latency.
MOT17 data is research-licensed: run it where you downloaded it and commit
only the resulting metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from backend.app.config import settings
from backend.app.evaluation.export import locate_grounding, tracks_by_frame
from backend.app.evaluation.mot17 import (
    drop_ignored,
    frames_to_video,
    load_gt,
    read_seqinfo,
)
from backend.app.evaluation.mot_metrics import evaluate_mot
from backend.app.perception.vocabulary import load_label_map
from backend.app.pipeline.pipeline import (
    Pipeline,
    build_default_detector,
    detector_name,
)
from backend.app.reasoning.answerer import Answerer
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever


def evaluate_sequence(seq_dir: Path, args, detector) -> dict:
    info = read_seqinfo(seq_dir)
    gt, ignore = load_gt(seq_dir)
    with tempfile.TemporaryDirectory() as tmp:
        video = frames_to_video(seq_dir, Path(tmp) / f"{info['name']}.mp4")
        pipeline = Pipeline(detector=detector, sample_fps=args.sample_fps)
        start = time.perf_counter()
        result = pipeline.run(video)
        elapsed = time.perf_counter() - start
        memory = result.memory
        fps = result.video_info.fps if result.video_info else info["fps"]

        pred = tracks_by_frame(memory, {"person"}, frame_offset=1)
        # Score exactly the frames the pipeline sampled, including sampled
        # frames where it found nothing (those count as misses).
        sampled = {fid + 1 for fid in result.sampled_frame_ids}
        pred_full = {f: pred.get(f, []) for f in sorted(sampled)}
        pred_full = drop_ignored(pred_full, gt, ignore)
        metrics = evaluate_mot(gt, pred_full)
        grounding = locate_grounding(
            memory, gt, "person", frame_offset=1, sample_frames=set(pred_full), fps=fps
        )

        question = "Where was the person last seen?"
        q_start = time.perf_counter()
        plan = QueryParser().parse(question)
        retrieved = Retriever(memory, source_video=str(video), fps=fps).retrieve(plan)
        answer = Answerer(memory=memory).answer(question, plan, retrieved)
        q_latency = time.perf_counter() - q_start

    return {
        "sequence": info["name"],
        "resolution": f"{info['width']}x{info['height']}",
        "source_fps": info["fps"],
        "frames": info["length"],
        "frames_sampled": result.frames_sampled,
        "processing_seconds": round(elapsed, 2),
        "throughput_fps": round(result.frames_processed / elapsed, 2),
        "tracking": metrics,
        "locate_grounding": grounding,
        "objects_remembered": len(memory.object_memory),
        "events": len(memory.event_memory.all()),
        "query": {
            "question": question,
            "answer": answer.text,
            "confidence": round(answer.confidence, 3),
            "latency_s": round(q_latency, 4),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="MOT17 evaluation for VISTA")
    ap.add_argument("--root", required=True, help="MOT17 train directory")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=settings.detection_confidence_threshold)
    ap.add_argument("--out", default="data/eval/mot17.json")
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
    report = {
        "detector": detector_name(detector),
        "detector_mode": settings.detector_mode,
        "confidence_threshold": args.conf,
        "sample_fps": args.sample_fps,
        "opencv": cv2.__version__,
        "opencv_file": cv2.__file__,
        "sequences": [],
    }
    for name in args.seqs:
        print(f"== {name}", flush=True)
        seq = evaluate_sequence(Path(args.root) / name, args, detector)
        t = seq["tracking"]
        print(
            f"   MOTA {t['mota']:.3f}  IDF1 {t['idf1']:.3f}  IDSW {t['id_switches']}  "
            f"P {t['precision']:.3f}  R {t['recall']:.3f}  "
            f"grounding {seq['locate_grounding']['grounding_accuracy']}  "
            f"{seq['throughput_fps']} fps",
            flush=True,
        )
        report["sequences"].append(seq)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
