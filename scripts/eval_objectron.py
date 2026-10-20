"""Evaluate "where is the <object>?" on Objectron test videos.

    python scripts/eval_objectron.py --classes cup bottle book laptop chair \\
        --per-class 4 --cache /data/objectron --out data/eval/objectron.json

For each clip the pipeline builds memory, the question "Where is the <cls>?"
goes through the normal parser -> retriever path, and the claimed last-seen
box of every returned object is scored against the clip's ground truth: the
axis-aligned hull of Objectron's projected 3D box (so it is looser than the
visible object; IoU is therefore reported at 0.3 together with a
centre-in-box test). Per-frame detection recall over the sampled frames is
reported too. Data is downloaded on demand and is never committed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from backend.app.config import settings
from backend.app.evaluation.export import tracks_by_frame
from backend.app.evaluation.mot_metrics import iou
from backend.app.evaluation.objectron import (
    BASE_URL,
    INDEX_URL,
    load_geometry,
)
from backend.app.perception.vocabulary import load_label_map
from backend.app.pipeline.pipeline import (
    Pipeline,
    build_default_detector,
    detector_name,
)
from backend.app.reasoning.answerer import Answerer
from backend.app.retrieval.query_parser import QueryParser
from backend.app.retrieval.retriever import Retriever

IOU_HIT = 0.3


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def select_clips(cls: str, n: int, seed: int) -> list[str]:
    with urllib.request.urlopen(INDEX_URL.format(cls=cls), timeout=60) as resp:
        index = sorted(line.strip() for line in resp.read().decode().splitlines() if line.strip())
    return random.Random(seed).sample(index, min(n, len(index)))


def fetch_clip(clip: str, cache: Path) -> tuple[Path, Path]:
    video = cache / "videos" / clip / "video.MOV"
    ann = cache / "annotations" / f"{clip}.pbdata"
    _download(f"{BASE_URL}/{clip}/video.MOV", video)
    _download(f"https://storage.googleapis.com/objectron/annotations/{clip}.pbdata", ann)
    return video, ann


def _center_in(box, gt) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def evaluate_clip(clip: str, cls: str, cache: Path, detector, args, fig_dir: Path | None) -> dict:
    video, ann = fetch_clip(clip, cache)
    frames = load_geometry(ann)
    cap = cv2.VideoCapture(str(video))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    first_ok, first = cap.read()
    cap.release()
    if first_ok:
        height, width = first.shape[:2]

    gt: dict[int, tuple[float, float, float, float]] = {}
    for f in frames:
        if f.objects:
            x1, y1, x2, y2 = f.objects[0].box_norm()
            gt[f.frame_id] = (x1 * width, y1 * height, x2 * width, y2 * height)

    pipeline = Pipeline(detector=detector, sample_fps=args.sample_fps)
    start = time.perf_counter()
    result = pipeline.run(video)
    elapsed = time.perf_counter() - start
    memory = result.memory
    fps = result.video_info.fps if result.video_info else 30.0

    question = f"Where is the {cls}?"
    q_start = time.perf_counter()
    plan = QueryParser().parse(question)
    retrieved = Retriever(memory, source_video=str(video), fps=fps).retrieve(plan)
    answer = Answerer(memory=memory).answer(question, plan, retrieved)
    q_latency = time.perf_counter() - q_start
    target = plan.target_category

    claims = []
    for obj in retrieved.objects:
        if not obj.observations:
            continue
        last = max(obj.observations, key=lambda o: o.frame_id)
        b = last.bounding_box
        box = (b.x_min, b.y_min, b.x_max, b.y_max)
        g = gt.get(last.frame_id)
        claims.append(
            {
                "object": obj.object_id,
                "frame": last.frame_id,
                "iou": round(iou(box, g), 3) if g else None,
                "center_in_gt": bool(g and _center_in(box, g)),
                "box": [round(v, 1) for v in box],
            }
        )
    hit_iou = any((c["iou"] or 0) >= IOU_HIT for c in claims)
    hit_center = any(c["center_in_gt"] for c in claims)

    per_frame = tracks_by_frame(memory, {target} if target else set())
    sampled = result.sampled_frame_ids
    detected = sum(
        1
        for fid in sampled
        if fid in gt and any(iou(b, gt[fid]) >= IOU_HIT for _, b in per_frame.get(fid, []))
    )
    scored = sum(1 for fid in sampled if fid in gt)

    if not retrieved.objects:
        outcome = "not_found"
    elif hit_iou:
        outcome = "hit"
    elif hit_center:
        outcome = "hit_loose"
    else:
        outcome = "wrong_location"

    if fig_dir is not None and first_ok and claims:
        img = first.copy()
        last_frame = claims[0]["frame"]
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
        ok, img = cap.read()
        cap.release()
        if ok and last_frame in gt:
            g = gt[last_frame]
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 6)
            for c in claims:
                if c["frame"] == last_frame:
                    p = c["box"]
                    cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 6)
            fig_dir.mkdir(parents=True, exist_ok=True)
            small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
            cv2.imwrite(str(fig_dir / f"{outcome}_{cls}_{clip.replace('/', '_')}.jpg"), small)

    return {
        "clip": clip,
        "class": cls,
        "target_category": target,
        "frame_size": f"{width}x{height}",
        "frames_sampled": result.frames_sampled,
        "processing_seconds": round(elapsed, 2),
        "outcome": outcome,
        "objects_returned": len(retrieved.objects),
        "claims": claims,
        "frame_recall": round(detected / scored, 3) if scored else None,
        "answer": answer.text,
        "confidence": round(answer.confidence, 3),
        "query_latency_s": round(q_latency, 4),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    count = lambda o: sum(1 for r in rows if r["outcome"] == o)  # noqa: E731
    recalls = [r["frame_recall"] for r in rows if r["frame_recall"] is not None]
    return {
        "clips": len(rows),
        "found_rate": round(1 - count("not_found") / n, 3),
        "hit_rate_iou30": round(count("hit") / n, 3),
        "hit_rate_center": round((count("hit") + count("hit_loose")) / n, 3),
        "wrong_location": count("wrong_location"),
        "not_found": count("not_found"),
        "mean_frame_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "mean_processing_s": round(sum(r["processing_seconds"] for r in rows) / n, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Objectron 'where is X' evaluation")
    ap.add_argument("--classes", nargs="+", default=["cup", "bottle", "book", "laptop", "chair"])
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=settings.detection_confidence_threshold)
    ap.add_argument("--cache", default="data/objectron")
    ap.add_argument("--figs", default=None, help="directory for outcome thumbnails")
    ap.add_argument("--out", default="data/eval/objectron.json")
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
    fig_dir = Path(args.figs) if args.figs else None
    rows: list[dict] = []
    for cls in args.classes:
        for clip in select_clips(cls, args.per_class, args.seed):
            print(f"== {cls} {clip}", flush=True)
            row = evaluate_clip(clip, cls, cache, detector, args, fig_dir)
            print(f"   {row['outcome']}  recall {row['frame_recall']}  {row['processing_seconds']}s", flush=True)
            rows.append(row)

    report = {
        "detector": detector_name(detector),
        "detector_mode": settings.detector_mode,
        "confidence_threshold": args.conf,
        "sample_fps": args.sample_fps,
        "iou_hit": IOU_HIT,
        "opencv": cv2.__version__,
        "overall": summarize(rows),
        "per_class": {c: summarize([r for r in rows if r["class"] == c]) for c in args.classes},
        "clips": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
