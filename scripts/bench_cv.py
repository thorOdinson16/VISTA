"""Fixed OpenCV workload for the COOL vs stock-OpenCV vs x86 comparison.

    python scripts/bench_cv.py --video clip.MOV --model data/models/yolov8n.onnx \\
        --repeats 5 --label cool-arm --out bench_cool-arm.json

Every stage is exactly what the VISTA pipeline runs, on identical input, so
the numbers are comparable across instances. Each stage is warmed up once and
timed ``--repeats`` times; the report holds median and min per stage plus
the OpenCV build (version, file, custom HAL such as KleidiCV), CPU and thread
count, so the claim "COOL executed this workload" is verifiable from the
JSON alone.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def cpu_model() -> str:
    """Human-readable CPU name (lscpu covers Arm, where /proc/cpuinfo has none)."""
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def build_flags() -> dict:
    info = cv2.getBuildInformation().splitlines()
    keys = ("Version control", "Custom HAL", "Baseline", "Dispatched", "FFMPEG", "Parallel framework")
    out = {}
    for line in info:
        for key in keys:
            if line.strip().startswith(key):
                out[key] = line.split(":", 1)[1].strip()
    return out


def load_frames(video: str, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def timed(fn, repeats: int) -> dict:
    fn()  # warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "samples_s": [round(s, 4) for s in samples],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="YOLO ONNX for cv2.dnn")
    ap.add_argument("--frames", type=int, default=120, help="frames used by per-frame stages")
    ap.add_argument("--dnn-frames", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--pipeline", action="store_true", help="also time the full VISTA pipeline")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()

    frames = load_frames(args.video, args.frames)
    if not frames:
        raise SystemExit(f"no frames decoded from {args.video}")
    h, w = frames[0].shape[:2]
    small = [cv2.resize(f, (320, 180)) for f in frames]
    gray_small = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]
    stages: dict[str, dict] = {}

    def decode():
        cap = cv2.VideoCapture(args.video)
        n = 0
        while n < args.frames:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()

    def resize():
        for f in frames:
            cv2.resize(f, (640, 640), interpolation=cv2.INTER_LINEAR)
            cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA)

    def imgproc():
        for f in small:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            cv2.Canny(g, 80, 160)
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])

    def flow():
        for a, b in zip(gray_small, gray_small[1:]):
            cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    orb = cv2.ORB_create(nfeatures=500)

    def features():
        for f in frames[:40]:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            orb.detectAndCompute(g, None)

    def background():
        mog = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
        for f in small:
            mog.apply(f)

    stages["decode"] = timed(decode, args.repeats)
    stages["resize"] = timed(resize, args.repeats)
    stages["imgproc"] = timed(imgproc, args.repeats)
    stages["optical_flow"] = timed(flow, args.repeats)
    stages["orb_features"] = timed(features, args.repeats)
    stages["background_mog2"] = timed(background, args.repeats)

    net = cv2.dnn.readNetFromONNX(args.model)
    dnn_in = frames[: args.dnn_frames]

    def dnn():
        for f in dnn_in:
            blob = cv2.dnn.blobFromImage(f, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            net.forward()

    stages["dnn_yolo_640"] = timed(dnn, args.repeats)

    counts = {
        "decode": len(frames),
        "resize": len(frames),
        "imgproc": len(small),
        "optical_flow": len(small) - 1,
        "orb_features": min(40, len(frames)),
        "background_mog2": len(small),
        "dnn_yolo_640": len(dnn_in),
    }
    for name, stage in stages.items():
        stage["items"] = counts[name]
        stage["items_per_s"] = round(counts[name] / stage["median_s"], 2)

    report = {
        "label": args.label,
        "opencv_version": cv2.__version__,
        "opencv_file": cv2.__file__,
        "build": build_flags(),
        "cv_threads": cv2.getNumThreads(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "frame_size": f"{w}x{h}",
        "video": Path(args.video).name,
        "repeats": args.repeats,
        "stages": stages,
        "total_median_s": round(sum(s["median_s"] for s in stages.values()), 3),
    }

    if args.pipeline:
        from backend.app.config import settings
        from backend.app.perception.vocabulary import load_label_map
        from backend.app.pipeline.pipeline import Pipeline, build_default_detector

        detector = build_default_detector(
            model_path=args.model,
            mode="real",
            conf_threshold=settings.detection_confidence_threshold,
            label_map=load_label_map(settings.detector_label_map),
        )
        runs = []
        processed = 0
        for _ in range(max(1, args.repeats // 2)):
            start = time.perf_counter()
            result = Pipeline(detector=detector).run(args.video)
            runs.append(time.perf_counter() - start)
            processed = result.frames_processed
        report["pipeline"] = {
            "frames_processed": processed,
            "median_s": round(statistics.median(runs), 3),
            "throughput_fps": round(processed / statistics.median(runs), 2),
            "samples_s": [round(r, 3) for r in runs],
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("label", "opencv_version", "cpu", "total_median_s")}, indent=2))
    for name, stage in stages.items():
        print(f"  {name:16s} {stage['median_s']:8.3f}s  {stage['items_per_s']:9.1f}/s")
    if "pipeline" in report:
        print(f"  pipeline         {report['pipeline']['median_s']}s  {report['pipeline']['throughput_fps']} fps")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
