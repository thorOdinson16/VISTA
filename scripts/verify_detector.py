"""Run a detector on an image or video and print what it finds.

This is the "does the real detector actually run" check. Example:

    uv run python scripts/verify_detector.py --source data/sample.jpg
    uv run python scripts/verify_detector.py --source data/clip.mp4 --frames 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from backend.app.pipeline.pipeline import build_default_detector, detector_name


def main() -> int:
    parser = argparse.ArgumentParser(description="VISTA detector smoke test")
    parser.add_argument("--source", required=True, help="image or video path")
    parser.add_argument("--mode", default=settings.detector_mode)
    parser.add_argument("--model", default=settings.yolo_model_path)
    parser.add_argument(
        "--conf", type=float, default=settings.detection_confidence_threshold
    )
    parser.add_argument("--frames", type=int, default=3, help="frames for video")
    args = parser.parse_args()

    detector = build_default_detector(
        model_path=args.model,
        mode=args.mode,
        models_dir=settings.models_dir,
        conf_threshold=args.conf,
        class_names_path=settings.yolo_class_names_path,
    )
    print(f"Detector: {detector_name(detector)}")

    suffix = Path(args.source).suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Could not open video: {args.source}", file=sys.stderr)
            return 1
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, n // max(1, args.frames))
        total = 0
        for i in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            fake_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            detections = detector.detect(frame, i, i / fake_fps)
            total += len(detections)
            print(f"frame {i}: {len(detections)} detections")
            for d in detections:
                print(
                    f"   {d.category:>12}  conf={d.confidence:.2f}  "
                    f"box=({d.bounding_box.x_min:.0f},{d.bounding_box.y_min:.0f},"
                    f"{d.bounding_box.x_max:.0f},{d.bounding_box.y_max:.0f})"
                )
        cap.release()
        print(f"Total detections: {total}")
        return 0

    image = cv2.imread(args.source)
    if image is None:
        print(f"Could not read image: {args.source}", file=sys.stderr)
        return 1
    detections = detector.detect(image, 0, 0.0)
    print(f"{len(detections)} detections")
    for d in detections:
        print(f"   {d.category}: conf={d.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
