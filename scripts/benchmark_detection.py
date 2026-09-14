"""Benchmark a detector (A/B) against labeled boxes.

Ground-truth JSON format::

    {
      "images": [
        {"path": "data/bench/0001.jpg",
         "boxes": [{"label": "person", "bbox": [x1, y1, x2, y2]}]}
      ]
    }

Usage:

    uv run python scripts/benchmark_detection.py --gt data/bench/gt.json
    uv run python scripts/benchmark_detection.py --gt gt.json --mode openvocab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import settings
from backend.app.evaluation.detection import (
    evaluate_detection,
    format_report,
    load_dataset,
)
from backend.app.perception.vocabulary import load_label_map
from backend.app.pipeline.pipeline import build_default_detector


def main() -> int:
    parser = argparse.ArgumentParser(description="VISTA detection benchmark")
    parser.add_argument("--gt", required=True, help="ground-truth JSON")
    parser.add_argument("--mode", default=settings.detector_mode)
    parser.add_argument("--model", default=settings.yolo_model_path)
    parser.add_argument(
        "--conf", type=float, default=settings.detection_confidence_threshold
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--json", default=None, help="write report as JSON")
    args = parser.parse_args()

    items = load_dataset(args.gt)
    if not items:
        print("No images in ground truth.", file=sys.stderr)
        return 1

    detector = build_default_detector(
        model_path=args.model,
        mode=args.mode,
        models_dir=settings.models_dir,
        conf_threshold=args.conf,
        class_names_path=settings.yolo_class_names_path,
        label_map=load_label_map(settings.detector_label_map),
        labels=settings.detector_label_list,
        openvocab_model=settings.openvocab_model,
    )
    print(f"Detector: {getattr(detector, 'name', type(detector).__name__)}")
    report = evaluate_detection(detector, items, iou_threshold=args.iou)
    print(format_report(report))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
