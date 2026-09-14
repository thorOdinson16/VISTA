"""Fetch/export a YOLO ONNX detector into ``data/models``.

Downloads the YOLO weights via ultralytics (network) and exports them to
ONNX for ``cv2.dnn``. Run once before the first ingest when
``VISTA_DETECTOR_MODE=real``:

    uv sync --group detection
    uv run python scripts/fetch_model.py --model yolov8s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_onnx import export_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a YOLO ONNX model")
    parser.add_argument("--model", default="yolov8s")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--out", default=None, help="destination (default data/models/<model>.onnx)"
    )
    args = parser.parse_args()

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ultralytics is not installed. Install the detection group first:\n"
            "    uv sync --group detection",
            file=sys.stderr,
        )
        return 2

    out = args.out or f"data/models/{args.model}.onnx"
    dest = export_model(args.model, args.imgsz, out)
    print(f"Wrote {dest}")
    print("VISTA will auto-discover the first data/models/*.onnx in real mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
