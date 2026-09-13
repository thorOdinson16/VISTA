"""Export a YOLO model to ONNX for ``cv2.dnn`` inference.

Run with the detection dependency group:

    uv run --group detection python scripts/export_onnx.py --model yolov8s
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--model", default="yolov8s")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=None, help="destination path")
    args = parser.parse_args()

    model = YOLO(f"{args.model}.pt")
    exported = model.export(
        format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True
    )

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), dest)
        print(f"Exported {args.model} -> {dest}")
    else:
        print(f"Exported {args.model} -> {exported}")


if __name__ == "__main__":
    main()
