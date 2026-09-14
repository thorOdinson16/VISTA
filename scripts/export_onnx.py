"""Export a YOLO model to ONNX for ``cv2.dnn`` inference.

Run with the detection dependency group:

    uv run --group detection python scripts/export_onnx.py --model yolov8s
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_model(
    model_name: str = "yolov8s", imgsz: int = 640, out: str | None = None
) -> Path:
    """Export ``<model_name>.pt`` to ONNX and return the resulting path."""
    from ultralytics import YOLO

    model = YOLO(f"{model_name}.pt")
    exported = Path(
        model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    )
    if out:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), dest)
        return dest
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--model", default="yolov8s")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=None, help="destination path")
    args = parser.parse_args()

    dest = export_model(args.model, args.imgsz, args.out)
    if args.out:
        print(f"Exported {args.model} -> {dest}")
    else:
        print(f"Exported {args.model} -> {dest}")


if __name__ == "__main__":
    main()
