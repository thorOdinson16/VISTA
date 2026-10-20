"""Detection benchmark: precision/recall/mAP@0.5 against labeled boxes.

Ground truth format (JSON)::

    {
      "images": [
        {"path": "frames/0001.jpg",
         "boxes": [{"label": "person", "bbox": [x1, y1, x2, y2]}, ...]},
        ...
      ]
    }

``bbox`` coordinates are in source-image pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..memory.models import BoundingBox


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def load_dataset(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("images", []))


def _gt_boxes(item: dict) -> list[tuple[str, BoundingBox]]:
    out = []
    for entry in item.get("boxes", []):
        x1, y1, x2, y2 = entry["bbox"]
        out.append(
            (
                entry["label"],
                BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
            )
        )
    return out


def average_precision(predictions: list[tuple[float, bool]], total_gt: int) -> float:
    """All-point-interpolated AP from confidence-sorted match flags."""
    if total_gt == 0:
        return 0.0
    predictions = sorted(predictions, key=lambda p: -p[0])
    tp = fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, hit in predictions:
        if hit:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)
    # Precision envelope (monotonic decreasing recalls).
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls, strict=False):
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def evaluate_detection(
    detector,
    items: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Run ``detector`` over labeled images and compute P/R/mAP@0.5."""
    per_class_preds: dict[str, list[tuple[float, bool]]] = {}
    gt_counts: dict[str, int] = {}
    total_tp = total_fp = total_fn = 0

    for image_id, item in enumerate(items):
        image = cv2.imread(item["path"])
        if image is None:
            continue
        predictions = detector.detect(image, image_id, 0.0)
        ground_truth = _gt_boxes(item)
        for label, _ in ground_truth:
            gt_counts[label] = gt_counts.get(label, 0) + 1

        matched: set[int] = set()
        for pred in sorted(predictions, key=lambda d: -d.confidence):
            best_iou = 0.0
            best_idx = -1
            for idx, (label, box) in enumerate(ground_truth):
                if idx in matched or label != pred.category:
                    continue
                iou = _iou(pred.bounding_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            hit = best_iou >= iou_threshold
            if hit:
                matched.add(best_idx)
                total_tp += 1
            else:
                total_fp += 1
            per_class_preds.setdefault(pred.category, []).append((pred.confidence, hit))
        total_fn += len(ground_truth) - len(matched)

    aps = {
        label: average_precision(per_class_preds.get(label, []), count)
        for label, count in gt_counts.items()
    }
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return {
        "images": len(items),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "map50": round(sum(aps.values()) / len(aps), 3) if aps else 0.0,
        "per_class_ap": {k: round(v, 3) for k, v in aps.items()},
    }


def format_report(report: dict) -> str:
    lines = [
        f"Images: {report['images']}",
        f"Precision: {report['precision']}  Recall: {report['recall']}",
        f"mAP@0.5: {report['map50']}",
        f"TP={report['true_positives']} FP={report['false_positives']} "
        f"FN={report['false_negatives']}",
        "Per-class AP@0.5:",
    ]
    for label, ap in sorted(report["per_class_ap"].items()):
        lines.append(f"  {label}: {ap}")
    return "\n".join(lines)


__all__ = [
    "average_precision",
    "evaluate_detection",
    "format_report",
    "load_dataset",
]
