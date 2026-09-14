"""Object detection via a YOLO ONNX model run through ``cv2.dnn``.

Detection itself is external (a pre-trained model), but inference and all
post-processing (letterboxing, decoding, class-aware NMS, box
de-normalisation) are performed with OpenCV, keeping the detector inside the
OpenCV pipeline (FR-003/FR-004).

The decoder is a pure function (:func:`decode_yolo`) so it can be unit
tested from synthetic tensors without a model file, and it supports the
YOLOv8/v11 detection layout ``(1, 4+C, N)`` (no objectness column). Models
with a different layout raise a clear error instead of silently producing
garbage.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..memory.models import BoundingBox, Detection

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def letterbox(
    image: np.ndarray, new_shape: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape - new_w, new_shape - new_h
    left = int(round(dw - 0.1)) // 2
    top = int(round(dh - 0.1)) // 2
    right = dw - left
    bottom = dh - top
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, r, (left, top)


def decode_yolo(
    output: np.ndarray,
    input_size: int,
    ratio: float,
    pad: tuple[int, int],
    frame_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    class_names: list[str],
    frame_id: int = 0,
    timestamp: float = 0.0,
) -> list[Detection]:
    """Decode raw YOLOv8/v11 detection output into detections.

    ``output`` is the raw network output; accepted shapes are
    ``(1, 4+C, N)``, ``(4+C, N)`` or ``(N, 4+C)``. Class-aware NMS is
    applied (boxes are converted to ``x, y, w, h`` because that is what
    ``cv2.dnn.NMSBoxes`` expects). Raises ``ValueError`` when the layout or
    class count is unsupported, rather than returning wrong boxes.
    """
    if output.ndim == 3:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(
            f"Unsupported YOLO output shape {output.shape}; expected 2D/3D "
            "(YOLOv8/v11 detection output)."
        )
    # Orient to (N, 4 + C) using the class count, which is unambiguous.
    expected_cols = 4 + len(class_names)
    if output.shape[1] == expected_cols:
        pass
    elif output.shape[0] == expected_cols:
        output = output.T
    else:
        raise ValueError(
            f"Unsupported YOLO output shape {output.shape} for "
            f"{len(class_names)} class names; expected 4 + num_classes "
            "columns (YOLOv8/v11 detection output, no objectness column)."
        )

    boxes = output[:, :4].astype(np.float32)
    scores = output[:, 4:].astype(np.float32)
    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)

    keep = confidences > conf_threshold
    if not np.any(keep):
        return []
    boxes, class_ids, confidences = boxes[keep], class_ids[keep], confidences[keep]

    fh, fw = frame_shape[:2]
    pad_x, pad_y = pad
    # Ultralytics YOLOv8/v11 exports boxes in input-pixel space (0..imgsz);
    # some exports emit normalised (0..1) boxes. Detect and normalise.
    box_scale = input_size if float(np.max(boxes[:, :4])) <= 4.0 else 1.0
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * box_scale - pad_x
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * box_scale - pad_y
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * box_scale - pad_x
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * box_scale - pad_y
    xyxy /= ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, fw)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, fh)

    detections: list[Detection] = []
    for class_id in np.unique(class_ids):
        idx = np.where(class_ids == class_id)[0]
        class_boxes = xyxy[idx]
        # NMSBoxes expects (x, y, width, height).
        nms_input = np.column_stack(
            [
                class_boxes[:, 0],
                class_boxes[:, 1],
                class_boxes[:, 2] - class_boxes[:, 0],
                class_boxes[:, 3] - class_boxes[:, 1],
            ]
        )
        kept = cv2.dnn.NMSBoxes(
            nms_input.tolist(),
            confidences[idx].tolist(),
            float(conf_threshold),
            float(iou_threshold),
        )
        if kept is None or len(kept) == 0:
            continue
        for local in np.asarray(kept).flatten():
            i = int(idx[int(local)])
            x1, y1, x2, y2 = xyxy[i]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    category=class_names[int(class_ids[i])],
                    bounding_box=BoundingBox(
                        x_min=float(x1),
                        y_min=float(y1),
                        x_max=float(x2),
                        y_max=float(y2),
                    ),
                    confidence=float(confidences[i]),
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
    return detections


class YoloDetector:
    """Run a YOLOv8/v11-family ONNX model through ``cv2.dnn``."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: list[str] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or COCO_CLASSES
        self.name = f"YOLO ONNX ({Path(self.model_path).name})"

    def detect(
        self, image: np.ndarray, frame_id: int, timestamp: float
    ) -> list[Detection]:
        padded, ratio, pad = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        return decode_yolo(
            outputs,
            self.input_size,
            ratio,
            pad,
            image.shape[:2],
            self.conf_threshold,
            self.iou_threshold,
            self.class_names,
            frame_id=frame_id,
            timestamp=timestamp,
        )
