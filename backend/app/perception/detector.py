"""Object detection via a YOLO ONNX model run through ``cv2.dnn``.

Detection itself is external (a pre-trained model), but inference and all
post-processing (letterboxing, NMS, box de-normalisation) are performed with
OpenCV, keeping the detector inside the OpenCV pipeline (FR-003/FR-004).
"""

from __future__ import annotations

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


class YoloDetector:
    """Run a YOLOv8-family ONNX model through ``cv2.dnn``."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: list[str] | None = None,
    ) -> None:
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or COCO_CLASSES

    @staticmethod
    def _letterbox(
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
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, r, (left, top)

    def detect(
        self, image: np.ndarray, frame_id: int, timestamp: float
    ) -> list[Detection]:
        h, w = image.shape[:2]
        padded, ratio, (pad_x, pad_y) = self._letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        output = outputs[0][0].transpose(1, 0)  # (N, 4 + num_classes)

        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        keep = confidences > self.conf_threshold
        boxes, class_ids, confidences = boxes[keep], class_ids[keep], confidences[keep]

        if boxes.shape[0] == 0:
            return []

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * self.input_size - pad_x
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * self.input_size - pad_y
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * self.input_size - pad_x
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * self.input_size - pad_y
        xyxy /= ratio

        x_min = np.clip(xyxy[:, 0], 0, w)
        y_min = np.clip(xyxy[:, 1], 0, h)
        x_max = np.clip(xyxy[:, 2], 0, w)
        y_max = np.clip(xyxy[:, 3], 0, h)

        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), confidences.tolist(),
            self.conf_threshold, self.iou_threshold,
        )
        if indices is None or len(indices) == 0:
            return []

        detections: list[Detection] = []
        for i in np.asarray(indices).flatten():
            detections.append(
                Detection(
                    category=self.class_names[int(class_ids[i])],
                    bounding_box=BoundingBox(
                        x_min=float(x_min[i]),
                        y_min=float(y_min[i]),
                        x_max=float(x_max[i]),
                        y_max=float(y_max[i]),
                    ),
                    confidence=float(confidences[i]),
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
        return detections
