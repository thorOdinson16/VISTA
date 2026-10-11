"""Open-vocabulary detection (B) and detector composition.

COCO covers 80 everyday classes; the project needs arbitrary nouns
(``coin``, ``shelf``, ...). :class:`YoloWorldDetector` wraps Ultralytics
YOLO-World, which detects a caller-supplied text vocabulary. The
``ultralytics`` dependency is optional and imported lazily so the rest of
VISTA (and the test suite) runs without it.

:class:`CompositeDetector` runs a primary detector and falls back to a
secondary one (e.g. open-vocabulary) when the primary finds nothing.
"""

from __future__ import annotations

from ..memory.models import BoundingBox, Detection


class YoloWorldDetector:
    """Ultralytics YOLO-World with a configurable text vocabulary."""

    def __init__(
        self,
        labels: list[str],
        model: str = "yolov8s-worldv2.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ultralytics is required for open-vocabulary detection. "
                "Install it with: uv sync --group detection"
            ) from exc
        if not labels:
            raise ValueError("YoloWorldDetector requires at least one label")
        self.labels = list(labels)
        self.model_name = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = YOLOWorld(model)
        self._model.set_classes(self.labels)
        self.name = f"YOLO-World ({model}, {len(self.labels)} labels)"

    def detect(
        self, image, frame_id: int, timestamp: float
    ) -> list[Detection]:
        results = self._model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    category=names[int(box.cls[0])],
                    bounding_box=BoundingBox(
                        x_min=x1, y_min=y1, x_max=x2, y_max=y2
                    ),
                    confidence=float(box.conf[0]),
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
        return detections


class Owlv2Detector:
    """OWLv2 open-vocabulary detector (HuggingFace ``transformers``).

    Markedly stronger than YOLO-World on small, thin or low-contrast
    objects (a charger and its cable on a patterned sheet, for instance).
    Slower (~2s/frame on CPU) but scores each text prompt independently, so
    a ``phone charger`` is not forced into the nearest ``phone`` class.
    ``transformers`` and ``torch`` are imported lazily.
    """

    PROMPT = "a photo of a {}"

    def __init__(
        self,
        labels: list[str],
        model: str = "google/owlv2-base-patch16-ensemble",
        conf_threshold: float = 0.2,
        nms_iou: float = 0.5,
    ) -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required for detector_mode='owlv2'. "
                "Install it with: uv sync --group owlv2"
            ) from exc
        if not labels:
            raise ValueError("Owlv2Detector requires at least one label")
        self._torch = torch
        self.labels = list(dict.fromkeys(labels))
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self._processor = Owlv2Processor.from_pretrained(model)
        self._model = Owlv2ForObjectDetection.from_pretrained(model).eval()
        self._texts = [[self.PROMPT.format(label) for label in self.labels]]
        self.name = f"OWLv2 ({model.split('/')[-1]}, {len(self.labels)} labels)"

    def detect(
        self, image, frame_id: int, timestamp: float
    ) -> list[Detection]:
        import cv2
        from PIL import Image

        height, width = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(
            text=self._texts, images=pil, return_tensors="pt"
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        # OWLv2 pads to a square; boxes come back in padded-square pixels.
        side = max(width, height)
        results = self._processor.image_processor.post_process_object_detection(
            outputs,
            threshold=self.conf_threshold,
            target_sizes=self._torch.tensor([[side, side]]),
        )[0]

        boxes, scores, labels = [], [], []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(score))
            labels.append(self.labels[int(label)])
        if not boxes:
            return []
        # Class-agnostic NMS: one physical object, one label.
        keep = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.nms_iou
        )
        detections: list[Detection] = []
        for index in (int(i) for i in (keep.flatten() if len(keep) else [])):
            x, y, w, h = boxes[index]
            detections.append(
                Detection(
                    category=labels[index],
                    bounding_box=BoundingBox(
                        x_min=x, y_min=y, x_max=x + w, y_max=y + h
                    ),
                    confidence=scores[index],
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )
        return detections


class CompositeDetector:
    """Primary detector, with an optional fallback when it finds nothing."""

    def __init__(self, primary, secondary=None) -> None:
        self.primary = primary
        self.secondary = secondary
        names = [getattr(primary, "name", type(primary).__name__)]
        if secondary is not None:
            names.append(getattr(secondary, "name", type(secondary).__name__))
        self.name = " -> ".join(names) + (
            " (fallback)" if secondary is not None else ""
        )

    def detect(self, image, frame_id: int, timestamp) -> list[Detection]:
        detections = self.primary.detect(image, frame_id, timestamp)
        if detections or self.secondary is None:
            return detections
        return self.secondary.detect(image, frame_id, timestamp)
