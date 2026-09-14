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
