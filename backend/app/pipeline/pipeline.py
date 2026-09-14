"""End-to-end processing pipeline: video -> memory.

Wires ingestion, detection, tracking, and visual memory into a single runnable
flow, optionally writing evidence frames to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..ingestion.reader import VideoInfo, VideoReader
from ..ingestion.sampler import FrameSampler
from ..memory.memory import VisualMemory
from ..memory.scene_change import SceneChangeDetector
from ..perception.color_detector import ColorDetector
from ..perception.detector import YoloDetector
from ..perception.motion import MotionAnalyzer
from ..perception.saliency import SaliencyConfirmer, SalientObjectDetector
from ..perception.vocabulary import LabelMapper, MappedDetector, default_label_map
from ..tracking.tracker import MultiObjectTracker

# Deterministic scene poster written on every ingest (see write_evidence_frames).
POSTER_FRAME = "scene.jpg"


@dataclass
class PipelineResult:
    memory: VisualMemory
    video_info: VideoInfo | None = None
    evidence_refs: list[dict] = field(default_factory=list)
    frames_processed: int = 0
    frames_sampled: int = 0


class Pipeline:
    def __init__(
        self,
        detector,
        tracker: MultiObjectTracker | None = None,
        sampler: FrameSampler | None = None,
        fps: float = 30.0,
        evidence_dir: Path | None = None,
        store_path: str | Path | None = None,
        saliency: SalientObjectDetector | None = None,
        saliency_fallback: bool = False,
        saliency_min_area: int = 600,
        saliency_min_overlap: float = 0.1,
        scene_change_threshold: float = 0.15,
        scene_change_motion_gate: float = 2.0,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or MultiObjectTracker()
        self.sampler = sampler
        self.fps = fps  # fallback only; the source video's FPS wins
        self.evidence_dir = evidence_dir
        self.store_path = store_path
        self.saliency = saliency if saliency is not None else (
            SalientObjectDetector(min_area=saliency_min_area)
            if saliency_fallback
            else None
        )
        self.saliency_confirmer = SaliencyConfirmer(min_overlap=saliency_min_overlap)
        self.scene_change_threshold = scene_change_threshold
        self.scene_change_motion_gate = scene_change_motion_gate

    def run(self, video_path: str | Path) -> PipelineResult:
        video_path = Path(video_path)
        with VideoReader(str(video_path)) as reader:
            info = reader.info
            fps = info.fps or self.fps
            motion = MotionAnalyzer()
            sampler = self.sampler or FrameSampler(
                strategy="adaptive",
                sample_fps=2.0,
                source_fps=fps,
                motion=motion,
            )
            scene_changes = SceneChangeDetector(
                change_threshold=self.scene_change_threshold,
                motion_gate=self.scene_change_motion_gate,
            )
            memory = VisualMemory(
                video_duration=info.duration,
                store_path=self.store_path,
                source_video=str(video_path),
                fps=fps,
            )

            frames_processed = 0
            frames_sampled = 0
            for frame in reader.frames():
                frames_processed += 1
                foreground_mask, foreground_score = motion.foreground(frame.image)
                sampled = sampler.decide(frame)
                scene_changes.observe(
                    frame.frame_id,
                    frame.timestamp,
                    foreground_score,
                    motion.last_flow_magnitude,
                )
                if not sampled:
                    continue
                frames_sampled += 1
                detections = self.detector.detect(
                    frame.image, frame.frame_id, frame.timestamp
                )
                if detections:
                    self.saliency_confirmer.reset()
                elif (
                    self.saliency is not None
                    and motion.last_flow_magnitude <= self.scene_change_motion_gate
                ):
                    proposals = self.saliency.propose(
                        foreground_mask,
                        frame.frame_id,
                        frame.timestamp,
                        frame.image.shape[:2],
                    )
                    # A region must persist across two sampled frames before
                    # it is trusted as an object.
                    detections = self.saliency_confirmer.confirm(proposals)
                tracks = self.tracker.update(
                    detections, frame.frame_id, frame.timestamp, frame.image
                )
                memory.update(frame.frame_id, frame.timestamp, tracks)

            for event in scene_changes.events():
                memory.event_memory.add(event)
            memory.finalize()
            if memory.store is not None:
                memory.save()

            evidence_refs: list[dict] = []
            if self.evidence_dir is not None:
                evidence_refs = self.write_evidence_frames(
                    video_path, memory, self.evidence_dir, fps=fps
                )

        return PipelineResult(
            memory=memory,
            video_info=info,
            evidence_refs=evidence_refs,
            frames_processed=frames_processed,
            frames_sampled=frames_sampled,
        )

    @staticmethod
    def write_evidence_frames(
        video_path: str | Path,
        memory: VisualMemory,
        evidence_dir: Path,
        fps: float | None = None,
    ) -> list[dict]:
        """Persist representative evidence frames for events and objects.

        Event clips are keyed by ``event_id``; object thumbnails are keyed
        by ``object_id`` and use the object's last observation frame so the
        UI can show an object's history. Frame indices always derive from
        the source video's actual FPS.
        """
        evidence_dir.mkdir(parents=True, exist_ok=True)
        refs: list[dict] = []
        with VideoReader(str(video_path)) as reader:
            actual_fps = reader.info.fps or fps or 30.0
            # Always write one scene poster (middle frame) so the UI can
            # show "here is the frame" even when nothing was detected.
            poster_frame = max(0, reader.info.frame_count // 2)
            poster = reader.read_frame(poster_frame)
            if poster is not None:
                cv2.imwrite(str(evidence_dir / POSTER_FRAME), poster)
                refs.append(
                    {
                        "kind": "scene",
                        "timestamp": poster_frame / actual_fps,
                        "frame_id": poster_frame,
                        "image": POSTER_FRAME,
                    }
                )
            for event in memory.event_memory.all():
                frame_id = int(round(event.start_time * actual_fps))
                image = reader.read_frame(frame_id)
                if image is None:
                    continue
                name = f"{event.event_id}.jpg"
                cv2.imwrite(str(evidence_dir / name), image)
                refs.append(
                    {
                        "kind": "event",
                        "event_id": event.event_id,
                        "timestamp": event.start_time,
                        "frame_id": frame_id,
                        "image": name,
                    }
                )
            for obj in memory.object_memory.all():
                if obj.last_seen is None:
                    continue
                frame_id = int(round(obj.last_seen * actual_fps))
                image = reader.read_frame(frame_id)
                if image is None:
                    continue
                name = f"{obj.object_id}.jpg"
                cv2.imwrite(str(evidence_dir / name), image)
                refs.append(
                    {
                        "kind": "object",
                        "object_id": obj.object_id,
                        "timestamp": obj.last_seen,
                        "frame_id": frame_id,
                        "image": name,
                    }
                )
        return refs


def _find_model(
    model_path: str | Path | None, models_dir: str | Path | None
) -> Path | None:
    if model_path and Path(model_path).exists():
        return Path(model_path)
    if models_dir:
        candidates = sorted(Path(models_dir).glob("*.onnx"))
        if candidates:
            return candidates[0]
    return None


def _load_class_names(path: str | Path | None) -> list[str] | None:
    if not path or not Path(path).exists():
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return None
    if text.startswith("["):
        import json

        return [str(name) for name in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def detector_name(detector) -> str:
    return getattr(detector, "name", type(detector).__name__)


def build_default_detector(
    model_path: str | None = None,
    mode: str = "real",
    models_dir: str | Path | None = None,
    conf_threshold: float = 0.25,
    class_names_path: str | Path | None = None,
    label_map: dict[str, str] | None = None,
    labels: list[str] | None = None,
    openvocab_model: str = "yolov8s-worldv2.pt",
) -> object:
    """Build a detector for one of the explicit modes.

    * ``synthetic`` -- :class:`ColorDetector` for deterministic tests and
      the generated demo video. Never the conceptual implementation.
    * ``real`` -- :class:`YoloDetector` (YOLO ONNX via ``cv2.dnn``) for
      real footage. When no explicit model path is given, the first
      ``*.onnx`` in ``models_dir`` is used.
    * ``openvocab`` -- :class:`YoloWorldDetector` (Ultralytics YOLO-World)
      detecting the configured text ``labels`` (e.g. ``coin``, ``shelf``).
    * ``hybrid`` -- ``real`` first, falling back to ``openvocab`` when the
      primary finds nothing.

    A missing model in ``real`` mode is an explicit error: the system never
    silently downgrades to the synthetic detector for real footage.
    """
    if mode == "synthetic":
        return ColorDetector()

    mapper = LabelMapper(label_map or default_label_map())

    def _real() -> object:
        found = _find_model(model_path, models_dir)
        if found is None:
            raise ValueError(
                "detector_mode='real' but no YOLO ONNX model was found. "
                "Run `uv run python scripts/fetch_model.py` or set "
                "VISTA_YOLO_MODEL_PATH / place a model in data/models/."
            )
        detector = YoloDetector(
            str(found),
            conf_threshold=conf_threshold,
            class_names=_load_class_names(class_names_path),
        )
        return MappedDetector(detector, mapper)

    def _openvocab() -> object:
        from ..perception.openvocab import YoloWorldDetector

        detector = YoloWorldDetector(
            labels or [],
            model=openvocab_model,
            conf_threshold=conf_threshold,
        )
        return MappedDetector(detector, mapper)

    if mode == "real":
        return _real()
    if mode == "openvocab":
        return _openvocab()
    if mode == "hybrid":
        from ..perception.openvocab import CompositeDetector

        return CompositeDetector(_real(), _openvocab())
    raise ValueError(f"Unknown detector mode: {mode!r}")
