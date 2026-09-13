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
from ..tracking.tracker import MultiObjectTracker


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
    ) -> None:
        self.detector = detector
        self.tracker = tracker or MultiObjectTracker()
        self.sampler = sampler
        self.fps = fps  # fallback only; the source video's FPS wins
        self.evidence_dir = evidence_dir
        self.store_path = store_path

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
            scene_changes = SceneChangeDetector()
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
                _, foreground_score = motion.foreground(frame.image)
                scene_changes.observe(
                    frame.frame_id, frame.timestamp, foreground_score
                )
                if not sampler.decide(frame):
                    continue
                frames_sampled += 1
                detections = self.detector.detect(
                    frame.image, frame.frame_id, frame.timestamp
                )
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


def build_default_detector(
    model_path: str | None = None,
    mode: str = "synthetic",
    models_dir: str | Path | None = None,
) -> object:
    """Build a detector for one of two explicit modes.

    * ``synthetic`` -- :class:`ColorDetector` for deterministic tests and
      the generated demo video. Never the conceptual implementation.
    * ``real`` -- :class:`YoloDetector` (YOLO ONNX via ``cv2.dnn``) for
      real footage. When no explicit model path is given, the first
      ``*.onnx`` in ``models_dir`` is used; if none is found, falls back
      to synthetic with a warning.
    """
    if mode == "real":
        found = _find_model(model_path, models_dir)
        if found is not None:
            return YoloDetector(str(found))
        import warnings

        warnings.warn(
            "Real detector requested but no YOLO model found; "
            "falling back to the synthetic ColorDetector.",
            stacklevel=2,
        )
    return ColorDetector()
