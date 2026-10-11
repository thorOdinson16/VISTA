"""VISTA session: single-session service holding memory + source video.

Replaces the previous scatter of global ``_STATE`` variables with one
explicit object per ingested video::

    API -> VistaSession -> memory -> source video -> evidence

Single-session for the MVP (no multi-user backend yet); the abstraction
exists so session state is isolated behind a clean service boundary.
"""

from __future__ import annotations

from ..ingestion.reader import VideoInfo
from ..memory.memory import VisualMemory


class VistaSession:
    def __init__(self) -> None:
        self.memory: VisualMemory | None = None
        self.source_video: str | None = None
        self.video_info: VideoInfo | None = None
        self.detector_mode: str | None = None
        self.detector_name: str | None = None
        # Labels requested by queries that were not in the configured
        # vocabulary; the detector is re-prompted with them on demand.
        self.extra_labels: list[str] = []

    @property
    def ready(self) -> bool:
        return self.memory is not None

    @property
    def fps(self) -> float:
        if self.video_info is not None and self.video_info.fps > 0:
            return self.video_info.fps
        return 30.0

    @property
    def duration(self) -> float | None:
        if self.video_info is not None:
            return self.video_info.duration
        return None

    def attach(
        self, memory: VisualMemory, source_video: str, info: VideoInfo
    ) -> None:
        self.memory = memory
        self.source_video = source_video
        self.video_info = info
        memory.set_source(source_video, self.fps)

    def reset(self) -> None:
        self.memory = None
        self.source_video = None
        self.video_info = None
        self.detector_mode = None
        self.detector_name = None
        self.extra_labels = []
