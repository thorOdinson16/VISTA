"""Video ingestion: decode frames with preserved timestamps and indices."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


@dataclass(frozen=True)
class Frame:
    frame_id: int
    timestamp: float
    image: np.ndarray
    source_id: str


class VideoReader:
    """Thin wrapper over ``cv2.VideoCapture`` exposing frames with metadata.

    Rejects unopenable/corrupted inputs with a clear ``ValueError`` so the
    caller can handle ingestion failures gracefully (FR-001).
    """

    def __init__(self, path: str, source_id: str | None = None) -> None:
        self.path = str(path)
        self.source_id = source_id or self.path
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"Could not open video: {self.path}")

        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = (self._frame_count / self._fps) if self._fps > 0 else 0.0

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(
            path=self.path,
            fps=self._fps,
            width=self._width,
            height=self._height,
            frame_count=self._frame_count,
            duration=self._duration,
        )

    def _timestamp(self, frame_id: int) -> float:
        return frame_id / self._fps if self._fps > 0 else float(frame_id)

    def frames(self):
        frame_id = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(
                frame_id=frame_id,
                timestamp=self._timestamp(frame_id),
                image=image,
                source_id=self.source_id,
            )
            frame_id += 1

    def seek(self, frame_id: int) -> bool:
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))

    def read_frame(self, frame_id: int) -> np.ndarray | None:
        self.seek(frame_id)
        ok, image = self._cap.read()
        return image if ok else None

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
