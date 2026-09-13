"""Scene-change detection from OpenCV MOG2 foreground scores.

The pipeline feeds one foreground-ratio sample per decoded frame (the
fraction of pixels the background subtractor flags as foreground). A
sustained shift away from a slowly-adapting baseline indicates that the
scene itself changed (lighting, camera shift, or a large new element),
rather than an object merely moving.

Detection uses the same jitter-gate philosophy as ``movement.py``: a
single-frame spike is not a change; the deviation must persist for a
minimum number of consecutive frames before a ``SCENE_CHANGED`` event is
emitted. The baseline resets after each event so one sustained change
produces exactly one event.
"""

from __future__ import annotations

from .models import EventRecord, EventType


class SceneChangeDetector:
    def __init__(
        self,
        change_threshold: float = 0.15,
        persistence: int = 2,
        baseline_alpha: float = 0.1,
        min_confidence: float = 0.5,
        max_confidence: float = 1.0,
    ) -> None:
        self.change_threshold = change_threshold
        self.persistence = max(1, persistence)
        self.baseline_alpha = baseline_alpha
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self._baseline: float | None = None
        self._streak = 0
        self._streak_start: tuple[int, float] | None = None
        self._events: list[EventRecord] = []

    def observe(self, frame_id: int, timestamp: float, score: float) -> None:
        """Feed one foreground-ratio sample (0..1) for a decoded frame."""
        if self._baseline is None:
            self._baseline = score
            return

        delta = abs(score - self._baseline)
        if delta >= self.change_threshold:
            self._streak += 1
            if self._streak == 1:
                self._streak_start = (frame_id, timestamp)
            if self._streak >= self.persistence:
                self._emit(delta, frame_id, timestamp)
                self._baseline = score
                self._streak = 0
                self._streak_start = None
            return

        self._streak = 0
        self._streak_start = None
        self._baseline = (
            (1.0 - self.baseline_alpha) * self._baseline
            + self.baseline_alpha * score
        )

    def _emit(self, delta: float, frame_id: int, timestamp: float) -> None:
        start_time = (
            self._streak_start[1] if self._streak_start is not None else timestamp
        )
        confidence = min(
            self.max_confidence,
            max(self.min_confidence, self.min_confidence + delta),
        )
        self._events.append(
            EventRecord(
                type=EventType.SCENE_CHANGED,
                start_time=start_time,
                subject_refs=[],
                confidence=confidence,
                kind="observed",
                details={
                    "foreground_delta": round(delta, 3),
                    "frame_id": frame_id,
                },
            )
        )

    def events(self) -> list[EventRecord]:
        return list(self._events)
