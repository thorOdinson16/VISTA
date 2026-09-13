"""Segmented movement detection over object trajectories.

Replaces the previous whole-trajectory-vs-first-point comparison (which
produced one broad event per object) with interval segmentation so the
system can represent ``stationary -> moving -> stationary`` as meaningful
movement intervals with start/end times, source/destination positions,
confidence, and evidence.

Detector jitter is rejected by two gates: a per-interval minimum
displacement and a minimum duration (temporal persistence). A brief spike
that does not persist is not movement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import TrajectoryPoint


@dataclass
class MovementSegment:
    start_time: float
    end_time: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    displacement: float
    duration: float
    confidence: float


class MovementSegmenter:
    """Split a trajectory into stationary/moving intervals."""

    def __init__(
        self,
        min_displacement: float = 8.0,
        min_duration: float = 1.0,
        moved_threshold: float = 50.0,
        merge_gap: float = 1.0,
    ) -> None:
        self.min_displacement = min_displacement
        self.min_duration = min_duration
        self.moved_threshold = moved_threshold
        self.merge_gap = merge_gap

    def segment(self, trajectory: list[TrajectoryPoint]) -> list[MovementSegment]:
        pts = sorted(trajectory, key=lambda p: p.timestamp)
        if len(pts) < 2:
            return []
        # Label each consecutive interval as moving when its displacement
        # clears the jitter gate.
        import math

        moving = []
        for a, b in zip(pts, pts[1:], strict=False):
            d = math.hypot(b.x - a.x, b.y - a.y)
            moving.append(d >= self.min_displacement)

        # Collect runs of moving intervals.
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, m in enumerate(moving):
            if m and start is None:
                start = i
            elif not m and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(moving)))

        # Merge runs separated by only a brief stationary gap.
        merged: list[tuple[int, int]] = []
        for s, e in runs:
            if not merged:
                merged.append((s, e))
                continue
            gap = pts[s].timestamp - pts[merged[-1][1]].timestamp
            if gap <= self.merge_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        segments: list[MovementSegment] = []
        for s, e in merged:
            p0, p1 = pts[s], pts[e]
            duration = p1.timestamp - p0.timestamp
            disp = math.hypot(p1.x - p0.x, p1.y - p0.y)
            # Temporal persistence + significance gates: short or tiny
            # runs are jitter, not movement.
            if duration < self.min_duration:
                continue
            if disp < self.moved_threshold:
                continue
            strength = min(1.0, disp / (2.0 * self.moved_threshold))
            persistence = min(1.0, duration / (2.0 * self.min_duration))
            confidence = 0.5 + 0.5 * (0.6 * strength + 0.4 * persistence)
            segments.append(
                MovementSegment(
                    start_time=p0.timestamp,
                    end_time=p1.timestamp,
                    start_x=p0.x,
                    start_y=p0.y,
                    end_x=p1.x,
                    end_y=p1.y,
                    displacement=disp,
                    duration=duration,
                    confidence=float(confidence),
                )
            )
        return segments
