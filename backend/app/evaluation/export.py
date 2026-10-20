"""Export VISTA memory as per-frame tracks and score query grounding."""

from __future__ import annotations

from collections import Counter, defaultdict

from ..memory.memory import VisualMemory
from .mot_metrics import Box, Sequence, iou


def _box(obs) -> Box:
    b = obs.bounding_box
    return (b.x_min, b.y_min, b.x_max, b.y_max)


def tracks_by_frame(
    memory: VisualMemory,
    categories: set[str] | None = None,
    frame_offset: int = 0,
) -> Sequence:
    """``{frame: [(object_id, box)]}`` from persistent object memory.

    ``frame_offset`` converts VISTA's 0-based frame ids to the dataset's
    numbering (MOT17 is 1-based, so pass 1).
    """
    out: Sequence = defaultdict(list)
    for obj in memory.object_memory.all():
        if categories is not None and obj.category not in categories:
            continue
        for obs in obj.observations:
            out[obs.frame_id + frame_offset].append((obj.object_id, _box(obs)))
    return dict(out)


def locate_grounding(
    memory: VisualMemory,
    gt: Sequence,
    category: str,
    frame_offset: int = 0,
    sample_frames: set[int] | None = None,
    threshold: float = 0.5,
    fps: float = 30.0,
) -> dict:
    """Score the "where is / last seen" answer against ground truth.

    For every remembered object of ``category`` the claimed last sighting
    (frame, box) is *grounded* if it overlaps a ground-truth box in that
    frame (IoU >= ``threshold``). Each object is also associated with the
    ground-truth track it overlaps most, and the error of its claimed
    last-seen time against that track's last sampled appearance is
    reported.
    """
    sample_frames = sample_frames or set()
    grounded = total = 0
    time_errors: list[float] = []
    for obj in memory.object_memory.all():
        if obj.category != category or not obj.observations:
            continue
        total += 1
        last = max(obj.observations, key=lambda o: o.frame_id)
        frame = last.frame_id + frame_offset
        box = _box(last)
        if any(iou(box, b) >= threshold for _, b in gt.get(frame, [])):
            grounded += 1
        # Majority-overlap GT track -> compare last-seen timing.
        votes: Counter = Counter()
        for obs in obj.observations:
            f = obs.frame_id + frame_offset
            for gid, gb in gt.get(f, []):
                if iou(_box(obs), gb) >= threshold:
                    votes[gid] += 1
        if votes:
            gid, _ = votes.most_common(1)[0]
            gt_frames = [
                f for f in sample_frames if any(g == gid for g, _ in gt.get(f, []))
            ]
            if gt_frames:
                true_last = max(gt_frames)
                time_errors.append(abs(frame - true_last) / fps)
    time_errors.sort()
    median = time_errors[len(time_errors) // 2] if time_errors else None
    return {
        "objects": total,
        "grounded": grounded,
        "grounding_accuracy": round(grounded / total, 4) if total else None,
        "last_seen_median_error_s": round(median, 3) if median is not None else None,
        "last_seen_matched_objects": len(time_errors),
    }
