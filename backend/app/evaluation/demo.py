"""Synthetic demo scene from requirements section 15.

Renders a controlled scene (notebook, mug, Person A) with scripted
movement plus a ground-truth event list for evaluation. Objects are solid
colored shapes so the ``ColorDetector`` locates them end-to-end without a
trained model, keeping the demo deterministic and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# BGR colors. Furniture uses desaturated grays (S~0) so they never match
# the object palette in the ColorDetector.
COLORS = {
    "background": (240, 240, 240),
    "desk": (190, 190, 190),
    "shelf": (150, 150, 150),
    "occluder": (60, 60, 60),
    "notebook": (0, 0, 220),  # red
    "mug": (220, 0, 0),  # blue
    "person": (200, 0, 200),  # magenta
}

NOTEBOOK_SIZE = (60, 40)
MUG_SIZE = (40, 50)
PERSON_SIZE = (50, 140)

# (time_seconds, value) keyframes, linearly interpolated.
PERSON_X = [
    (0, (-60,)),
    (35, (200,)),
    (48, (420,)),
    (60, (560,)),
    (70, (635,)),
    (77, (720,)),
    (84, (1000,)),
    (110, (1150,)),
    (120, (1340,)),
]
NOTEBOOK = [
    (0, (400, 520)),
    (48, (400, 520)),
    (49, (420, 460)),
    (70, (635, 460)),
    (77, (720, 460)),
    (84, (1000, 460)),
    (86, (1060, 240)),
    (180, (1060, 240)),
]

GROUND_TRUTH: dict = {
    "scenario": "section 15 demo",
    "fps": 30,
    "objects": {"notebook": "red", "mug": "blue", "person": "magenta"},
    "events": [
        {"type": "person_entered", "subject": "person", "start_time": 35.0},
        {
            "type": "object_picked_up",
            "subject": "notebook",
            "actor": "person",
            "start_time": 48.0,
        },
        {
            "type": "object_carried",
            "subject": "notebook",
            "actor": "person",
            "start_time": 48.0,
            "end_time": 84.0,
        },
        {
            "type": "object_placed",
            "subject": "notebook",
            "actor": "person",
            "start_time": 84.0,
        },
        {"type": "person_exited", "subject": "person", "start_time": 120.0},
    ],
}


def lerp(keyframes: list[tuple[float, tuple]], t: float) -> tuple:
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    for (t0, v0), (t1, v1) in zip(keyframes, keyframes[1:], strict=False):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            return tuple(a + (b - a) * f for a, b in zip(v0, v1, strict=True))
    return keyframes[-1][1]


def _draw_rect(
    canvas: np.ndarray,
    center: tuple[float, float],
    size: tuple[int, int],
    color: tuple,
) -> None:
    cx, cy = center
    w, h = size
    cv2.rectangle(
        canvas,
        (int(cx - w / 2), int(cy - h / 2)),
        (int(cx + w / 2), int(cy + h / 2)),
        color,
        -1,
    )


def generate_demo(
    out: str | Path = "data/demo.mp4",
    gt: str | Path = "data/demo_gt.json",
    fps: int = 30,
    size: tuple[int, int] = (1280, 720),
    duration: float = 180.0,
) -> tuple[Path, Path]:
    width, height = size
    total_frames = int(duration * fps)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for frame_id in range(total_frames):
        t = frame_id / fps
        canvas = np.full((height, width, 3), COLORS["background"], np.uint8)

        # static furniture
        cv2.rectangle(canvas, (200, 540), (720, 700), COLORS["desk"], -1)
        cv2.rectangle(canvas, (980, 260), (1160, 340), COLORS["shelf"], -1)

        # person (drawn before occluder so it can be occluded)
        person_cx = lerp(PERSON_X, t)[0]
        if -PERSON_SIZE[0] < person_cx < width + PERSON_SIZE[0]:
            _draw_rect(canvas, (person_cx, 610), PERSON_SIZE, COLORS["person"])

        # notebook
        _draw_rect(canvas, lerp(NOTEBOOK, t), NOTEBOOK_SIZE, COLORS["notebook"])

        # static mug on desk
        _draw_rect(canvas, (300, 515), MUG_SIZE, COLORS["mug"])

        # occluder drawn last (hides anything behind it)
        cv2.rectangle(canvas, (600, 0), (670, 720), COLORS["occluder"], -1)

        writer.write(canvas)

    writer.release()

    gt_data = dict(GROUND_TRUTH)
    gt_data["fps"] = fps
    gt_path = Path(gt)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(gt_data, indent=2))
    return out_path, gt_path
