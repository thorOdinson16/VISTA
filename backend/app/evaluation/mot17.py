"""MOT17 (MOTChallenge) helpers: sequence info, ground truth, video assembly.

MOT17 ships frames as ``img1/000001.jpg`` plus ``gt/gt.txt``; the VISTA
pipeline consumes video files, so :func:`frames_to_video` assembles one.
Ground-truth columns: ``frame,id,x,y,w,h,consider,class,visibility`` with
class 1 = pedestrian. Classes 2 (person on vehicle), 7 (static person),
8 (distractor) and 12 (reflection) are *ignore* regions: predictions on
them are not counted as false positives.

Data is research-licensed and is never committed to the repository.
"""

from __future__ import annotations

import configparser
from collections import defaultdict
from pathlib import Path

import cv2

from .mot_metrics import Box, Sequence, iou

PEDESTRIAN = 1
IGNORE_CLASSES = {2, 7, 8, 12}


def read_seqinfo(seq_dir: str | Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(Path(seq_dir) / "seqinfo.ini")
    s = parser["Sequence"]
    return {
        "name": s["name"],
        "fps": float(s["frameRate"]),
        "length": int(s["seqLength"]),
        "width": int(s["imWidth"]),
        "height": int(s["imHeight"]),
        "img_dir": s.get("imDir", "img1"),
        "ext": s.get("imExt", ".jpg"),
    }


def load_gt(seq_dir: str | Path) -> tuple[Sequence, dict[int, list[Box]]]:
    """Return (pedestrian ground truth, ignore regions), both keyed by 1-based frame."""
    gt: Sequence = defaultdict(list)
    ignore: dict[int, list[Box]] = defaultdict(list)
    path = Path(seq_dir) / "gt" / "gt.txt"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f, tid, x, y, w, h, consider, cls, _vis = (
            float(v) for v in line.split(",")[:9]
        )
        box = (x, y, x + w, y + h)
        frame = int(f)
        if int(cls) == PEDESTRIAN and int(consider) == 1:
            gt[frame].append((int(tid), box))
        elif int(cls) in IGNORE_CLASSES or int(consider) == 0:
            ignore[frame].append(box)
    return dict(gt), dict(ignore)


def drop_ignored(
    pred: Sequence,
    gt: Sequence,
    ignore: dict[int, list[Box]],
    threshold: float = 0.5,
) -> Sequence:
    """Remove predictions sitting on ignore regions and not on a pedestrian."""
    out: Sequence = {}
    for frame, items in pred.items():
        gt_boxes = [b for _, b in gt.get(frame, [])]
        ignore_boxes = ignore.get(frame, [])
        kept = []
        for pid, box in items:
            on_gt = any(iou(box, g) >= threshold for g in gt_boxes)
            on_ignore = any(iou(box, r) >= threshold for r in ignore_boxes)
            if on_ignore and not on_gt:
                continue
            kept.append((pid, box))
        out[frame] = kept
    return out


def frames_to_video(seq_dir: str | Path, out_path: str | Path) -> Path:
    """Assemble ``img1/*.jpg`` into an mp4 at the sequence's frame rate."""
    seq_dir = Path(seq_dir)
    info = read_seqinfo(seq_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {out_path}")
    for i in range(1, info["length"] + 1):
        image = cv2.imread(str(seq_dir / info["img_dir"] / f"{i:06d}{info['ext']}"))
        if image is None:
            raise FileNotFoundError(f"Missing frame {i} in {seq_dir}")
        writer.write(image)
    writer.release()
    return out_path
