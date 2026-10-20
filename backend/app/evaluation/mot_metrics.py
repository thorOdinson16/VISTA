"""Multi-object tracking metrics: CLEAR-MOT (MOTA/MOTP/ID switches) and IDF1.

Self-contained (numpy + scipy) so the evaluation runs on the same Python
environment as the pipeline. Boxes are ``(x1, y1, x2, y2)``; a "sequence" is
``{frame_id: [(track_id, box), ...]}`` and only frames present in the
prediction (the frames the pipeline actually sampled) are scored.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]
Sequence = dict[int, list[tuple[int | str, Box]]]


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(gt: list[Box], pred: list[Box]) -> np.ndarray:
    mat = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            mat[i, j] = iou(g, p)
    return mat


def evaluate_mot(
    gt: Sequence, pred: Sequence, iou_threshold: float = 0.5
) -> dict:
    """Score ``pred`` against ``gt`` on the frames present in ``pred``."""
    frames = sorted(pred)
    last_match: dict = {}  # gt id -> pred id it was last matched to
    matched_frames: dict = defaultdict(int)  # gt id -> frames matched
    gt_frames: dict = defaultdict(int)  # gt id -> frames present
    pair_frames: dict = defaultdict(int)  # (gt id, pred id) -> co-occurring frames
    pred_total = gt_total = 0
    tp = fp = fn = idsw = 0
    iou_sum = 0.0

    for frame in frames:
        g_items = gt.get(frame, [])
        p_items = pred.get(frame, [])
        g_ids = [i for i, _ in g_items]
        g_boxes = [b for _, b in g_items]
        p_ids = [i for i, _ in p_items]
        p_boxes = [b for _, b in p_items]
        gt_total += len(g_items)
        pred_total += len(p_items)
        for gid in g_ids:
            gt_frames[gid] += 1

        mat = _iou_matrix(g_boxes, p_boxes)
        matches: list[tuple[int, int]] = []
        used_g: set[int] = set()
        used_p: set[int] = set()
        # Keep last frame's correspondences when still valid (CLEAR-MOT).
        for gi, gid in enumerate(g_ids):
            pid = last_match.get(gid)
            if pid is None or pid not in p_ids:
                continue
            pj = p_ids.index(pid)
            if pj not in used_p and mat[gi, pj] >= iou_threshold:
                matches.append((gi, pj))
                used_g.add(gi)
                used_p.add(pj)
        free_g = [i for i in range(len(g_ids)) if i not in used_g]
        free_p = [j for j in range(len(p_ids)) if j not in used_p]
        if free_g and free_p:
            sub = mat[np.ix_(free_g, free_p)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= iou_threshold:
                    matches.append((free_g[r], free_p[c]))
        for gi, pj in matches:
            gid, pid = g_ids[gi], p_ids[pj]
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
            matched_frames[gid] += 1
            iou_sum += mat[gi, pj]
        tp += len(matches)
        fn += len(g_items) - len(matches)
        fp += len(p_items) - len(matches)

        # Co-occurrence counts for the global (IDF1) assignment.
        for gi, gid in enumerate(g_ids):
            for pj, pid in enumerate(p_ids):
                if mat[gi, pj] >= iou_threshold:
                    pair_frames[(gid, pid)] += 1

    # IDF1: one global one-to-one assignment of GT tracks to predicted tracks.
    gt_ids = sorted({g for g, _ in pair_frames}, key=str)
    pr_ids = sorted({p for _, p in pair_frames}, key=str)
    idtp = 0
    if gt_ids and pr_ids:
        weight = np.zeros((len(gt_ids), len(pr_ids)))
        gi = {g: i for i, g in enumerate(gt_ids)}
        pi = {p: j for j, p in enumerate(pr_ids)}
        for (g, p), n in pair_frames.items():
            weight[gi[g], pi[p]] = n
        rows, cols = linear_sum_assignment(-weight)
        idtp = int(weight[rows, cols].sum())
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0

    tracks = max(len(gt_frames), 1)
    mostly_tracked = sum(
        1 for g, n in gt_frames.items() if matched_frames[g] / n >= 0.8
    )
    mostly_lost = sum(
        1 for g, n in gt_frames.items() if matched_frames[g] / n <= 0.2
    )
    return {
        "frames_scored": len(frames),
        "gt_tracks": len(gt_frames),
        "gt_detections": gt_total,
        "pred_detections": pred_total,
        "mota": round(1 - (fn + fp + idsw) / gt_total, 4) if gt_total else 0.0,
        "motp_iou": round(iou_sum / tp, 4) if tp else 0.0,
        "idf1": round(idf1, 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "id_switches": idsw,
        "false_positives": fp,
        "false_negatives": fn,
        "mostly_tracked": round(mostly_tracked / tracks, 4),
        "mostly_lost": round(mostly_lost / tracks, 4),
    }
