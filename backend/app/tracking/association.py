"""Detection-to-track association via a global cost matrix.

Replaces the previous greedy per-detection matcher (which assigned each
detection to its best-IoU track independently and broke when two similar
objects crossed). Here every track/detection pair gets one combined cost
from category compatibility, predicted position (IoU + center distance),
appearance similarity, and temporal gap; the matrix is then solved
globally so no two detections claim the same track and no two tracks
claim the same detection.

No third-party assignment solver is used (dependency set unchanged): a
greedy minimum-cost matching over the full matrix is applied, which is
exact for the unambiguous cases and degrades gracefully (high cost ->
unmatched) under ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..memory.models import BoundingBox, Detection
from .geometry import center_distance, iou


@dataclass
class AssociationConfig:
    w_iou: float = 0.45
    w_distance: float = 0.25
    w_appearance: float = 0.30
    # Pairs worse than this cost are never matched (stay unmatched).
    max_cost: float = 0.85
    # Hard gates: below this IoU *and* beyond this distance -> no match.
    min_iou: float = 0.05
    max_distance: float = 250.0
    distance_scale: float = 150.0
    gap_scale: float = 10.0


@dataclass
class Association:
    track_idx: int
    det_idx: int
    cost: float
    confidence: float
    components: dict


def pair_cost(
    pred_bbox: BoundingBox,
    last_seen_gap: float,
    appearance_score: float,
    category_match: bool,
    det_bbox: BoundingBox,
    cfg: AssociationConfig,
) -> tuple[float, dict]:
    """Combined association cost in [0, ~1]; inf when gated out."""
    overlap = iou(pred_bbox, det_bbox)
    dist = center_distance(pred_bbox, det_bbox)
    if not category_match:
        return float("inf"), {"gated": "category"}
    if overlap < cfg.min_iou and dist > cfg.max_distance:
        return float("inf"), {"gated": "spatial", "iou": overlap, "distance": dist}
    import math

    gap_penalty = 1.0 - math.exp(-max(0.0, last_seen_gap) / cfg.gap_scale)
    dist_cost = 1.0 - math.exp(-dist / cfg.distance_scale)
    cost = (
        cfg.w_iou * (1.0 - overlap)
        + cfg.w_distance * dist_cost
        + cfg.w_appearance * (1.0 - appearance_score)
    )
    # Temporal gap can only veto, never force: cap its influence.
    cost = min(1.0, cost + 0.2 * gap_penalty)
    components = {
        "iou": overlap,
        "distance": dist,
        "appearance": appearance_score,
        "gap_penalty": gap_penalty,
    }
    return cost, components


def solve(
    cost: list[list[float]], max_cost: float, n_dets: int = 0
) -> tuple[list[Association], list[int], list[int]]:
    """Greedy global minimum-cost matching.

    Returns (assignments, unmatched_track_indices, unmatched_det_indices).
    """
    n_tracks = len(cost)
    if n_tracks == 0:
        return [], [], list(range(n_dets))
    n_dets = len(cost[0])
    candidates = sorted(
        (
            (cost[i][j], i, j)
            for i in range(n_tracks)
            for j in range(n_dets)
            if cost[i][j] <= max_cost
        ),
        key=lambda t: t[0],
    )
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    assignments: list[Association] = []
    for c, i, j in candidates:
        if i in used_tracks or j in used_dets:
            continue
        used_tracks.add(i)
        used_dets.add(j)
        assignments.append(
            Association(
                track_idx=i,
                det_idx=j,
                cost=c,
                confidence=float(max(0.0, min(1.0, 1.0 - c))),
                components={},
            )
        )
    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
    return assignments, unmatched_tracks, unmatched_dets


def match_detections(
    pred_bboxes: list[BoundingBox],
    gaps: list[float],
    appearance_scores: list[list[float]],
    category_matches: list[list[bool]],
    detections: list[Detection],
    cfg: AssociationConfig,
) -> tuple[list[Association], list[int], list[int]]:
    """Build the cost matrix and solve it; attach cost components."""
    n_tracks = len(pred_bboxes)
    n_dets = len(detections)
    matrix: list[list[float]] = []
    comp_grid: list[list[dict]] = []
    for i in range(n_tracks):
        row: list[float] = []
        crow: list[dict] = []
        for j in range(n_dets):
            c, comp = pair_cost(
                pred_bboxes[i],
                gaps[i],
                appearance_scores[i][j],
                category_matches[i][j],
                detections[j].bounding_box,
                cfg,
            )
            row.append(c)
            crow.append(comp)
        matrix.append(row)
        comp_grid.append(crow)
    assignments, unmatched_tracks, unmatched_dets = solve(
        matrix, cfg.max_cost, n_dets
    )
    for a in assignments:
        a.components = comp_grid[a.track_idx][a.det_idx]
    return assignments, unmatched_tracks, unmatched_dets
