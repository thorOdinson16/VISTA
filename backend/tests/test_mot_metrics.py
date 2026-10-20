"""Sanity tests for the MOT metrics against hand-computable cases."""

from backend.app.evaluation.mot_metrics import evaluate_mot, iou


def box(x):
    return (x, 0.0, x + 10.0, 10.0)


def seq(spec):
    """spec: {frame: [(id, x), ...]}"""
    return {f: [(i, box(x)) for i, x in items] for f, items in spec.items()}


def test_iou_basics():
    assert iou(box(0), box(0)) == 1.0
    assert iou(box(0), box(20)) == 0.0
    assert abs(iou(box(0), box(5)) - (50 / 150)) < 1e-9


def test_perfect_tracking():
    gt = seq({1: [(1, 0), (2, 50)], 2: [(1, 1), (2, 51)], 3: [(1, 2), (2, 52)]})
    pred = seq(
        {1: [("a", 0), ("b", 50)], 2: [("a", 1), ("b", 51)], 3: [("a", 2), ("b", 52)]}
    )
    m = evaluate_mot(gt, pred)
    assert m["mota"] == 1.0 and m["idf1"] == 1.0 and m["id_switches"] == 0
    assert m["mostly_tracked"] == 1.0


def test_id_switch_is_counted_and_hurts_idf1():
    gt = seq({f: [(1, f)] for f in range(1, 5)})
    pred = seq({1: [("a", 1)], 2: [("a", 2)], 3: [("b", 3)], 4: [("b", 4)]})
    m = evaluate_mot(gt, pred)
    assert m["id_switches"] == 1
    assert m["mota"] == 0.75  # 1 - 1/4
    assert abs(m["idf1"] - 0.5) < 1e-9  # best single id covers 2 of 4 frames


def test_misses_and_false_positives():
    gt = seq({1: [(1, 0)], 2: [(1, 0)], 3: [(1, 0)], 4: [(1, 0)]})
    pred = seq({1: [("a", 0)], 2: [], 3: [("a", 0), ("x", 80)], 4: [("a", 0)]})
    m = evaluate_mot(gt, pred)
    assert m["false_negatives"] == 1 and m["false_positives"] == 1
    assert m["mota"] == 0.5
    assert m["recall"] == 0.75 and m["precision"] == 0.75


def test_only_predicted_frames_are_scored():
    gt = seq({1: [(1, 0)], 2: [(1, 0)], 3: [(1, 0)]})
    pred = seq({1: [("a", 0)], 3: [("a", 0)]})  # frame 2 not sampled
    m = evaluate_mot(gt, pred)
    assert m["frames_scored"] == 2 and m["mota"] == 1.0
