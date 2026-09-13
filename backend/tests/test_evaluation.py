"""Evaluation harness: metric logic and end-to-end demo scoring."""

from __future__ import annotations

import pytest

from backend.app.evaluation.demo import generate_demo
from backend.app.evaluation.harness import (
    EVENT_F1_MIN,
    QUERY_ACCURACY_MIN,
    match_events,
    run_evaluation,
)


def _sig(etype, subject, start, actor=None, end=None):
    return {
        "type": etype,
        "subject": subject,
        "actor": actor,
        "start": start,
        "end": end,
    }


def test_match_events_counts_tp_fp_fn():
    gt = [
        _sig("object_picked_up", "notebook", 48.0, actor="person"),
        _sig("object_carried", "notebook", 48.0, actor="person", end=84.0),
    ]
    preds = [
        _sig("object_picked_up", "notebook", 49.0, actor="person"),
        _sig("object_moved", "notebook", 50.0),
    ]
    metrics = match_events(preds, gt)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_match_events_rejects_wrong_actor_and_time():
    gt = [_sig("object_picked_up", "notebook", 48.0, actor="person")]
    wrong_actor = [_sig("object_picked_up", "notebook", 48.0, actor="mug")]
    wrong_time = [_sig("object_picked_up", "notebook", 60.0, actor="person")]
    assert match_events(wrong_actor, gt)["true_positives"] == 0
    assert match_events(wrong_time, gt)["true_positives"] == 0


@pytest.fixture(scope="module")
def demo_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    video = tmp / "demo.mp4"
    gt = tmp / "demo_gt.json"
    generate_demo(out=video, gt=gt, fps=8, duration=125)
    return run_evaluation(video, gt)


def test_demo_event_f1_meets_threshold(demo_report):
    assert demo_report["events"]["f1"] >= EVENT_F1_MIN


def test_demo_query_accuracy_meets_threshold(demo_report):
    assert demo_report["queries"]["accuracy"] >= QUERY_ACCURACY_MIN


def test_demo_overall_passes(demo_report):
    assert demo_report["passed"]


def test_demo_has_zero_throughput(demo_report):
    assert demo_report["throughput_fps"] > 0
    assert demo_report["frames_processed"] > 0
