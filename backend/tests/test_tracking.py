from __future__ import annotations

import numpy as np

from backend.app.memory.models import BoundingBox, Detection, TrackStatus
from backend.app.tracking.reid import ReIdentifier, color_histogram
from backend.app.tracking.tracker import MultiObjectTracker


def make_det(
    category: str,
    cx: float,
    cy: float,
    frame_id: int = 0,
    ts: float = 0.0,
    w: float = 40.0,
    h: float = 40.0,
) -> Detection:
    return Detection(
        category=category,
        bounding_box=BoundingBox(
            x_min=cx - w / 2, y_min=cy - h / 2, x_max=cx + w / 2, y_max=cy + h / 2
        ),
        confidence=0.9,
        frame_id=frame_id,
        timestamp=ts,
    )


def make_image(cx: float, cy: float, color: tuple = (0, 0, 220)) -> np.ndarray:
    img = np.full((120, 160, 3), 255, np.uint8)
    x1, y1 = int(cx - 20), int(cy - 20)
    img[y1 : y1 + 40, x1 : x1 + 40] = color
    return img


def test_persistent_id_across_frames():
    tracker = MultiObjectTracker()
    ids: set[str] = set()
    for f in range(5):
        tracks = tracker.update(
            [make_det("notebook", 50, 50, frame_id=f, ts=f)], f, float(f)
        )
        ids.add(tracks[0].track_id)
    assert len(ids) == 1


def test_distinct_objects_get_distinct_ids():
    tracker = MultiObjectTracker()
    tracks = tracker.update(
        [make_det("notebook", 50, 50), make_det("mug", 100, 100)], 0, 0.0
    )
    assert len({t.track_id for t in tracks}) == 2


def test_occlusion_within_tolerance_retains_id():
    tracker = MultiObjectTracker(occlusion_tolerance=30)
    track_id = tracker.update(
        [make_det("notebook", 50, 50, ts=0.0)], 0, 0.0
    )[0].track_id
    for f in range(1, 3):
        tracker.update([make_det("notebook", 50, 50, frame_id=f, ts=f)], f, float(f))
    for f in range(3, 8):
        tracker.update([], f, float(f))
    tracks = tracker.update([make_det("notebook", 50, 50, frame_id=8, ts=8.0)], 8, 8.0)
    assert tracks[0].track_id == track_id
    assert len(tracks[0].observations) == 4


def test_reid_relinks_after_loss():
    tracker = MultiObjectTracker(occlusion_tolerance=3, reid_threshold=0.6)
    image = make_image(50, 50)
    track_id = tracker.update(
        [make_det("notebook", 50, 50, ts=0.0)], 0, 0.0, image=image
    )[0].track_id
    for f in range(1, 3):
        tracker.update(
            [make_det("notebook", 50, 50, frame_id=f, ts=f)], f, float(f), image=image
        )
    for f in range(3, 7):
        tracker.update([], f, float(f))
    tracks = tracker.update(
        [make_det("notebook", 50, 50, frame_id=7, ts=7.0)], 7, 7.0, image=image
    )
    assert tracks[0].track_id == track_id
    assert len(tracks[0].observations) == 4


def test_reid_scores_similar_higher_than_dissimilar():
    reid = ReIdentifier()
    box = BoundingBox(x_min=30, y_min=30, x_max=70, y_max=70)
    red = color_histogram(make_image(50, 50, color=(0, 0, 220)))
    blue = color_histogram(make_image(50, 50, color=(220, 0, 0)))
    same = reid.match_score(red, red, box, box, 0.0)
    different = reid.match_score(red, blue, box, box, 0.0)
    assert same > 0.9
    assert same > different


def test_crossing_objects_keep_identity():
    tracker = MultiObjectTracker()
    id_a = None
    for f in range(10):
        dets = [
            make_det("notebook", 30 + 10 * f, 60, frame_id=f, ts=float(f)),
            make_det("notebook", 130 - 10 * f, 72, frame_id=f, ts=float(f)),
        ]
        tracks = tracker.update(dets, f, float(f))
        assert len(tracks) == 2
        if f == 0:
            id_a = next(t.track_id for t in tracks if t.trajectory[-1].x < 80)
    assert id_a is not None
    ended_right = [t for t in tracks if t.trajectory[-1].x > 100]
    assert len(ended_right) == 1
    assert ended_right[0].track_id == id_a


def test_single_missed_detection_keeps_id():
    tracker = MultiObjectTracker()
    track_id = tracker.update(
        [make_det("notebook", 50, 50, ts=0.0)], 0, 0.0
    )[0].track_id
    tracker.update([], 1, 1.0)
    tracks = tracker.update(
        [make_det("notebook", 52, 50, frame_id=2, ts=2.0)], 2, 2.0
    )
    assert tracks[0].track_id == track_id
    assert len(tracks[0].observations) == 2


def test_occluded_status_while_missed():
    tracker = MultiObjectTracker(occlusion_tolerance=30)
    tracker.update([make_det("notebook", 50, 50, ts=0.0)], 0, 0.0)
    tracks = tracker.update([], 1, 1.0)
    assert len(tracks) == 1
    assert tracks[0].status is TrackStatus.OCCLUDED


def test_two_same_color_objects_stay_distinct():
    tracker = MultiObjectTracker()
    red_a = make_image(50, 50, color=(0, 0, 220))
    red_b = make_image(120, 60, color=(0, 0, 220))
    for f in range(4):
        tracks = tracker.update(
            [
                make_det("notebook", 50, 50, frame_id=f, ts=float(f)),
                make_det("notebook", 120, 60, frame_id=f, ts=float(f)),
            ],
            f,
            float(f),
            image=red_a if f % 2 == 0 else red_b,
        )
    assert len(tracks) == 2
    assert len({t.track_id for t in tracks}) == 2


def test_different_color_elsewhere_is_not_relinked():
    tracker = MultiObjectTracker(occlusion_tolerance=1, reid_threshold=0.6)
    red = make_image(50, 50, color=(0, 0, 220))
    track_id = tracker.update(
        [make_det("notebook", 50, 50, ts=0.0)], 0, 0.0, image=red
    )[0].track_id
    tracker.update([], 1, 1.0)
    tracker.update([], 2, 2.0)
    tracker.update([], 3, 3.0)
    blue = make_image(140, 100, color=(220, 0, 0))
    tracks = tracker.update(
        [make_det("notebook", 140, 100, frame_id=4, ts=4.0)], 4, 4.0, image=blue
    )
    assert tracks[0].track_id != track_id


def test_object_leaving_scene_becomes_inactive():
    tracker = MultiObjectTracker(occlusion_tolerance=30)
    img = np.full((120, 160, 3), 255, np.uint8)
    track_id = tracker.update(
        [make_det("notebook", 140, 60, w=20, h=20, ts=0.0)], 0, 0.0, image=img
    )[0].track_id
    tracker.update(
        [make_det("notebook", 155, 60, w=20, h=20, frame_id=1, ts=1.0)],
        1,
        1.0,
        image=img,
    )
    for f in range(2, 8):
        tracker.update([], f, float(f), image=img)
    assert tracker.live_tracks() == []
    assert [t.track_id for t in tracker.inactive_tracks()] == [track_id]


def test_observation_appearance_populated():
    tracker = MultiObjectTracker()
    image = make_image(50, 50, color=(0, 0, 220))
    tracks = tracker.update(
        [make_det("notebook", 50, 50, ts=0.0)], 0, 0.0, image=image
    )
    obs = tracks[0].observations[0]
    assert obs.appearance_features
    assert obs.attributes.get("color") == "red"
    assert obs.association_confidence is not None
    assert 0.0 <= obs.association_confidence <= 1.0
