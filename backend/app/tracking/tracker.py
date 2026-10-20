"""Multi-object tracking with persistent identities (FR-005, FR-006).

Associates detections across frames with a global cost-matrix assignment
over category compatibility, Kalman-predicted position (using the actual
elapsed time ``dt``), IoU / spatial distance, and appearance similarity
(color histogram + named color + local ORB features). Tolerates missed
detections, distinguishes short occlusion from leaving the scene, and
re-identifies objects after longer interruptions via ``ReIdentifier``.

Track lifecycle::

    ACTIVE -> OCCLUDED (missed, within tolerance, still expected in view)
    OCCLUDED -> ACTIVE (re-associated)
    OCCLUDED -> LOST (missed beyond tolerance; kept for re-identification)
    LOST -> ACTIVE (re-identified)
    any -> INACTIVE (left the scene bounds, or expired beyond max_lost)

An occluded object stays in memory and is never treated as having
disappeared from the world; only INACTIVE tracks are retired.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..memory.models import (
    BoundingBox,
    Detection,
    Observation,
    Track,
    TrackStatus,
    TrajectoryPoint,
)
from ..perception.appearance import AppearanceProfile, encode_crop
from .association import AssociationConfig, match_detections
from .geometry import iou
from .reid import ReIdentifier, crop


class _Tracklet:
    def __init__(
        self,
        track_id: str,
        category: str,
        detection: Detection,
        timestamp: float,
        image: np.ndarray | None,
        association_confidence: float = 1.0,
    ) -> None:
        self.track_id = track_id
        self.category = category
        self.missed = 0
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_timestamp = timestamp
        self.last_bbox = detection.bounding_box
        # Constant-velocity model; velocity is per-second so prediction
        # must use the actual elapsed dt.
        self.velocity = (0.0, 0.0)
        self.status = TrackStatus.ACTIVE
        self.observations: list[Observation] = []
        self.trajectory: list[TrajectoryPoint] = []
        self.profile: AppearanceProfile = self._encode(image, detection.bounding_box)
        self.state_history: list[tuple[float, str]] = [(timestamp, "active")]

        cx, cy = detection.bounding_box.center
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)

        self._record(detection, timestamp, association_confidence)

    @staticmethod
    def _encode(image: np.ndarray | None, bbox: BoundingBox) -> AppearanceProfile:
        if image is None:
            return AppearanceProfile()
        return encode_crop(crop(image, bbox))

    def _set_transition(self, dt: float) -> None:
        dt = max(float(dt), 1e-3)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )

    def predict_bbox(self, dt: float) -> BoundingBox:
        self._set_transition(dt)
        pred = self.kf.predict()
        cx, cy = float(pred[0, 0]), float(pred[1, 0])
        w = self.last_bbox.width
        h = self.last_bbox.height
        return BoundingBox(
            x_min=cx - w / 2,
            y_min=cy - h / 2,
            x_max=cx + w / 2,
            y_max=cy + h / 2,
        )

    def predict_center(self, dt: float) -> tuple[float, float]:
        box = self.predict_bbox(dt)
        return box.center

    def update(
        self,
        detection: Detection,
        timestamp: float,
        image: np.ndarray | None,
        association_confidence: float = 1.0,
    ) -> None:
        cx, cy = detection.bounding_box.center
        prev_cx, prev_cy = self.last_bbox.center
        dt = max(timestamp - self.last_timestamp, 1e-3)
        self.velocity = ((cx - prev_cx) / dt, (cy - prev_cy) / dt)
        self._set_transition(dt)
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.last_bbox = detection.bounding_box
        self.missed = 0
        self.last_seen = timestamp
        self.last_timestamp = timestamp
        self._set_status(TrackStatus.ACTIVE, timestamp)
        self.profile = self._encode(image, detection.bounding_box)
        self._record(detection, timestamp, association_confidence)

    def _set_status(self, status: TrackStatus, timestamp: float) -> None:
        if self.status is not status:
            self.state_history.append((timestamp, status.value))
        self.status = status

    def mark_missed(self, timestamp: float, tolerance: int) -> None:
        self.missed += 1
        self.last_timestamp = timestamp
        # Predict forward so last_bbox follows the motion model during gaps.
        self._set_transition(max(timestamp - self.last_seen, 1e-3))
        self.kf.predict()
        if self.missed <= tolerance:
            self._set_status(TrackStatus.OCCLUDED, timestamp)
        else:
            self._set_status(TrackStatus.LOST, timestamp)

    def mark_inactive(self, timestamp: float) -> None:
        self._set_status(TrackStatus.INACTIVE, timestamp)

    def _record(
        self,
        detection: Detection,
        timestamp: float,
        association_confidence: float,
    ) -> None:
        cx, cy = detection.bounding_box.center
        features = (
            self.profile.descriptor.flatten().tolist()
            if self.profile.descriptor is not None
            else None
        )
        self.observations.append(
            Observation(
                observation_id=f"{self.track_id}:{detection.frame_id}",
                object_id=self.track_id,
                frame_id=detection.frame_id,
                timestamp=timestamp,
                bounding_box=detection.bounding_box,
                appearance_features=features,
                attributes=dict(self.profile.attributes),
                confidence=detection.confidence,
                association_confidence=float(association_confidence),
                evidence_refs=[],
            )
        )
        self.trajectory.append(
            TrajectoryPoint(
                frame_id=detection.frame_id, timestamp=timestamp, x=cx, y=cy
            )
        )

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            object_class=self.category,
            observations=self.observations,
            trajectory=self.trajectory,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            status=self.status,
        )


class MultiObjectTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        occlusion_tolerance: int = 30,
        reid_threshold: float = 0.6,
        max_lost: int = 100,
        assoc_config: AssociationConfig | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.occlusion_tolerance = occlusion_tolerance
        self.reid = ReIdentifier(threshold=reid_threshold)
        self.max_lost = max_lost
        self.assoc = assoc_config or AssociationConfig()
        self._tracks: dict[str, _Tracklet] = {}
        self._lost: list[_Tracklet] = []
        self._inactive: list[_Tracklet] = []
        self._next_id = 0

    def _new_id(self) -> str:
        track_id = f"obj_{self._next_id}"
        self._next_id += 1
        return track_id

    def _relink(
        self,
        detection: Detection,
        timestamp: float,
        image: np.ndarray | None,
        extra: list[_Tracklet] | None = None,
    ) -> _Tracklet | None:
        """Try to re-identify an unmatched detection as a lost track.

        Returns the relinked tracklet (with its association confidence)
        or None when no hypothesis passes the threshold. Color-only
        agreement is never enough on its own: candidates whose local
        visual features contribute nothing must also be spatiotemporally
        plausible, and the returned confidence reflects the breakdown.

        ``extra`` are still-active tracks that found no detection this frame
        (occluded, or the camera moved too far for the motion gate). They are
        candidates alongside the lost tracks: without them a moving camera
        spawns a new object for every sampled frame and one physical object
        fragments into many.
        """
        from ..perception.appearance import encode_crop as _encode

        det_profile = (
            _encode(crop(image, detection.bounding_box))
            if image is not None
            else AppearanceProfile()
        )
        best: _Tracklet | None = None
        best_score = -1.0
        for lost in [*self._lost, *(extra or [])]:
            if lost.status is TrackStatus.INACTIVE:
                continue
            if lost.category != detection.category:
                continue
            score, breakdown = self.reid.match_with_breakdown(
                lost.profile,
                det_profile,
                lost.last_bbox,
                detection.bounding_box,
                timestamp - lost.last_seen,
            )
            appearance = breakdown.get("appearance", {})
            # Guard against "similar color histogram == same object": when
            # local features carry no evidence, require strong
            # spatiotemporal agreement as well.
            if appearance.get("local", 0.0) <= 0.0 and (
                breakdown.get("spatial", 0.0) < 0.5 or score < self.reid.threshold + 0.1
            ):
                continue
            if score > best_score:
                best_score = score
                best = lost

        if best is not None and best_score >= self.reid.threshold:
            if best in self._lost:
                self._lost.remove(best)
            confidence = float(best_score)
            best.update(detection, timestamp, image, association_confidence=confidence)
            self._tracks[best.track_id] = best
            return best
        return None

    def _spawn(
        self, detection: Detection, timestamp: float, image: np.ndarray | None
    ) -> _Tracklet:
        track = _Tracklet(
            self._new_id(), detection.category, detection, timestamp, image
        )
        self._tracks[track.track_id] = track
        return track

    def _out_of_bounds(self, bbox: BoundingBox, image: np.ndarray | None) -> bool:
        if image is None:
            return False
        h, w = image.shape[:2]
        cx, cy = bbox.center
        margin = max(bbox.width, bbox.height)
        return cx < -margin or cy < -margin or cx > w + margin or cy > h + margin

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp: float,
        image: np.ndarray | None = None,
    ) -> list[Track]:
        active = list(self._tracks.values())
        last_seen_map = {t.track_id: t.last_timestamp for t in active}
        dt_map = {
            t.track_id: max(timestamp - last_seen_map[t.track_id], 1e-3) for t in active
        }
        predicted = {
            t.track_id: t.predict_bbox(dt_map[t.track_id]) for t in active
        }

        # Appearance similarity of every track/detection pair for the cost.
        det_profiles = [
            (
                encode_crop(crop(image, d.bounding_box))
                if image is not None
                else AppearanceProfile()
            )
            for d in detections
        ]
        from ..perception.appearance import profile_similarity as _sim

        appearance_scores = [
            [
                _sim(t.profile, det_profiles[j])["score"]
                for j in range(len(detections))
            ]
            for t in active
        ]
        category_matches = [
            [t.category == d.category for d in detections] for t in active
        ]
        assignments, unmatched_tracks, unmatched_dets = match_detections(
            [predicted[t.track_id] for t in active],
            [max(timestamp - last_seen_map[t.track_id], 0.0) for t in active],
            appearance_scores,
            category_matches,
            detections,
            self.assoc,
        )

        matched_ids: set[str] = set()
        for a in assignments:
            track = active[a.track_idx]
            det = detections[a.det_idx]
            matched_ids.add(track.track_id)
            track.update(det, timestamp, image, association_confidence=a.confidence)

        unmatched_active = [
            active[i] for i in unmatched_tracks if active[i].track_id not in matched_ids
        ]
        for j in unmatched_dets:
            det = detections[j]
            relinked = self._relink(det, timestamp, image, extra=unmatched_active)
            if relinked is None:
                self._spawn(det, timestamp, image)
            elif relinked in unmatched_active:
                unmatched_active.remove(relinked)
                matched_ids.add(relinked.track_id)

        for i in unmatched_tracks:
            track = active[i]
            if track.track_id in matched_ids:
                continue
            track.mark_missed(timestamp, self.occlusion_tolerance)
            left_view = self._out_of_bounds(predicted[track.track_id], image)
            drifted_far = (
                iou(predicted[track.track_id], track.last_bbox) < 0.01
                and track.missed > self.occlusion_tolerance * 2
            )
            if left_view or drifted_far:
                # Object predicted well outside the view (or its motion
                # model has drifted far beyond tolerance): it left the
                # scene rather than being briefly occluded.
                track.mark_inactive(timestamp)
                del self._tracks[track.track_id]
                self._inactive.append(track)
            elif track.status is TrackStatus.LOST:
                del self._tracks[track.track_id]
                self._lost.append(track)
                if len(self._lost) > self.max_lost:
                    oldest = self._lost.pop(0)
                    oldest.mark_inactive(timestamp)
                    self._inactive.append(oldest)

        return self.live_tracks()

    def live_tracks(self) -> list[Track]:
        return [t.to_track() for t in self._tracks.values()]

    def lost_tracks(self) -> list[Track]:
        return [t.to_track() for t in self._lost]

    def inactive_tracks(self) -> list[Track]:
        return [t.to_track() for t in self._inactive]
