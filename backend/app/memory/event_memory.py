"""Event memory: storage and extraction of temporal events (FR-010, FR-011)."""

from __future__ import annotations

import math

from .models import EventRecord, EventType, ObjectRecord, TrajectoryPoint


class EventMemory:
    def __init__(self) -> None:
        self._events: list[EventRecord] = []
        self._counter = 0

    def add(self, event: EventRecord) -> EventRecord:
        if not event.event_id:
            event.event_id = f"evt_{self._counter}"
            self._counter += 1
        self._events.append(event)
        return event

    def all(self) -> list[EventRecord]:
        return list(self._events)

    def by_type(self, event_type: EventType) -> list[EventRecord]:
        return [e for e in self._events if e.type is event_type]

    def by_subject(self, subject: str) -> list[EventRecord]:
        return [e for e in self._events if subject in e.subject_refs]

    def by_actor(self, actor: str) -> list[EventRecord]:
        return [e for e in self._events if actor in e.actor_refs]

    def in_range(self, start: float, end: float) -> list[EventRecord]:
        return [e for e in self._events if start <= e.start_time <= end]

    def reseed(self) -> None:
        """Restore the ID counter after loading persisted events."""
        best = -1
        for event in self._events:
            if event.event_id.startswith("evt_"):
                try:
                    best = max(best, int(event.event_id.split("_", 1)[1]))
                except ValueError:
                    continue
        self._counter = max(self._counter, best + 1)


def _interp(points: list[TrajectoryPoint], t: float) -> tuple[float, float]:
    if t <= points[0].timestamp:
        return points[0].x, points[0].y
    if t >= points[-1].timestamp:
        return points[-1].x, points[-1].y
    for a, b in zip(points, points[1:], strict=False):
        if a.timestamp <= t <= b.timestamp:
            f = (
                (t - a.timestamp) / (b.timestamp - a.timestamp)
                if b.timestamp != a.timestamp
                else 0.0
            )
            return a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f
    return points[-1].x, points[-1].y


class EventExtractor:
    """Derive events from persistent object records.

    Two event families are produced:

    * **Observed** (``kind="observed"``) -- directly supported by visual
      evidence: appeared/disappeared, person entered/exited, object
      visible, object in a person's vicinity (``OBJECT_APPROACHED``).
    * **Inferred** (``kind="inferred"``) -- pickup/carry/place hypotheses
      that require *multiple* supporting observations (stationary-before,
      approach, correlated joint motion, displacement from origin,
      separation-after). Proximity + movement alone is never treated as
      proof of pickup; each inference carries its component scores in
      ``details`` and real evidence references once finalized.
    """

    def __init__(
        self,
        moved_threshold: float = 50.0,
        proximity_threshold: float = 80.0,
        move_epsilon: float = 5.0,
        video_duration: float | None = None,
        end_margin: float = 2.0,
        stationary_speed: float = 8.0,
        min_correlation: float = 0.6,
        min_joint_fraction: float = 0.4,
        pickup_threshold: float = 0.55,
        sample_step: float = 0.5,
        pre_window: float = 3.0,
    ) -> None:
        self.moved_threshold = moved_threshold
        self.proximity_threshold = proximity_threshold
        self.move_epsilon = move_epsilon
        self.video_duration = video_duration
        self.end_margin = end_margin
        self.stationary_speed = stationary_speed
        self.min_correlation = min_correlation
        self.min_joint_fraction = min_joint_fraction
        self.pickup_threshold = pickup_threshold
        self.sample_step = sample_step
        self.pre_window = pre_window
        from .movement import MovementSegmenter

        self.segmenter = MovementSegmenter(moved_threshold=moved_threshold)

    def extract(self, objects: list[ObjectRecord]) -> list[EventRecord]:
        events: list[EventRecord] = []
        for obj in objects:
            events.extend(self._appearance_events(obj))
            events.extend(self._movement_events(obj))
        events.extend(self._interaction_events(objects))
        return events

    def _appearance_events(self, obj: ObjectRecord) -> list[EventRecord]:
        if obj.first_seen is None:
            return []
        events: list[EventRecord] = []
        if obj.category == "person":
            events.append(
                EventRecord(
                    type=EventType.PERSON_ENTERED,
                    start_time=obj.first_seen,
                    subject_refs=[obj.object_id],
                    confidence=obj.confidence,
                    kind="observed",
                )
            )
        else:
            events.append(
                EventRecord(
                    type=EventType.OBJECT_APPEARED,
                    start_time=obj.first_seen,
                    subject_refs=[obj.object_id],
                    confidence=obj.confidence,
                    kind="observed",
                )
            )

        if (
            self.video_duration is not None
            and obj.last_seen is not None
            and obj.last_seen < self.video_duration - self.end_margin
        ):
            exit_type = (
                EventType.PERSON_EXITED
                if obj.category == "person"
                else EventType.OBJECT_DISAPPEARED
            )
            events.append(
                EventRecord(
                    type=exit_type,
                    start_time=obj.last_seen,
                    subject_refs=[obj.object_id],
                    confidence=obj.confidence,
                    kind="observed",
                )
            )
        return events

    def _movement_events(self, obj: ObjectRecord) -> list[EventRecord]:
        segments = self.segmenter.segment(obj.trajectory)
        events: list[EventRecord] = []
        for seg in segments:
            events.append(
                EventRecord(
                    type=EventType.OBJECT_MOVED,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    subject_refs=[obj.object_id],
                    source_state={"x": seg.start_x, "y": seg.start_y},
                    destination_state={"x": seg.end_x, "y": seg.end_y},
                    confidence=min(1.0, seg.confidence * max(obj.confidence, 0.5)),
                    kind="inferred",
                    details={
                        "displacement": seg.displacement,
                        "duration": seg.duration,
                    },
                )
            )
        return events

    def _interaction_events(
        self, objects: list[ObjectRecord]
    ) -> list[EventRecord]:
        persons = [o for o in objects if o.category == "person"]
        things = [o for o in objects if o.category != "person"]
        events: list[EventRecord] = []

        for person in persons:
            p_pts = sorted(person.trajectory, key=lambda p: p.timestamp)
            if len(p_pts) < 2:
                continue
            for obj in things:
                o_pts = sorted(obj.trajectory, key=lambda p: p.timestamp)
                if len(o_pts) < 2:
                    continue
                events.extend(self._pair_events(person, p_pts, obj, o_pts))
        return events

    def _pair_events(
        self,
        person: ObjectRecord,
        p_pts: list[TrajectoryPoint],
        obj: ObjectRecord,
        o_pts: list[TrajectoryPoint],
    ) -> list[EventRecord]:
        lo = max(p_pts[0].timestamp, o_pts[0].timestamp)
        hi = min(p_pts[-1].timestamp, o_pts[-1].timestamp)
        if lo >= hi:
            return []

        # Fine regular sampling (not just trajectory knots) so brief
        # proximity and the exact onset of joint motion are not missed.
        span = hi - lo
        n = max(2, min(400, int(span / self.sample_step) + 1))
        times = [lo + span * i / (n - 1) for i in range(n)]

        runs: list[tuple[float, float]] = []
        run_start: float | None = None
        prev: float | None = None
        for t in times:
            px, py = _interp(p_pts, t)
            ox, oy = _interp(o_pts, t)
            gap = self._gap(person, px, py, t, obj, ox, oy, t)
            near = gap <= self.proximity_threshold
            if near and run_start is None:
                run_start = t
            elif not near and run_start is not None:
                runs.append((run_start, prev if prev is not None else t))
                run_start = None
            prev = t
        if run_start is not None:
            runs.append((run_start, times[-1]))

        confidence = min(person.confidence, obj.confidence)
        events: list[EventRecord] = []
        for rs, re in runs:
            events.extend(
                self._run_events(person, p_pts, obj, o_pts, rs, re, confidence)
            )
        return events

    # -- pickup hypothesis ------------------------------------------------

    @staticmethod
    def _box_size(obj: ObjectRecord, t: float) -> tuple[float, float]:
        """Interpolated (width, height) of an object at time t.

        Returns (0, 0) when no observations exist, degrading gracefully
        to center distance.
        """
        obs = sorted(obj.observations, key=lambda o: o.timestamp)
        if not obs:
            return (0.0, 0.0)
        if t <= obs[0].timestamp:
            box = obs[0].bounding_box
        elif t >= obs[-1].timestamp:
            box = obs[-1].bounding_box
        else:
            box = obs[-1].bounding_box
            for a, b in zip(obs, obs[1:], strict=False):
                if a.timestamp <= t <= b.timestamp:
                    f = (
                        (t - a.timestamp) / (b.timestamp - a.timestamp)
                        if b.timestamp != a.timestamp
                        else 0.0
                    )
                    w = a.bounding_box.width
                    w += (b.bounding_box.width - a.bounding_box.width) * f
                    h = a.bounding_box.height
                    h += (b.bounding_box.height - a.bounding_box.height) * f
                    return (w, h)
        return (box.width, box.height)

    def _gap(
        self,
        person: ObjectRecord,
        px: float,
        py: float,
        pt: float,
        obj: ObjectRecord,
        ox: float,
        oy: float,
        ot: float,
    ) -> float:
        """Edge-to-edge box gap (0 when boxes overlap).

        A person standing over a notebook has distant centers but
        adjacent boxes; measuring the true gap keeps interaction
        hypotheses grounded in actual contact geometry.
        """
        w1, h1 = self._box_size(person, pt)
        w2, h2 = self._box_size(obj, ot)
        dx = abs(px - ox) - (w1 + w2) / 2.0
        dy = abs(py - oy) - (h1 + h2) / 2.0
        return math.hypot(max(0.0, dx), max(0.0, dy))

    def _speed(self, pts: list[TrajectoryPoint], t: float, window: float) -> float:
        """Mean speed around time t over the given window (px/s)."""
        x0, y0 = _interp(pts, max(t - window / 2.0, pts[0].timestamp))
        x1, y1 = _interp(pts, min(t + window / 2.0, pts[-1].timestamp))
        dt = min(t + window / 2.0, pts[-1].timestamp) - max(
            t - window / 2.0, pts[0].timestamp
        )
        if dt <= 1e-6:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    def _distance_series(
        self,
        person: ObjectRecord,
        p_pts: list[TrajectoryPoint],
        obj: ObjectRecord,
        o_pts: list[TrajectoryPoint],
        lo: float,
        hi: float,
        step: float,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Sample (t, px, py, ox, oy, edge_gap) over [lo, hi]."""
        span = hi - lo
        if span <= 0:
            return []
        n = max(2, min(200, int(span / step) + 1))
        out = []
        for i in range(n):
            t = lo + span * i / (n - 1)
            px, py = _interp(p_pts, t)
            ox, oy = _interp(o_pts, t)
            gap = self._gap(person, px, py, t, obj, ox, oy, t)
            out.append((t, px, py, ox, oy, gap))
        return out

    def _run_events(
        self,
        person: ObjectRecord,
        p_pts: list[TrajectoryPoint],
        obj: ObjectRecord,
        o_pts: list[TrajectoryPoint],
        rs: float,
        re: float,
        confidence: float,
    ) -> list[EventRecord]:
        events: list[EventRecord] = [
            EventRecord(
                type=EventType.OBJECT_APPROACHED,
                start_time=rs,
                end_time=re,
                subject_refs=[obj.object_id],
                actor_refs=[person.object_id],
                confidence=confidence,
                # Vicinity is directly observed; it is evidence *for* a
                # possible pickup, not the pickup itself.
                kind="observed",
            )
        ]
        hypothesis = self._pickup_hypothesis(person, p_pts, obj, o_pts, rs, re)
        if hypothesis is None:
            return events

        details = hypothesis["details"]
        base_conf = min(1.0, confidence * hypothesis["score"])
        pickup_t = hypothesis["pickup_time"]
        carry_end = hypothesis["carry_end"]
        events.append(
            EventRecord(
                type=EventType.OBJECT_PICKED_UP,
                start_time=pickup_t,
                subject_refs=[obj.object_id],
                actor_refs=[person.object_id],
                source_state={"x": hypothesis["origin_x"], "y": hypothesis["origin_y"]},
                destination_state={
                    "x": hypothesis["pickup_x"],
                    "y": hypothesis["pickup_y"],
                },
                confidence=base_conf,
                kind="inferred",
                details=details,
            )
        )
        events.append(
            EventRecord(
                type=EventType.OBJECT_CARRIED,
                start_time=pickup_t,
                end_time=carry_end,
                subject_refs=[obj.object_id],
                actor_refs=[person.object_id],
                confidence=min(1.0, base_conf * 0.95),
                kind="inferred",
                details=details,
            )
        )
        placed_conf = base_conf * (1.0 if details["separation"] >= 0.75 else 0.7)
        events.append(
            EventRecord(
                type=EventType.OBJECT_PLACED,
                start_time=carry_end,
                subject_refs=[obj.object_id],
                actor_refs=[person.object_id],
                source_state={
                    "x": hypothesis["carry_end_x"],
                    "y": hypothesis["carry_end_y"],
                },
                confidence=min(1.0, placed_conf),
                kind="inferred",
                details=details,
            )
        )
        return events

    def _pickup_hypothesis(
        self,
        person: ObjectRecord,
        p_pts: list[TrajectoryPoint],
        obj: ObjectRecord,
        o_pts: list[TrajectoryPoint],
        rs: float,
        re: float,
    ) -> dict | None:
        """Evaluate the pickup hypothesis for one proximity run.

        Required (gated) evidence::

            object initially stationary
            -> person approaches (distance decreasing before the run)
            -> object moves jointly with the person (correlated velocities
               while both are moving, sustained over a joint interval)
            -> object displaced from its original location

        Supporting (confidence) evidence::

            -> object later separates from the person / becomes stationary

        Returns None when the gated evidence is absent -- proximity plus
        incidental movement is *not* pickup.
        """
        step = max(0.1, (re - rs) / 60.0)

        # (1) Object stationary before the run.
        pre_lo = max(o_pts[0].timestamp, rs - self.pre_window)
        pre_speed = self._speed(o_pts, (pre_lo + rs) / 2.0, rs - pre_lo)
        stationary_before = 1.0 - min(1.0, pre_speed / self.stationary_speed)
        if stationary_before < 0.5:
            return None

        # (2) Person approaching: distance decreasing over the pre-window.
        pre = self._distance_series(person, p_pts, obj, o_pts, pre_lo, rs, step)
        approach = 0.0
        if len(pre) >= 2:
            drops = sum(
                1 for a, b in zip(pre, pre[1:], strict=False) if b[5] < a[5]
            )
            approach = drops / (len(pre) - 1)
        if approach < 0.5:
            return None

        # (3) Joint motion during the run: both moving with aligned velocity.
        run = self._distance_series(person, p_pts, obj, o_pts, rs, re, step)
        if len(run) < 2:
            return None
        joint_times: list[float] = []
        cosines: list[float] = []
        for (t0, px0, py0, ox0, oy0, _), (t1, px1, py1, ox1, oy1, _) in zip(
            run, run[1:], strict=False
        ):
            dt = t1 - t0
            if dt <= 1e-9:
                continue
            pvx, pvy = (px1 - px0) / dt, (py1 - py0) / dt
            ovx, ovy = (ox1 - ox0) / dt, (oy1 - oy0) / dt
            ps = math.hypot(pvx, pvy)
            os_ = math.hypot(ovx, ovy)
            if ps < self.stationary_speed or os_ < self.stationary_speed:
                continue
            cos = (pvx * ovx + pvy * ovy) / (ps * os_)
            if cos >= self.min_correlation:
                joint_times.append(t0)
                cosines.append(cos)
        joint_span = (max(joint_times) - min(joint_times)) if joint_times else 0.0
        run_span = max(re - rs, 1e-6)
        joint_fraction = min(1.0, joint_span / run_span)
        mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
        joint = joint_fraction * mean_cos
        if joint_fraction < self.min_joint_fraction or not joint_times:
            return None

        # (4) Object displaced from its original location.
        ox_pre, oy_pre = _interp(o_pts, pre_lo)
        ox_end, oy_end = _interp(o_pts, re)
        disp_from_origin = math.hypot(ox_end - ox_pre, oy_end - oy_pre)
        displaced = min(1.0, disp_from_origin / self.moved_threshold)
        if displaced < 0.5:
            return None

        # (5) Separation afterwards (supporting, not gating).
        post_hi = min(o_pts[-1].timestamp, p_pts[-1].timestamp, re + self.pre_window)
        separation = 0.4  # discounted default at data boundary / unknown
        if post_hi - re > 1e-6:
            post = self._distance_series(person, p_pts, obj, o_pts, re, post_hi, step)
            if post:
                grew = post[-1][5] - post[0][5]
                obj_post_speed = self._speed(o_pts, (re + post_hi) / 2.0, post_hi - re)
                stationary_after = 1.0 - min(
                    1.0, obj_post_speed / self.stationary_speed
                )
                separation = max(
                    min(1.0, grew / self.proximity_threshold),
                    0.7 * stationary_after,
                )

        score = (
            0.25 * stationary_before
            + 0.20 * approach
            + 0.30 * joint
            + 0.15 * displaced
            + 0.10 * separation
        )
        if score < self.pickup_threshold:
            return None

        pickup_time = min(joint_times)
        carry_end = max(joint_times)
        pkx, pky = _interp(o_pts, pickup_time)
        cex, cey = _interp(o_pts, carry_end)
        return {
            "pickup_time": pickup_time,
            "carry_end": carry_end,
            "origin_x": ox_pre,
            "origin_y": oy_pre,
            "pickup_x": pkx,
            "pickup_y": pky,
            "carry_end_x": cex,
            "carry_end_y": cey,
            "score": score,
            "details": {
                "stationary_before": round(stationary_before, 3),
                "approach": round(approach, 3),
                "joint_motion": round(joint, 3),
                "joint_fraction": round(joint_fraction, 3),
                "mean_velocity_cosine": round(mean_cos, 3),
                "displaced": round(displaced, 3),
                "separation": round(separation, 3),
            },
        }
