"""Evidence memory: stable, stored evidence records (FR-013, AR-004).

Evidence is created once -- at event-extraction/finalize time -- with a
stable ID, stored in memory, persisted to SQLite, and resolved by lookup
afterwards. Nothing downstream (retrieval, answering, frontend) invents
evidence IDs; they only reference records that actually exist.
"""

from __future__ import annotations

from .models import EventRecord, EvidenceRecord, Observation


class EvidenceMemory:
    def __init__(self, source_video: str = "", fps: float = 30.0) -> None:
        self.source_video = source_video
        self.fps = fps if fps and fps > 0 else 30.0
        self._records: dict[str, EvidenceRecord] = {}
        self._counter = 0

    def set_source(self, source_video: str, fps: float) -> None:
        self.source_video = source_video
        if fps and fps > 0:
            self.fps = fps

    def _frame(self, t: float) -> int:
        return int(round(t * self.fps))

    def _new_id(self) -> str:
        evidence_id = f"evd_{self._counter}"
        self._counter += 1
        return evidence_id

    def record(
        self,
        frame_start: int,
        frame_end: int | None,
        timestamp_start: float,
        timestamp_end: float | None,
        object_refs: list[str] | None = None,
        event_refs: list[str] | None = None,
    ) -> EvidenceRecord:
        rec = EvidenceRecord(
            evidence_id=self._new_id(),
            source_video=self.source_video,
            frame_start=frame_start,
            frame_end=frame_end,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            object_refs=list(object_refs or []),
            event_refs=list(event_refs or []),
        )
        self._records[rec.evidence_id] = rec
        return rec

    def record_for_event(self, event: EventRecord) -> EvidenceRecord:
        end = event.end_time
        end_frame = (
            self._frame(end) if end is not None else self._frame(event.start_time)
        )
        return self.record(
            frame_start=self._frame(event.start_time),
            frame_end=end_frame,
            timestamp_start=event.start_time,
            timestamp_end=end,
            object_refs=list(event.subject_refs) + list(event.actor_refs),
            event_refs=[event.event_id],
        )

    def record_for_observation(self, obs: Observation) -> EvidenceRecord:
        return self.record(
            frame_start=obs.frame_id,
            frame_end=obs.frame_id,
            timestamp_start=obs.timestamp,
            timestamp_end=obs.timestamp,
            object_refs=[obs.object_id],
            event_refs=list(obs.evidence_refs),
        )

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def all(self) -> list[EvidenceRecord]:
        return list(self._records.values())

    def by_event(self, event_id: str) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if event_id in r.event_refs]

    def by_object(self, object_id: str) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if object_id in r.object_refs]

    def restore(self, records: list[EvidenceRecord]) -> None:
        for rec in records:
            self._records[rec.evidence_id] = rec
        self.reseed()

    def reseed(self) -> None:
        """Restore the ID counter after loading persisted records."""
        best = -1
        for eid in self._records:
            if eid.startswith("evd_"):
                try:
                    best = max(best, int(eid.split("_", 1)[1]))
                except ValueError:
                    continue
        self._counter = max(self._counter, best + 1)

    def __len__(self) -> int:
        return len(self._records)
