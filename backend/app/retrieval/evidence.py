"""Evidence resolution by lookup of stored records (FR-013/018).

The resolver never invents evidence IDs: every record it returns was
created at finalize time, stored in ``VisualMemory.evidence_memory``,
and carries the source video, frame range, timestamp range, object refs
and event refs needed to locate the source.
"""

from __future__ import annotations

from ..memory.models import EventRecord, EvidenceRecord, ObjectRecord


class EvidenceResolver:
    def __init__(self, memory=None, source_video: str = "", fps: float = 30.0) -> None:
        self.memory = memory
        self.source_video = source_video
        self.fps = fps

    def _store(self):
        if self.memory is not None:
            return self.memory.evidence_memory
        return None

    def for_event(self, event: EventRecord) -> EvidenceRecord | None:
        """Look up the stored evidence for an event (None if absent)."""
        store = self._store()
        if store is not None:
            for eid in event.evidence_refs:
                rec = store.get(eid)
                if rec is not None:
                    return rec
            found = store.by_event(event.event_id)
            return found[0] if found else None
        return _synthesize(event, self.source_video, self.fps)

    def for_event_all(self, event: EventRecord) -> list[EvidenceRecord]:
        store = self._store()
        if store is not None:
            recs = [store.get(eid) for eid in event.evidence_refs]
            recs = [r for r in recs if r is not None]
            if recs:
                return recs
            return store.by_event(event.event_id)
        rec = _synthesize(event, self.source_video, self.fps)
        return [rec] if rec else []

    def for_object(self, obj: ObjectRecord) -> list[EvidenceRecord]:
        store = self._store()
        if store is not None:
            return store.by_object(obj.object_id)
        return []

    def for_object_last(self, obj: ObjectRecord) -> EvidenceRecord | None:
        """Latest stored evidence for an object (None when never observed)."""
        recs = self.for_object(obj)
        if not recs:
            return None
        return max(recs, key=lambda r: r.timestamp_start)


def _synthesize(
    event: EventRecord, source_video: str, fps: float
) -> EvidenceRecord | None:
    """Fallback frame math when no memory is attached (tests/utilities).

    This is the only place a record is constructed outside finalize, and
    it is never used in the answering path.
    """
    if fps is None or fps <= 0:
        return None
    end = event.end_time
    return EvidenceRecord(
        evidence_id=event.event_id,
        source_video=source_video,
        frame_start=int(round(event.start_time * fps)),
        frame_end=int(round((end if end is not None else event.start_time) * fps)),
        timestamp_start=event.start_time,
        timestamp_end=end,
        object_refs=list(event.subject_refs) + list(event.actor_refs),
        event_refs=[event.event_id],
    )
