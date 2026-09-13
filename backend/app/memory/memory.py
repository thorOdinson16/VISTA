"""VisualMemory: orchestration facade over object/scene/event/graph stores."""

from __future__ import annotations

from pathlib import Path

from ..retrieval.vector import VectorStore
from .event_memory import EventExtractor, EventMemory
from .evidence_memory import EvidenceMemory
from .graph import RelationshipGraph
from .models import EventType, Track
from .object_memory import ObjectMemory
from .scene_memory import SceneMemory
from .store import MemoryStore

_PREDICATES = {
    EventType.OBJECT_PICKED_UP: "picked_up",
    EventType.OBJECT_PLACED: "placed",
    EventType.OBJECT_CARRIED: "carried",
    EventType.OBJECT_APPROACHED: "approached",
}


class VisualMemory:
    def __init__(
        self,
        video_duration: float | None = None,
        store_path: str | Path | None = None,
        extractor: EventExtractor | None = None,
        source_video: str = "",
        fps: float = 30.0,
    ) -> None:
        self.object_memory = ObjectMemory()
        self.scene_memory = SceneMemory()
        self.event_memory = EventMemory()
        self.evidence_memory = EvidenceMemory(source_video=source_video, fps=fps)
        self.graph = RelationshipGraph()
        self.extractor = extractor or EventExtractor(video_duration=video_duration)
        self.store = MemoryStore(store_path) if store_path else None
        # OpenCV appearance descriptors indexed for visual similarity search.
        self.visual_index: VectorStore | None = None

    def set_source(self, source_video: str, fps: float) -> None:
        self.evidence_memory.set_source(source_video, fps)

    def update(self, frame_id: int, timestamp: float, tracks: list[Track]) -> None:
        for track in tracks:
            self.object_memory.upsert(track)
        self.scene_memory.snapshot(frame_id, timestamp, tracks)

    def finalize(self) -> list:
        for event in self.extractor.extract(self.object_memory.all()):
            self.event_memory.add(event)
        self._attach_evidence()
        self._link_object_events()
        self._build_relationships()
        self._build_visual_index()
        return self.event_memory.all()

    def _build_visual_index(self) -> None:
        """Index object appearance descriptors for visual similarity search."""
        indexed = [
            obj
            for obj in self.object_memory.all()
            if obj.appearance_embedding
        ]
        if not indexed:
            self.visual_index = None
            return
        dim = len(indexed[0].appearance_embedding)
        store = VectorStore(dim=dim)
        for obj in indexed:
            vec = obj.appearance_embedding
            if vec and len(vec) == dim:
                store.add(obj.object_id, vec)
        self.visual_index = store

    def similar_objects(
        self, object_id: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """Return visually similar objects by OpenCV appearance descriptor."""
        obj = self.object_memory.get(object_id)
        if obj is None or not obj.appearance_embedding:
            return []
        if self.visual_index is None:
            self._build_visual_index()
        if self.visual_index is None:
            return []
        results = self.visual_index.search(obj.appearance_embedding, k=k + 1)
        return [(oid, score) for oid, score in results if oid != object_id][:k]

    def _attach_evidence(self) -> None:
        """Create one stored evidence record per event and per observation.

        This is the single place evidence IDs come from: extraction time.
        Retrieval and answering only look these records up by ID.
        """
        for event in self.event_memory.all():
            if not event.evidence_refs:
                rec = self.evidence_memory.record_for_event(event)
                event.evidence_refs.append(rec.evidence_id)
        for obj in self.object_memory.all():
            for obs in obj.observations:
                if not obs.evidence_refs:
                    rec = self.evidence_memory.record_for_observation(obs)
                    obs.evidence_refs.append(rec.evidence_id)

    def _link_object_events(self) -> None:
        for obj in self.object_memory.all():
            refs = [
                e.event_id
                for e in self.event_memory.all()
                if obj.object_id in e.subject_refs or obj.object_id in e.actor_refs
            ]
            obj.event_refs = refs

    def _build_relationships(self) -> None:
        for event in self.event_memory.all():
            subject = event.subject_refs[0] if event.subject_refs else None
            actor = event.actor_refs[0] if event.actor_refs else None
            if subject is None or actor is None:
                continue
            predicate = _PREDICATES.get(event.type)
            if predicate is None:
                continue
            self.graph.add(
                actor,
                predicate,
                subject,
                timestamp=event.start_time,
                confidence=event.confidence,
                evidence_refs=list(event.evidence_refs),
            )

    def save(self) -> None:
        if self.store is None:
            raise RuntimeError("VisualMemory was not constructed with a store_path")
        self.store.save(
            self.object_memory.all(),
            self.event_memory.all(),
            self.graph.all(),
            self.evidence_memory.all(),
            self.scene_memory.all(),
        )
        if self.visual_index is not None and len(self.visual_index) > 0:
            self.visual_index.save(self.store.vector_path)

    @classmethod
    def load(cls, path: str | Path) -> VisualMemory:
        store = MemoryStore(path)
        memory = cls()
        memory.store = store
        for obj in store.load_objects():
            memory.object_memory.upsert_from_object(obj)
        for event in store.load_events():
            memory.event_memory.add(event)
        memory.event_memory.reseed()
        memory.evidence_memory.restore(store.load_evidence())
        memory.scene_memory.extend(store.load_scenes())
        memory.graph._relationships = store.load_relationships()
        if store.vector_path.exists():
            vector_store = VectorStore(dim=1)
            vector_store.load(store.vector_path)
            memory.visual_index = vector_store
        return memory
