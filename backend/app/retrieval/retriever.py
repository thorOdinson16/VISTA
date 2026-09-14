"""Retrieval: convert a query plan into objects, events, and evidence."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..memory.memory import VisualMemory
from ..memory.models import EventRecord, EventType, ObjectRecord, Relationship
from . import temporal
from .evidence import EvidenceResolver
from .query_parser import QueryPlan

_RELATION_TYPES: dict[str, list[EventType]] = {
    "moved": [
        EventType.OBJECT_PICKED_UP,
        EventType.OBJECT_CARRIED,
        EventType.OBJECT_PLACED,
    ],
    "picked_up": [EventType.OBJECT_PICKED_UP],
    "placed": [EventType.OBJECT_PLACED],
    "carried": [EventType.OBJECT_CARRIED],
    "approached": [EventType.OBJECT_APPROACHED],
}

_RELATION_PREDICATES: dict[str, list[str]] = {
    "moved": ["picked_up", "carried", "placed"],
    "picked_up": ["picked_up"],
    "placed": ["placed"],
    "carried": ["carried"],
    "approached": ["approached"],
}

_ANCHOR_TYPES: dict[str, EventType] = {
    "disappeared": EventType.OBJECT_DISAPPEARED,
    "appeared": EventType.OBJECT_APPEARED,
    "entered": EventType.PERSON_ENTERED,
    "exited": EventType.PERSON_EXITED,
    "picked_up": EventType.OBJECT_PICKED_UP,
    "placed": EventType.OBJECT_PLACED,
    "moved": EventType.OBJECT_MOVED,
    "carried": EventType.OBJECT_CARRIED,
}

_CHANGE_TYPES = [
    EventType.OBJECT_APPEARED,
    EventType.OBJECT_DISAPPEARED,
    EventType.PERSON_ENTERED,
    EventType.PERSON_EXITED,
    EventType.OBJECT_MOVED,
    EventType.OBJECT_PICKED_UP,
    EventType.OBJECT_PLACED,
    EventType.OBJECT_CARRIED,
    EventType.OBJECT_APPROACHED,
    EventType.SCENE_CHANGED,
]


class RetrievalResult(BaseModel):
    objects: list[ObjectRecord] = Field(default_factory=list)
    events: list[EventRecord] = Field(default_factory=list)
    evidence: list = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    # Spatial facts from a scene snapshot: {"subject", "predicate", "object"}.
    facts: list[dict] = Field(default_factory=list)
    facts_time: float | None = None
    confidence: float = 0.0


class Retriever:
    def __init__(
        self,
        memory: VisualMemory,
        source_video: str = "",
        fps: float = 30.0,
        llm=None,
        semantic_threshold: float = 0.85,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.semantic_threshold = semantic_threshold
        if source_video:
            memory.evidence_memory.source_video = source_video
        if fps:
            memory.evidence_memory.fps = fps
        self.evidence = EvidenceResolver(memory=memory)

    def retrieve(self, plan: QueryPlan) -> RetrievalResult:
        return {
            "locate": self._locate,
            "history": self._history,
            "actor": self._actor,
            "event": self._event,
            "change": self._change,
            "state": self._state,
            "describe": self._describe,
            "general": self._general,
        }.get(plan.query_type, self._general)(plan)

    def _match_objects(
        self, category: str | None, attributes: list[str] | None = None
    ) -> list[ObjectRecord]:
        """Match by category *and* attributes.

        A requested attribute (e.g. "red") only matches an object whose
        stored attributes establish that value. Unknown attributes never
        match silently -- the caller sees an empty list and must report
        uncertainty instead of answering about the wrong object.
        """
        wanted = [a.lower() for a in (attributes or [])]
        out = []
        for o in self.memory.object_memory.all():
            if category is not None and o.category != category:
                continue
            if wanted and not _attributes_match(o, wanted):
                continue
            out.append(o)
        return out

    def _objects_for_plan(self, plan: QueryPlan) -> list[ObjectRecord]:
        objects = self._match_objects(plan.target_category, plan.attributes)
        if objects:
            return objects
        return self._semantic_match(plan)

    @staticmethod
    def _object_caption(obj: ObjectRecord) -> str:
        attributes = [str(v) for v in obj.attributes.values()]
        return " ".join(attributes + [obj.category.replace("_", " ")])

    def _semantic_match(self, plan: QueryPlan) -> list[ObjectRecord]:
        """Resolve a category via provider embeddings when exact match fails.

        Only used when there is a target category and no explicit attribute
        filter: an explicit filter that matched nothing (e.g. "green
        notebook" when only red exists) must stay unresolved rather than be
        overridden by a semantically-similar object. Returns [] whenever no
        embedding provider is available.
        """
        if (
            self.llm is None
            or not plan.target_category
            or plan.attributes
        ):
            return []
        objects = self.memory.object_memory.all()
        if not objects:
            return []
        texts = [self._object_caption(o) for o in objects] + [plan.target_category]
        vectors = self.llm.embed(texts)
        if not vectors or len(vectors) != len(texts):
            return []

        import numpy as np

        query = np.asarray(vectors[-1], dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []
        query = query / query_norm

        scored: list[tuple[float, ObjectRecord]] = []
        for obj, vector in zip(objects, vectors[:-1], strict=True):
            candidate = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(candidate))
            if norm == 0:
                continue
            similarity = float(query @ (candidate / norm))
            if similarity >= self.semantic_threshold:
                scored.append((similarity, obj))
        scored.sort(key=lambda pair: -pair[0])
        return [obj for _, obj in scored]

    def _events_for(self, obj: ObjectRecord) -> list[EventRecord]:
        return [
            e for e in self.memory.event_memory.all() if obj.object_id in e.subject_refs
        ]

    # -- temporal --------------------------------------------------------

    def _resolve_anchor_events(self, plan: QueryPlan) -> list[EventRecord]:
        """Resolve anchor events against the target entity first.

        The anchor is searched among events referencing the resolved
        object(s), so "before the notebook disappeared" uses *the
        notebook's* disappearance -- not the first disappearance globally.
        """
        spec = plan.temporal
        if spec is None or not spec.anchor_event:
            return []
        event_type = _ANCHOR_TYPES.get(spec.anchor_event or "")
        if event_type is None:
            return []
        objects = self._objects_for_plan(plan)
        if objects:
            ids = {o.object_id for o in objects}
            scoped = [
                e
                for e in self.memory.event_memory.all()
                if e.type is event_type
                and any(s in ids for s in e.subject_refs)
            ]
            if scoped:
                return sorted(scoped, key=lambda e: e.start_time)
        # Fall back to unscoped only when no target entity resolved.
        return sorted(
            [e for e in self.memory.event_memory.all() if e.type is event_type],
            key=lambda e: e.start_time,
        )

    def _apply_temporal(
        self, events: list[EventRecord], plan: QueryPlan
    ) -> list[EventRecord]:
        spec = plan.temporal
        if spec is None or spec.op == "none":
            return events
        if spec.op == "last":
            ev = temporal.last(events)
            return [ev] if ev else []
        if spec.op == "first":
            ev = temporal.first(events)
            return [ev] if ev else []
        if spec.op == "nearest":
            anchors = self._resolve_anchor_events(plan)
            if not anchors:
                return events
            t = anchors[-1].start_time
            ev = temporal.nearest(events, t)
            return [ev] if ev else []
        if spec.op in ("before", "after"):
            anchors = self._resolve_anchor_events(plan)
            if not anchors:
                return []
            anchor = anchors[-1].start_time
            return (
                temporal.before(events, anchor)
                if spec.op == "before"
                else temporal.after(events, anchor)
            )
        if spec.op in ("between", "during"):
            anchors = self._resolve_anchor_events(plan)
            if not anchors:
                return []
            if spec.op == "between" and spec.anchor_event_end:
                end_type = _ANCHOR_TYPES.get(spec.anchor_event_end or "")
                objects = self._objects_for_plan(plan)
                ids = {o.object_id for o in objects}
                ends = [
                    e
                    for e in self.memory.event_memory.all()
                    if e.type is end_type
                    and (not ids or any(s in ids for s in e.subject_refs))
                ]
                if not ends:
                    return []
                lo, hi = anchors[0].start_time, max(
                    e.start_time for e in ends
                )
            else:
                lo = anchors[0].start_time
                hi = anchors[-1].start_time
                if lo == hi:
                    # Single anchor with during/between: use its own span.
                    ev = anchors[0]
                    hi = ev.end_time if ev.end_time is not None else ev.start_time
            if hi < lo:
                lo, hi = hi, lo
            return temporal.between(events, lo, hi)
        return events


    @staticmethod
    def _mean_confidence(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _evidence_for_events(
        self, events: list[EventRecord]
    ) -> list:
        out = []
        for e in events:
            out.extend(self.evidence.for_event_all(e))
        return out

    def _locate(self, plan: QueryPlan) -> RetrievalResult:
        objects = self._objects_for_plan(plan)
        evidence = [
            e
            for o in objects
            if (e := self.evidence.for_object_last(o)) is not None
        ]
        return RetrievalResult(
            objects=objects,
            evidence=evidence,
            confidence=self._mean_confidence([o.confidence for o in objects]),
        )

    @staticmethod
    def _parse_spatial(relationship: str) -> dict | None:
        parts = relationship.split()
        if len(parts) != 3:
            return None
        return {"subject": parts[0], "predicate": parts[1], "object": parts[2]}

    def _state(self, plan: QueryPlan) -> RetrievalResult:
        """Answer spatial/state questions from a stored scene snapshot."""
        spec = plan.temporal
        at_time = spec.start if spec is not None else None
        state = (
            self.memory.scene_memory.at(at_time)
            if at_time is not None
            else self.memory.scene_memory.latest()
        )
        if state is None:
            return RetrievalResult(confidence=0.0)

        objects = self._objects_for_plan(plan)
        ids = {o.object_id for o in objects}
        facts = []
        for rel in state.spatial_relationships:
            parsed = self._parse_spatial(rel)
            if parsed is None:
                continue
            if ids and not (
                parsed["subject"] in ids or parsed["object"] in ids
            ):
                continue
            facts.append(parsed)

        ref_ids = {f["subject"] for f in facts} | {f["object"] for f in facts}
        ref_objects = [
            obj
            for oid in ref_ids
            if (obj := self.memory.object_memory.get(oid)) is not None
        ]
        evidence = [
            rec
            for obj in ref_objects
            for rec in [self.evidence.for_object_last(obj)]
            if rec is not None
        ]
        return RetrievalResult(
            objects=ref_objects,
            evidence=evidence,
            facts=facts,
            facts_time=state.timestamp,
            confidence=self._mean_confidence([o.confidence for o in ref_objects]),
        )

    def _unique_events(self, events: list[EventRecord]) -> list[EventRecord]:
        seen: set[str] = set()
        unique: list[EventRecord] = []
        for event in events:
            if event.event_id not in seen:
                seen.add(event.event_id)
                unique.append(event)
        return unique

    def _history(self, plan: QueryPlan) -> RetrievalResult:
        objects = self._objects_for_plan(plan)
        events = self._unique_events(
            [e for o in objects for e in self._events_for(o)]
        )
        events = sorted(events, key=lambda e: e.start_time)
        events = self._apply_temporal(events, plan)
        evidence = self._evidence_for_events(events)
        return RetrievalResult(
            objects=objects,
            events=events,
            evidence=evidence,
            confidence=self._mean_confidence([e.confidence for e in events]),
        )

    def _actor(self, plan: QueryPlan) -> RetrievalResult:
        types = _RELATION_TYPES.get(plan.relationship or "moved", [])
        objects = self._objects_for_plan(plan)
        ids = {o.object_id for o in objects}
        events = [
            e
            for e in self.memory.event_memory.all()
            if e.type in types
            and e.actor_refs
            and any(s in ids for s in e.subject_refs)
        ]
        relationships = [
            r
            for r in self.memory.graph.all()
            if r.object in ids
            and r.predicate
            in _RELATION_PREDICATES.get(plan.relationship or "moved", [])
        ]
        evidence = self._evidence_for_events(events)
        return RetrievalResult(
            objects=objects,
            events=events,
            evidence=evidence,
            relationships=relationships,
            confidence=self._mean_confidence([e.confidence for e in events]),
        )

    def _event(self, plan: QueryPlan) -> RetrievalResult:
        objects = self._objects_for_plan(plan)
        if objects:
            scoped = self._unique_events(
                [e for o in objects for e in self._events_for(o)]
            )
        else:
            scoped = list(self.memory.event_memory.all())
        scoped = sorted(scoped, key=lambda e: e.start_time)
        # For temporal questions the anchor events themselves are the
        # subject ("what happened before X"), so include anchor instances
        # in the candidate set.
        events = self._apply_temporal(scoped, plan)
        # Exclude the anchor instance itself from before/after answers.
        anchors = self._resolve_anchor_events(plan)
        if plan.temporal is not None and plan.temporal.op in ("before", "after"):
            anchor_ids = {e.event_id for e in anchors}
            events = [e for e in events if e.event_id not in anchor_ids]
        evidence = self._evidence_for_events(events)
        return RetrievalResult(
            objects=objects,
            events=events,
            evidence=evidence,
            confidence=self._mean_confidence([e.confidence for e in events]),
        )

    def _change(self, plan: QueryPlan) -> RetrievalResult:
        ids = {o.object_id for o in self._objects_for_plan(plan)}
        events = [
            e
            for e in self.memory.event_memory.all()
            if e.type in _CHANGE_TYPES
            and (not ids or any(s in ids for s in e.subject_refs))
        ]
        events = sorted(events, key=lambda e: e.start_time)
        evidence = self._evidence_for_events(events)
        return RetrievalResult(
            objects=self._objects_for_plan(plan),
            events=events,
            evidence=evidence,
            confidence=self._mean_confidence([e.confidence for e in events]),
        )

    def _describe(self, plan: QueryPlan) -> RetrievalResult:
        """Summarize what is visible: all objects, scene facts, evidence."""
        objects = self._match_objects(plan.target_category, plan.attributes)
        state = self.memory.scene_memory.latest()
        facts: list[dict] = []
        if state is not None:
            for rel in state.spatial_relationships:
                parsed = self._parse_spatial(rel)
                if parsed is not None:
                    facts.append(parsed)
        evidence = []
        for obj in objects:
            rec = self.evidence.for_object_last(obj)
            if rec is not None:
                evidence.append(rec)
        return RetrievalResult(
            objects=objects,
            evidence=evidence,
            facts=facts,
            facts_time=state.timestamp if state is not None else None,
            confidence=self._mean_confidence([o.confidence for o in objects]),
        )

    def _general(self, plan: QueryPlan) -> RetrievalResult:
        objects = self._objects_for_plan(plan)
        events = self._unique_events(
            [e for o in objects for e in self._events_for(o)]
        )
        events = sorted(events, key=lambda e: e.start_time)
        evidence = self._evidence_for_events(events)
        return RetrievalResult(objects=objects, events=events, evidence=evidence)


def _attributes_match(obj: ObjectRecord, wanted: list[str]) -> bool:
    values = {str(v).lower() for v in obj.attributes.values()}
    # Also consult per-observation attributes as a fallback.
    for obs in obj.observations:
        values.update(str(v).lower() for v in obs.attributes.values())
    return all(w in values for w in wanted)
