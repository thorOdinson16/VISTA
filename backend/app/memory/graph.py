"""Relationship graph between entities (FR-012)."""

from __future__ import annotations

from .models import Relationship


class RelationshipGraph:
    def __init__(self) -> None:
        self._relationships: list[Relationship] = []

    def add(
        self,
        subject: str,
        predicate: str,
        object: str,
        timestamp: float | None = None,
        confidence: float = 0.0,
        evidence_refs: list[str] | None = None,
    ) -> Relationship:
        rel = Relationship(
            subject=subject,
            predicate=predicate,
            object=object,
            timestamp=timestamp,
            confidence=confidence,
            evidence_refs=evidence_refs or [],
        )
        self._relationships.append(rel)
        return rel

    def all(self) -> list[Relationship]:
        return list(self._relationships)

    def by_subject(self, subject: str) -> list[Relationship]:
        return [r for r in self._relationships if r.subject == subject]

    def by_object(self, object: str) -> list[Relationship]:
        return [r for r in self._relationships if r.object == object]

    def by_predicate(self, predicate: str) -> list[Relationship]:
        return [r for r in self._relationships if r.predicate == predicate]

    def has_edge(self, subject: str, predicate: str, object: str) -> bool:
        return any(
            r.subject == subject and r.predicate == predicate and r.object == object
            for r in self._relationships
        )
