"""SQLite persistence for VISTA memory (structured DB)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import JSON, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .models import EventRecord, EvidenceRecord, ObjectRecord, Relationship, SceneState

Base = declarative_base()


class ObjectRow(Base):
    __tablename__ = "objects"

    object_id = Column(String, primary_key=True)
    category = Column(String)
    data = Column(JSON)


class EventRow(Base):
    __tablename__ = "events"

    event_id = Column(String, primary_key=True)
    type = Column(String)
    start_time = Column(Float)
    data = Column(JSON)


class RelationshipRow(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String)
    predicate = Column(String)
    object = Column(String)
    data = Column(JSON)


class EvidenceRow(Base):
    __tablename__ = "evidence"

    evidence_id = Column(String, primary_key=True)
    source_video = Column(String)
    data = Column(JSON)


class SceneRow(Base):
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(Integer)
    timestamp = Column(Float)
    data = Column(JSON)


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(bind=self.engine)

    @property
    def vector_path(self) -> Path:
        """Sidecar file for the visual (appearance) vector index."""
        return self.path.with_suffix(".vectors.json")

    def _replace_objects(self, objects: list[ObjectRecord]) -> None:
        with self._session() as session:
            session.query(ObjectRow).delete()
            for obj in objects:
                session.add(
                    ObjectRow(
                        object_id=obj.object_id,
                        category=obj.category,
                        data=obj.model_dump(mode="json"),
                    )
                )
            session.commit()

    def _replace_events(self, events: list[EventRecord]) -> None:
        with self._session() as session:
            session.query(EventRow).delete()
            for event in events:
                session.add(
                    EventRow(
                        event_id=event.event_id,
                        type=event.type.value,
                        start_time=event.start_time,
                        data=event.model_dump(mode="json"),
                    )
                )
            session.commit()

    def _replace_relationships(self, relationships: list[Relationship]) -> None:
        with self._session() as session:
            session.query(RelationshipRow).delete()
            for rel in relationships:
                session.add(
                    RelationshipRow(
                        subject=rel.subject,
                        predicate=rel.predicate,
                        object=rel.object,
                        data=rel.model_dump(mode="json"),
                    )
                )
            session.commit()

    def _replace_evidence(self, evidence: list[EvidenceRecord]) -> None:
        with self._session() as session:
            session.query(EvidenceRow).delete()
            for ev in evidence:
                session.add(
                    EvidenceRow(
                        evidence_id=ev.evidence_id,
                        source_video=ev.source_video,
                        data=ev.model_dump(mode="json"),
                    )
                )
            session.commit()

    def _replace_scenes(self, scenes: list[SceneState]) -> None:
        with self._session() as session:
            session.query(SceneRow).delete()
            for state in scenes:
                session.add(
                    SceneRow(
                        frame_id=state.frame_id,
                        timestamp=state.timestamp,
                        data=state.model_dump(mode="json"),
                    )
                )
            session.commit()

    def save(
        self,
        objects: list[ObjectRecord],
        events: list[EventRecord],
        relationships: list[Relationship],
        evidence: list[EvidenceRecord],
        scenes: list[SceneState] | None = None,
    ) -> None:
        self._replace_objects(objects)
        self._replace_events(events)
        self._replace_relationships(relationships)
        self._replace_evidence(evidence)
        self._replace_scenes(scenes or [])

    def load_objects(self) -> list[ObjectRecord]:
        with self._session() as session:
            rows = session.query(ObjectRow).all()
            return [ObjectRecord.model_validate(r.data) for r in rows]

    def load_events(self) -> list[EventRecord]:
        with self._session() as session:
            rows = session.query(EventRow).all()
            return [EventRecord.model_validate(r.data) for r in rows]

    def load_relationships(self) -> list[Relationship]:
        with self._session() as session:
            rows = session.query(RelationshipRow).all()
            return [Relationship.model_validate(r.data) for r in rows]

    def load_evidence(self) -> list[EvidenceRecord]:
        with self._session() as session:
            rows = session.query(EvidenceRow).all()
            return [EvidenceRecord.model_validate(r.data) for r in rows]

    def load_scenes(self) -> list[SceneState]:
        with self._session() as session:
            rows = (
                session.query(SceneRow).order_by(SceneRow.timestamp).all()
            )
            return [SceneState.model_validate(r.data) for r in rows]
