"""Answer synthesis: grounded, uncertainty-aware responses (FR-018/021/022)."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider
from ..llm.providers import RuleBasedProvider
from ..memory.models import EventRecord, EventType
from ..pipeline.pipeline import POSTER_FRAME
from ..retrieval.query_parser import QueryPlan
from ..retrieval.retriever import RetrievalResult

_LABELS = {
    EventType.OBJECT_APPEARED: "appeared",
    EventType.OBJECT_DISAPPEARED: "disappeared",
    EventType.PERSON_ENTERED: "entered",
    EventType.PERSON_EXITED: "exited",
    EventType.OBJECT_MOVED: "moved",
    EventType.OBJECT_PICKED_UP: "picked up",
    EventType.OBJECT_PLACED: "placed",
    EventType.OBJECT_CARRIED: "carried",
    EventType.OBJECT_APPROACHED: "approached",
    EventType.SCENE_CHANGED: "scene changed",
}

_ANSWER_PROMPT = (
    "Answer the user's question about a video archive using ONLY the retrieved "
    "evidence below.\n"
    "\n"
    "Rules:\n"
    "- Distinguish OBSERVED facts from INFERRED conclusions.\n"
    "- Do not invent facts. If evidence is insufficient, say so.\n"
    "- Return ONLY JSON with this schema:\n"
    '{"text": "<answer>", "observed": ["<fact>"], "inferred": ["<conclusion>"],\n'
    ' "confidence": <0..1>, "evidence_refs": ["<evidence_id>"],\n'
    ' "timeline": [{"time": <seconds>, "label": "<event>"}]}\n'
    "\n"
    "Retrieved data:\n"
    "__DATA__\n"
    "\n"
    "Question: __QUERY__"
)


# Extra "earlier" sightings listed after the best one in a locate answer.
_MAX_EARLIER = 2


class TimelineEntry(BaseModel):
    time: float
    label: str


class Answer(BaseModel):
    text: str
    observed: list[str] = Field(default_factory=list)
    inferred: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    # Representative scene frame ("here is the frame"), shown by the UI.
    poster: str | None = None


class Answerer:
    def __init__(self, memory=None, llm: LLMProvider | None = None) -> None:
        self.memory = memory
        self.llm = llm or RuleBasedProvider()

    def answer(self, query: str, plan: QueryPlan, result: RetrievalResult) -> Answer:
        answer = self._llm_answer(query, result)
        if answer is None:
            answer = self._template(plan, result)
        from .grounding import ensure_grounded

        return ensure_grounded(answer, result)

    def _llm_answer(self, query: str, result: RetrievalResult) -> Answer | None:
        try:
            data = json.dumps(
                {
                    "objects": [o.model_dump(mode="json") for o in result.objects],
                    "events": [e.model_dump(mode="json") for e in result.events],
                    "evidence": [e.model_dump(mode="json") for e in result.evidence],
                }
            )
            prompt = _ANSWER_PROMPT.replace("__DATA__", data)
            prompt = prompt.replace("__QUERY__", query)
            raw = self.llm.complete(prompt).strip()
            if not raw:
                return None
            return Answer.model_validate(json.loads(_strip_fence(raw)))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _template(self, plan: QueryPlan, result: RetrievalResult) -> Answer:
        evidence_refs = [e.evidence_id for e in result.evidence]
        handler = {
            "locate": self._locate,
            "history": self._history,
            "actor": self._actor,
            "event": self._event,
            "change": self._change,
            "state": self._state,
            "describe": self._describe_scene,
            "general": self._describe_scene,
        }.get(plan.query_type, self._describe_scene)
        return handler(plan, result, evidence_refs)

    def _describe(self, event: EventRecord) -> str:
        base = _LABELS.get(event.type, event.type.value)
        if event.type is EventType.SCENE_CHANGED:
            return "the scene changed"
        subject = event.subject_refs[0] if event.subject_refs else "object"
        actor = event.actor_refs[0] if event.actor_refs else None
        subject_label = self._entity_label(subject)
        actor_label = self._person_label(actor) if actor else None
        if actor_label:
            return f"{actor_label} {base} {subject_label}"
        return f"{subject_label} {base}"

    def _entity_label(self, object_id: str) -> str:
        if self.memory:
            obj = self.memory.object_memory.get(object_id)
            if obj is not None:
                if obj.category == "person":
                    return self._person_label(object_id)
                color = obj.attributes.get("color") if obj.attributes else None
                if color:
                    return f"{color} {obj.category}"
                return obj.category
        return object_id

    @staticmethod
    def _split(
        events: list[EventRecord], labels: list[str]
    ) -> tuple[list[str], list[str]]:
        observed = [
            lb for e, lb in zip(events, labels, strict=True) if e.kind == "observed"
        ]
        inferred = [
            lb for e, lb in zip(events, labels, strict=True) if e.kind != "observed"
        ]
        return observed, inferred

    def _person_label(self, object_id: str) -> str:
        persons = sorted(
            o.object_id
            for o in (self.memory.object_memory.all() if self.memory else [])
            if o.category == "person"
        )
        if object_id in persons:
            return f"Person {chr(65 + persons.index(object_id))}"
        if self.memory:
            obj = self.memory.object_memory.get(object_id)
            return obj.category if obj else object_id
        return object_id

    def _locate(self, plan, result, evidence_refs) -> Answer:
        observed: list[str] = []
        parts: list[str] = []
        for rank, obj in enumerate(result.objects):
            loc = ""
            if obj.trajectory:
                p = obj.trajectory[-1]
                loc = f" at ({p.x:.0f}, {p.y:.0f})"
            t = obj.last_seen
            if rank == 0:
                parts.append(f"The {obj.category} was last seen at {t:.1f}s{loc}.")
            elif rank <= _MAX_EARLIER:
                parts.append(f"Earlier: {t:.1f}s{loc}.")
            observed.append(f"The {obj.category} is visible at {t:.1f}s.")
        extra = len(result.objects) - 1 - _MAX_EARLIER
        if parts and extra > 0:
            parts.append(f"({extra} more tracked sighting(s) not listed.)")
        if not parts:
            category = plan.target_category or "object"
            return Answer(
                text=(
                    f"I cannot determine where the {category} was last seen "
                    "from the available visual evidence."
                ),
                confidence=0.0,
                evidence_refs=evidence_refs,
            )
        text = " ".join(parts)
        return Answer(
            text=text, observed=observed, confidence=result.confidence,
            evidence_refs=evidence_refs,
        )

    def _history(self, plan, result, evidence_refs) -> Answer:
        category = plan.target_category or "object"
        events = result.events
        labels = [self._describe(e) for e in events]
        observed, inferred = self._split(events, labels)
        timeline = [
            TimelineEntry(time=e.start_time, label=lb)
            for e, lb in zip(events, labels, strict=True)
        ]
        if events:
            first = events[0].start_time
            last = events[-1].start_time
            text = (
                f"The {category} has {len(events)} recorded event(s) "
                f"between {first:.1f}s and {last:.1f}s."
            )
        else:
            text = (
                f"I cannot determine what happened to the {category} "
                "from the available visual evidence."
            )
        return Answer(
            text=text, observed=observed, inferred=inferred,
            confidence=result.confidence if events else 0.0,
            evidence_refs=evidence_refs,
            timeline=timeline,
        )

    def _fact_label(self, fact: dict) -> str:
        subject = self._entity_label(fact.get("subject", "object"))
        other = self._entity_label(fact.get("object", "object"))
        predicate = fact.get("predicate", "")
        if predicate == "ON":
            return f"{subject} was on {other}"
        if predicate == "NEAR":
            return f"{subject} was near {other}"
        if predicate == "IN_HAND_OF":
            return f"{subject} was held by {other}"
        return f"{subject} {predicate.lower()} {other}"

    def _state(self, plan, result, evidence_refs) -> Answer:
        facts = getattr(result, "facts", None) or []
        if not facts:
            category = plan.target_category or "scene"
            return Answer(
                text=(
                    f"I cannot determine the state of the {category} "
                    "from the available visual evidence."
                ),
                confidence=0.0,
                evidence_refs=evidence_refs,
            )
        observed = [self._fact_label(f) for f in facts]
        prefix = (
            f"At {result.facts_time:.1f}s"
            if getattr(result, "facts_time", None) is not None
            else "In the scene"
        )
        return Answer(
            text=f"{prefix}: " + "; ".join(observed) + ".",
            observed=observed,
            confidence=result.confidence,
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def _poster_url() -> str:
        return f"/evidence/{POSTER_FRAME}"

    def _object_summary_label(self, obj) -> str:
        if obj.category == "object":
            return "an unclassified object"
        return self._entity_label(obj.object_id)

    def _describe_scene(self, plan, result, evidence_refs) -> Answer:
        objects = result.objects
        if not objects:
            category = plan.target_category
            subject = f"the {category}" if category else "any configured object"
            return Answer(
                text=(
                    f"No configured objects were detected for {subject}. "
                    "The active detector does not match this footage; "
                    "here is a representative frame."
                ),
                confidence=0.0,
                evidence_refs=evidence_refs,
                poster=self._poster_url(),
            )

        parts: list[str] = []
        observed: list[str] = []
        for obj in objects:
            label = self._object_summary_label(obj)
            if obj.first_seen is not None and obj.last_seen is not None:
                parts.append(f"{label} ({obj.first_seen:.1f}-{obj.last_seen:.1f}s)")
                observed.append(
                    f"{label} is visible from {obj.first_seen:.1f}s "
                    f"to {obj.last_seen:.1f}s."
                )
            else:
                parts.append(label)
            color = obj.attributes.get("color") if obj.attributes else None
            if color:
                observed.append(f"{label} is {color}.")

        text = "I can see: " + "; ".join(parts) + "."
        facts = getattr(result, "facts", None) or []
        if facts:
            fact_labels = [self._fact_label(f) for f in facts]
            text += " " + " ".join(f"{label}." for label in fact_labels)
            observed.extend(fact_labels)
        return Answer(
            text=text,
            observed=observed,
            confidence=result.confidence,
            evidence_refs=evidence_refs,
            poster=self._poster_url(),
        )

    def _actor(self, plan, result, evidence_refs) -> Answer:
        category = plan.target_category or "object"
        verb = (plan.relationship or "moved").replace("_", " ")
        actors = {a for e in result.events for a in e.actor_refs}
        # Only actors from genuinely inferential events count; a bare
        # proximity (observed) event never identifies a mover.
        inferential = {
            a
            for e in result.events
            if e.kind == "inferred"
            for a in e.actor_refs
        }
        actors = actors & inferential if result.events else set()
        if actors:
            labels = [self._person_label(a) for a in sorted(actors)]
            inferred = [f"{label} {verb} the {category}." for label in labels]
            return Answer(
                text=" ".join(inferred), inferred=inferred,
                confidence=result.confidence, evidence_refs=evidence_refs,
            )
        return Answer(
            text=(
                f"I cannot determine who {verb} the {category} "
                "from the available visual evidence."
            ),
            confidence=0.0, evidence_refs=evidence_refs,
        )

    def _event(self, plan, result, evidence_refs) -> Answer:
        category = plan.target_category or "object"
        return self._summarize(
            category, result.events, result.confidence, evidence_refs
        )

    def _change(self, plan, result, evidence_refs) -> Answer:
        if not result.events:
            return Answer(
                text=(
                    "I cannot determine what changed "
                    "from the available visual evidence."
                ),
                confidence=0.0,
                evidence_refs=evidence_refs,
            )
        text = f"{len(result.events)} change(s) detected."
        return self._summarize(
            None, result.events, result.confidence, evidence_refs, prefix=text
        )

    def _summarize(
        self, category, events, confidence, evidence_refs, prefix=None
    ) -> Answer:
        labels = [self._describe(e) for e in events]
        observed, inferred = self._split(events, labels)
        timeline = [
            TimelineEntry(time=e.start_time, label=lb)
            for e, lb in zip(events, labels, strict=True)
        ]
        if not events:
            text = (
                f"I cannot determine what happened to the {category} "
                "from the available visual evidence."
                if category
                else "I cannot determine what happened "
                "from the available visual evidence."
            )
            return Answer(text=text, confidence=0.0, evidence_refs=evidence_refs)
        if prefix:
            text = f"{prefix} " + "; ".join(labels[:5]) + "."
        else:
            text = f"For the {category}: " + "; ".join(labels) + "."
        return Answer(
            text=text, observed=observed, inferred=inferred,
            confidence=confidence, evidence_refs=evidence_refs, timeline=timeline,
        )


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text
