"""Natural-language query parsing into a structured query plan (FR-014/015)."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider
from ..llm.providers import RuleBasedProvider
from ..perception.vocabulary import query_terms

_ANCHOR_EVENTS = {
    "disappeared": "disappeared",
    "appeared": "appeared",
    "entered": "entered",
    "exited": "exited",
    "picked up": "picked_up",
    "picked_up": "picked_up",
    "placed": "placed",
    "moved": "moved",
}

_PARSER_PROMPT = (
    "Convert the user's question about a video archive into a JSON query plan.\n"
    "\n"
    "Return ONLY valid JSON with this schema:\n"
    '{"query_type": "locate|history|actor|event|change|state|describe|general",\n'
    ' "target_category": "<category or null>",\n'
    ' "attributes": ["<adjective>"],\n'
    ' "relationship": "<moved|picked_up|placed|carried|approached|null>",\n'
    ' "temporal": {"op": "none|before|after|during|between|first|last|nearest",\n'
    '              "anchor_event": "<event or null>",\n'
    '              "anchor_event_end": "<event or null>",\n'
    '              "start": null, "end": null}}\n'
    "\n"
    "Use canonical categories: notebook, mug, backpack, laptop, phone, person,\n"
    "chair, bottle, tool, desk, shelf, coin, tablet.\n"
    "\n"
    "Question: __QUERY__"
)


class TemporalSpec(BaseModel):
    op: str = "none"
    anchor_event: str | None = None
    anchor_event_end: str | None = None
    start: float | None = None
    end: float | None = None


class QueryPlan(BaseModel):
    query_type: str = "general"
    target_category: str | None = None
    attributes: list[str] = Field(default_factory=list)
    relationship: str | None = None
    temporal: TemporalSpec | None = None


def _mk(
    qtype: str,
    category: str | None,
    attributes: list[str],
    temporal: TemporalSpec | None,
    relationship: str | None = None,
) -> QueryPlan:
    return QueryPlan(
        query_type=qtype,
        target_category=category,
        attributes=attributes,
        temporal=temporal,
        relationship=relationship,
    )


class QueryParser:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        vocabulary: list[tuple[str, str]] | None = None,
    ) -> None:
        self.llm = llm or RuleBasedProvider()
        # (term, canonical) pairs, longest first, from the shared vocabulary.
        self._terms = vocabulary if vocabulary is not None else query_terms()

    def parse(self, query: str) -> QueryPlan:
        plan = self._llm_parse(query)
        return plan if plan is not None else self._rule_parse(query)

    def _llm_parse(self, query: str) -> QueryPlan | None:
        try:
            prompt = _PARSER_PROMPT.replace("__QUERY__", query)
            raw = self.llm.complete(prompt).strip()
            if not raw:
                return None
            data = json.loads(_strip_code_fence(raw))
            return QueryPlan.model_validate(data)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _rule_parse(self, query: str) -> QueryPlan:
        q = query.lower()
        matched = next(
            (
                (term, canonical)
                for term, canonical in self._terms
                if re.search(r"\b" + re.escape(term) + r"\b", q)
            ),
            None,
        )
        noun = extract_target_noun(q)
        if matched is None:
            category = noun
        else:
            term, category = matched
            # "phone charger" must not collapse to the known term "phone".
            if (
                noun
                and noun != term
                and re.search(r"\b" + re.escape(term) + r"\b", noun)
            ):
                category = noun
        attributes = [a for a in ("red", "blue", "green", "black", "white") if a in q]
        temporal = QueryParser._parse_temporal(q)

        if "who moved" in q or "who picked" in q or "who carried" in q:
            if "moved" in q:
                rel = "moved"
            elif "picked" in q:
                rel = "picked_up"
            else:
                rel = "carried"
            return _mk("actor", category, attributes, temporal, rel)
        if "where" in q or "last seen" in q or "when" in q:
            return _mk("locate", category, attributes, temporal)
        if (
            "what is shown" in q
            or "what's shown" in q
            or "what do you see" in q
            or "what can you see" in q
            or "describe" in q
            or "what is in the video" in q
            or "what's in the video" in q
            or "tell me about" in q
        ):
            return _mk("describe", category, attributes, temporal)
        if "history" in q or "what happened" in q:
            return _mk("history", category, attributes, temporal)
        if "change" in q:
            return _mk("change", category, attributes, temporal)
        if (
            " on the " in q
            or " on top" in q
            or q.startswith("what was on")
            or "what's on" in q
            or " near " in q
            or "holding" in q
            or " in the hand" in q
            or " in hand" in q
        ):
            return _mk("state", category, attributes, temporal)
        if temporal is not None and temporal.op in (
            "before",
            "after",
            "between",
            "during",
            "nearest",
        ):
            return _mk("event", category, attributes, temporal)
        return _mk("general", category, attributes, temporal)

    @staticmethod
    def _parse_temporal(q: str) -> TemporalSpec | None:
        anchor_end = None
        op = "none"
        if "between" in q and " and " in q:
            op = "between"
        elif "during" in q:
            op = "during"
        elif "before" in q:
            op = "before"
        elif "after" in q:
            op = "after"
        elif "nearest" in q or "immediately" in q:
            op = "nearest"
        elif "last" in q:
            op = "last"
        elif "first" in q:
            op = "first"
        else:
            op = "none"

        anchors = []
        for phrase, value in _ANCHOR_EVENTS.items():
            if phrase in q:
                anchors.append(value)
        anchor = anchors[0] if anchors else None
        if op == "between" and len(anchors) >= 2:
            anchor_end = anchors[1]
        elif op == "between" and len(anchors) == 1:
            anchor_end = anchors[0]

        if op == "none" and anchor is None:
            return None
        return TemporalSpec(op=op, anchor_event=anchor, anchor_event_end=anchor_end)


_NOUN_PATTERNS = [
    re.compile(
        r"\b(?:where(?:'s| is| are| was| were)|find|locate|show me|"
        r"when was|when did)\s+"
        r"(?:(?:the|my|a|an|our|his|her|their)\s+)?([a-z][a-z \-]{1,30}?)"
        r"(?:\s+(?:now|right now|last|before|after|during|on|in|at|"
        r"near|go|get|put|placed|left|kept)\b|\s*[?.!]*$)"
    ),
]
_COLOR_WORDS = {"red", "blue", "green", "black", "white"}
_NOT_OBJECTS = {
    "it", "that", "this", "everything", "anything", "something", "someone",
    "he", "she", "they", "there", "here", "i", "you",
}


def extract_target_noun(q: str) -> str | None:
    """Best-effort noun phrase for objects outside the known vocabulary.

    Lets an open-vocabulary detector be asked for *anything* ("where is my
    charger"): the phrase becomes the target category and the detector is
    prompted with it on demand.
    """
    q = q.strip().lower()
    for pattern in _NOUN_PATTERNS:
        match = pattern.search(q)
        if not match:
            continue
        phrase = match.group(1).strip(" -")
        words = [
            w
            for w in phrase.split()
            if w not in _NOT_OBJECTS and w not in _COLOR_WORDS
        ]
        if 0 < len(words) <= 3:
            return " ".join(words)
    return None


def _strip_code_fence(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text
