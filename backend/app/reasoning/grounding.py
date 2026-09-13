"""Evidence grounding: ensure answers trace back to evidence (FR-018).

Validation is structural, not cosmetic: every evidence reference in the
answer must resolve to a stored record, that record must reference the
event/object the answer claims it supports, and an answer with claims
but no supporting evidence is rewritten as an explicit uncertainty
statement instead of merely having its confidence lowered.
"""

from __future__ import annotations

from .answerer import Answer


def ensure_grounded(answer: Answer, result) -> Answer:
    valid_ids = set()
    evidence_by_id = {}
    for ev in result.evidence or []:
        eid = getattr(ev, "evidence_id", None)
        if eid:
            valid_ids.add(eid)
            evidence_by_id[eid] = ev

    # 1. Drop invented references: only stored, retrieved evidence counts.
    answer.evidence_refs = [r for r in answer.evidence_refs if r in valid_ids]

    # 2. Every cited record must actually support a retrieved event/object.
    known_events = {e.event_id for e in result.events or []}
    known_objects = {o.object_id for o in result.objects or []}
    kept = []
    for ref in answer.evidence_refs:
        rec = evidence_by_id[ref]
        refs = set(getattr(rec, "event_refs", []) or []) | set(
            getattr(rec, "object_refs", []) or []
        )
        if refs & known_events or refs & known_objects:
            kept.append(ref)
    answer.evidence_refs = kept

    has_claims = bool(answer.observed or answer.inferred)
    if has_claims and not answer.evidence_refs:
        # 3. Claims without evidence become explicit uncertainty.
        answer.confidence = min(answer.confidence, 0.4)
        if "No direct evidence" not in answer.text:
            answer.text += " (No direct evidence available.)"
    if has_claims and not result.events and not result.objects:
        answer.text = (
            "I cannot determine this from the available visual evidence."
        )
        answer.observed = []
        answer.inferred = []
        answer.confidence = 0.0
        answer.timeline = []
    return answer
