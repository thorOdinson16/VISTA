"""Temporal operators over event lists (FR-016)."""

from __future__ import annotations

from ..memory.models import EventRecord


def _sorted(events: list[EventRecord]) -> list[EventRecord]:
    return sorted(events, key=lambda e: e.start_time)


def before(events: list[EventRecord], t: float) -> list[EventRecord]:
    return [e for e in events if e.start_time < t]


def after(events: list[EventRecord], t: float) -> list[EventRecord]:
    return [e for e in events if e.start_time > t]


def during(
    events: list[EventRecord], start: float, end: float
) -> list[EventRecord]:
    return [e for e in events if start <= e.start_time <= end]


def between(
    events: list[EventRecord], start: float, end: float
) -> list[EventRecord]:
    return during(events, start, end)


def first(events: list[EventRecord]) -> EventRecord | None:
    ordered = _sorted(events)
    return ordered[0] if ordered else None


def last(events: list[EventRecord]) -> EventRecord | None:
    ordered = _sorted(events)
    return ordered[-1] if ordered else None


def nearest(events: list[EventRecord], t: float) -> EventRecord | None:
    if not events:
        return None
    return min(events, key=lambda e: abs(e.start_time - t))
