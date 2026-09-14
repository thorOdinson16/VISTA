"""API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..reasoning.answerer import Answer


class QueryRequest(BaseModel):
    question: str


class IngestResult(BaseModel):
    video: str
    objects: int
    events: int
    frames_processed: int
    frames_sampled: int
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    query_type: str
    target_category: str | None = None
    answer: Answer
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class StatusResponse(BaseModel):
    video: str | None = None
    objects: int = 0
    events: int = 0
    fps: float | None = None
    duration: float | None = None
    detector_mode: str | None = None
    detector_name: str | None = None
    saliency_fallback: bool = False


class TimelineResponse(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    duration: float | None = None
