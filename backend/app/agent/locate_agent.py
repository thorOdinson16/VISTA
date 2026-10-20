"""Perception -> decision -> action loop for "where is X?" questions.

A plain memory lookup returns whatever the tracker remembered, including
stray low-confidence boxes and fragments. This agent *checks* the answer
with OpenCV before committing to it, and what it sees decides what it does
next:

    lookup ─► no candidates ─────────────► rescan with the target label
      │
      ▼
    inspect top candidate  (crop its last box from the source frame, upscale,
      │                     re-run the detector on the crop)
      ├─ verified ───────────────────────► answer, confidence from the crop
      ├─ rejected ───────────────────────► inspect the next candidate
      └─ nothing verifies ─► rescan at denser sampling ─► lookup again
                                └─ still nothing ─► ask a human to choose

Every tool call, its numeric observation and the resulting decision is
appended to ``trace`` so the behaviour is observable and auditable. The
agent never fabricates a location: with no verified candidate it returns
``needs_confirmation`` (human in the loop) or ``not_found``.

No LLM is involved: the decisions are driven by detector output, which keeps
the loop deterministic and testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2

from ..ingestion.reader import VideoReader
from ..memory.memory import VisualMemory
from ..memory.models import BoundingBox, ObjectRecord
from ..retrieval.retriever import Retriever
from ..tracking.geometry import iou

VERIFIED = "verified"
NEEDS_CONFIRMATION = "needs_confirmation"
NOT_FOUND = "not_found"


@dataclass
class AgentConfig:
    verify_threshold: float = 0.35  # detector confidence required on the crop
    verify_iou: float = 0.3  # crop detection must overlap the claimed box
    max_candidates: int = 3  # candidates inspected per lookup
    crop_margin: float = 0.25  # context kept around the claimed box
    crop_target: int = 640  # longest crop side after upscaling
    max_upscale: float = 4.0
    dense_fps: float = 6.0  # sampling rate of the denser re-scan
    max_rescans: int = 1


@dataclass
class AgentOutcome:
    status: str
    objects: list[ObjectRecord]  # best first; empty when nothing was found
    memory: VisualMemory
    trace: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)  # for the human prompt
    confidence: float = 0.0
    rescans: int = 0

    @property
    def tool_calls(self) -> int:
        return len(self.trace)


Rescan = Callable[[float, "list[str] | None"], VisualMemory]


class LocateAgent:
    def __init__(
        self,
        memory: VisualMemory,
        detector,
        video_path: str,
        fps: float,
        rescan: Rescan | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.memory = memory
        self.detector = detector
        self.video_path = video_path
        self.fps = fps
        self.rescan = rescan
        self.cfg = config or AgentConfig()
        self.trace: list[dict] = []

    # ------------------------------------------------------------------ tools
    def _log(self, tool: str, args: dict, observation: dict, decision: str) -> None:
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "tool": tool,
                "args": args,
                "observation": observation,
                "decision": decision,
            }
        )

    def _lookup(
        self, category: str, attributes: list[str] | None
    ) -> list[ObjectRecord]:
        retriever = Retriever(self.memory, source_video=self.video_path, fps=self.fps)
        return retriever.rank_sightings(
            retriever._match_objects(category, attributes)  # noqa: SLF001
        )

    def inspect(self, obj: ObjectRecord, category: str) -> dict:
        """Second opinion on an object's last sighting from a zoomed crop."""
        last = max(obj.observations, key=lambda o: o.frame_id)
        box = last.bounding_box
        with VideoReader(self.video_path) as reader:
            frame = reader.read_frame(last.frame_id)
        if frame is None:
            return {"verified": False, "reason": "frame_unreadable", "confidence": 0.0}
        h, w = frame.shape[:2]
        # Square window around the box (side = longest box edge plus context),
        # padded with grey where it leaves the frame. A square keeps the object's
        # proportions when the detector resizes its input to a square; a tall or
        # wide crop would be stretched and thin objects (bottles) would not match.
        side = max(box.width, box.height) * (1 + 2 * self.cfg.crop_margin)
        cx, cy = box.center
        x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
        size = max(8, int(round(side)))
        x2, y2 = x1 + size, y1 + size
        if size < 8 or box.width < 2 or box.height < 2:
            return {"verified": False, "reason": "empty_crop", "confidence": 0.0}
        left, top = max(0, -x1), max(0, -y1)
        right, bottom = max(0, x2 - w), max(0, y2 - h)
        crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            return {"verified": False, "reason": "empty_crop", "confidence": 0.0}
        if left or top or right or bottom:
            crop = cv2.copyMakeBorder(
                crop,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        scale = min(
            self.cfg.max_upscale,
            max(1.0, self.cfg.crop_target / max(crop.shape[:2])),
        )
        zoom = (
            cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if scale > 1.0
            else crop
        )
        best, best_iou = 0.0, 0.0
        for det in self.detector.detect(zoom, last.frame_id, last.timestamp):
            if det.category != category:
                continue
            b = det.bounding_box
            mapped = BoundingBox(
                x_min=x1 + b.x_min / scale,
                y_min=y1 + b.y_min / scale,
                x_max=x1 + b.x_max / scale,
                y_max=y1 + b.y_max / scale,
            )
            overlap = iou(mapped, box)
            if overlap >= self.cfg.verify_iou and det.confidence > best:
                best, best_iou = det.confidence, overlap
        return {
            "verified": best >= self.cfg.verify_threshold,
            "confidence": round(best, 3),
            "iou": round(best_iou, 3),
            "frame": last.frame_id,
            "zoom": round(scale, 2),
            "support": len(obj.observations),
        }

    # ------------------------------------------------------------------- loop
    def locate(
        self, category: str, attributes: list[str] | None = None
    ) -> AgentOutcome:
        rescans = 0
        label_tried = False
        while True:
            candidates = self._lookup(category, attributes)
            self._log(
                "lookup_memory",
                {"category": category},
                {"candidates": len(candidates)},
                "inspect candidates" if candidates else "nothing remembered",
            )
            if not candidates and self.rescan and not label_tried:
                label_tried = True
                self.memory = self.rescan(0.0, [category])
                self._log(
                    "rescan",
                    {"labels": [category]},
                    {"objects_in_memory": len(self.memory.object_memory)},
                    "look up again with the new vocabulary",
                )
                continue
            if not candidates:
                return AgentOutcome(
                    NOT_FOUND, [], self.memory, self.trace, rescans=rescans
                )

            checked: list[tuple[ObjectRecord, dict]] = []
            for obj in candidates[: self.cfg.max_candidates]:
                result = self.inspect(obj, category)
                checked.append((obj, result))
                self._log(
                    "inspect_crop",
                    {"object": obj.object_id, "frame": result.get("frame")},
                    {k: v for k, v in result.items() if k != "frame"},
                    "accept as the answer"
                    if result["verified"]
                    else "reject, try next",
                )
                if result["verified"]:
                    ordered = [obj] + [c for c in candidates if c is not obj]
                    return AgentOutcome(
                        VERIFIED,
                        ordered,
                        self.memory,
                        self.trace,
                        confidence=result["confidence"],
                        rescans=rescans,
                    )

            if self.rescan and rescans < self.cfg.max_rescans:
                rescans += 1
                self.memory = self.rescan(self.cfg.dense_fps, None)
                self._log(
                    "rescan",
                    {"sample_fps": self.cfg.dense_fps},
                    {"objects_in_memory": len(self.memory.object_memory)},
                    "no candidate verified; look again with denser sampling",
                )
                continue

            # Nothing verified: do not guess, hand the choice to the human.
            options = [
                {
                    "object_id": obj.object_id,
                    "last_seen": obj.last_seen,
                    "crop_confidence": res.get("confidence", 0.0),
                    "support": res.get("support", len(obj.observations)),
                }
                for obj, res in checked
            ]
            self._log(
                "ask_human",
                {"options": len(options)},
                {"verified": 0},
                "no candidate verified; request confirmation",
            )
            return AgentOutcome(
                NEEDS_CONFIRMATION,
                candidates,
                self.memory,
                self.trace,
                candidates=options,
                rescans=rescans,
            )
