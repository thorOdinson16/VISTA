"""Objectron (Google) annotation reader.

Objectron ships ``video.MOV`` and ``geometry.pbdata`` per clip. The
``geometry.pbdata`` is a serialized ``Sequence`` protobuf; this module
decodes just the fields needed (frame id, timestamp, per-object 2D
keypoints) straight from the wire format, so no ``protobuf`` runtime or
generated code is required.

Schema (annotation_data.proto)::

    Sequence         { repeated FrameAnnotation frame_annotations = 2; }
    FrameAnnotation  { int32 frame_id = 1; repeated ObjectAnnotation annotations = 2;
                       double timestamp = 4; }
    ObjectAnnotation { int32 object_id = 1; repeated AnnotatedKeyPoint keypoints = 2;
                       float visibility = 3; }
    AnnotatedKeyPoint{ int32 id = 1; Point3D point_3d = 2; NormalizedPoint2D point_2d = 3; }
    NormalizedPoint2D{ float x = 1; float y = 2; float depth = 3; }

Keypoint 0 is the box centre; 1..8 are the box corners. Objectron data is
research-licensed (CC BY 4.0 / C-UDA): it is downloaded at run time and
never committed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/objectron/videos"
INDEX_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/Objectron/"
    "master/index/{cls}_annotations_test"
)


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    """Yield ``(field_number, wire_type, value)`` for one message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:  # groups are not used by this schema
            raise ValueError(f"unsupported wire type {wire}")
        yield number, wire, value


@dataclass
class ObjectFrame:
    object_id: int
    visibility: float
    keypoints: list[tuple[float, float]] = field(default_factory=list)  # normalized

    def box_norm(self) -> tuple[float, float, float, float]:
        """Axis-aligned 2D box (normalized) around the 8 projected corners."""
        corners = self.keypoints[1:9] or self.keypoints
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class FrameAnn:
    frame_id: int
    timestamp: float
    objects: list[ObjectFrame] = field(default_factory=list)


def parse_geometry(data: bytes) -> list[FrameAnn]:
    frames: list[FrameAnn] = []
    for number, _wire, value in _fields(data):
        if number != 2:
            continue
        frame = FrameAnn(frame_id=0, timestamp=0.0)
        for fnum, fwire, fval in _fields(value):
            if fnum == 1 and fwire == 0:
                frame.frame_id = int(fval)
            elif fnum == 4 and fwire == 1:
                frame.timestamp = struct.unpack("<d", fval)[0]
            elif fnum == 2 and fwire == 2:
                obj = ObjectFrame(object_id=0, visibility=0.0)
                for onum, owire, oval in _fields(fval):
                    if onum == 1 and owire == 0:
                        obj.object_id = int(oval)
                    elif onum == 3 and owire == 5:
                        obj.visibility = struct.unpack("<f", oval)[0]
                    elif onum == 2 and owire == 2:
                        x = y = 0.0
                        for knum, kwire, kval in _fields(oval):
                            if knum == 3 and kwire == 2:
                                for pnum, pwire, pval in _fields(kval):
                                    if pnum == 1 and pwire == 5:
                                        x = struct.unpack("<f", pval)[0]
                                    elif pnum == 2 and pwire == 5:
                                        y = struct.unpack("<f", pval)[0]
                        obj.keypoints.append((x, y))
                frame.objects.append(obj)
        frames.append(frame)
    return frames


def load_geometry(path: str | Path) -> list[FrameAnn]:
    return parse_geometry(Path(path).read_bytes())
