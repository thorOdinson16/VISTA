"""API flow for the verify agent: defer to the human, confirm, reject."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api import main
from backend.app.memory.models import BoundingBox, Detection

W, H = 320, 240


class FixedDetector:
    """Always reports a mug at the same pixel box.

    On a zoomed crop that box does not coincide with the claimed one, so the
    agent cannot verify the sighting and must hand the decision to the human.
    """

    name = "fixed"

    def detect(self, image, frame_id, timestamp):
        return [
            Detection(
                category="mug",
                bounding_box=BoundingBox(x_min=100, y_min=80, x_max=220, y_max=160),
                confidence=0.9,
                frame_id=frame_id,
                timestamp=timestamp,
            )
        ]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main, "build_default_detector", lambda **kwargs: FixedDetector()
    )
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    for _ in range(30):
        frame = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(frame, (100, 80), (220, 160), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    main.SESSION.reset()
    test_client = TestClient(main.app)
    with path.open("rb") as handle:
        resp = test_client.post(
            "/api/ingest", files={"file": ("clip.mp4", handle, "video/mp4")}
        )
    assert resp.status_code == 200
    yield test_client
    main.SESSION.reset()


def test_unverifiable_sighting_is_deferred_then_confirmed(client):
    r = client.post("/api/query", json={"question": "Where is the mug?"}).json()
    assert r["needs_confirmation"] is True
    assert r["agent_status"] == "needs_confirmation"
    assert [s["tool"] for s in r["agent_trace"]][-1] == "ask_human"
    assert r["candidates"]

    oid = r["candidates"][0]["object_id"]
    done = client.post("/api/confirm", json={"object_id": oid, "accepted": True}).json()
    assert done["agent_status"] == "confirmed_by_human"
    assert "confirmed by you" in done["answer"]["text"]
    assert done["agent_trace"][0]["tool"] == "human_confirmation"


def test_rejecting_the_only_candidate_reports_not_found(client):
    r = client.post("/api/query", json={"question": "Where is the mug?"}).json()
    oid = r["candidates"][0]["object_id"]
    out = client.post("/api/confirm", json={"object_id": oid, "accepted": False}).json()
    assert out["agent_status"] == "not_found"
    assert "cannot determine" in out["answer"]["text"].lower()


def test_confirm_rejects_unknown_ids(client):
    assert client.post("/api/confirm", json={"object_id": "obj_999"}).status_code == 404


def test_agent_can_be_disabled_per_query(client):
    r = client.post(
        "/api/query", json={"question": "Where is the mug?", "agent": False}
    ).json()
    assert r["agent_status"] is None and r["agent_trace"] == []
    assert r["needs_confirmation"] is False
