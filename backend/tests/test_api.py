from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api.main import SESSION, app
from backend.app.config import settings
from backend.app.pipeline.pipeline import build_default_detector


def _make_video_bytes(frames: int = 60, fps: float = 30.0) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
        )
        for i in range(frames):
            img = np.full((240, 320, 3), 255, np.uint8)
            x = int(20 + i * 3)
            img[100:160, x : x + 40] = (0, 0, 220)  # notebook
            img[100:160, 200:250] = (220, 0, 0)     # mug
            writer.write(img)
        writer.release()
        return path.read_bytes()


def test_status_and_ingest_and_query(monkeypatch):
    # Tests run without a model/network: pin the deterministic synthetic
    # detector. Production defaults to the real YOLO detector.
    monkeypatch.setattr(settings, "detector_mode", "synthetic")
    client = TestClient(app)
    SESSION.reset()

    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["objects"] == 0

    video_bytes = _make_video_bytes()
    resp = client.post(
        "/api/ingest",
        files={"file": ("demo.mp4", video_bytes, "video/mp4")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["objects"] >= 1
    assert data["events"] >= 1

    resp = client.get("/api/status")
    assert resp.json()["objects"] >= 1

    resp = client.get("/api/timeline")
    assert resp.status_code == 200
    assert len(resp.json()["events"]) >= 1

    resp = client.post("/api/query", json={"question": "Where is the notebook?"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "answer" in payload
    assert payload["answer"]["text"]

    resp = client.get("/video")
    assert resp.status_code == 200


def test_real_mode_without_model_raises():
    with pytest.raises(ValueError, match="no YOLO ONNX model"):
        build_default_detector(mode="real", models_dir="data/models_missing")


def test_unknown_detector_mode_raises():
    with pytest.raises(ValueError, match="Unknown detector mode"):
        build_default_detector(mode="banana")
