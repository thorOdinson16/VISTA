"""Run the hand-checked clip questions through the real API path.

    python scripts/eval_clips.py --spec docs/eval/clips_spec.json \\
        --dir . --out docs/eval/clips_real.json

Uses FastAPI's TestClient against ``backend.app.api.main`` so ingest, the
query parser, on-demand open-vocabulary re-scan (when the detector mode
allows it) and the grounded answerer are exercised exactly as the UI does.
The detector mode comes from the VISTA_* environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = tempfile.mkdtemp(prefix="vista-eval-")
os.environ.setdefault("VISTA_DATA_DIR", _tmp)

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api.main import app  # noqa: E402
from backend.app.config import settings  # noqa: E402


def last_seen(response: dict) -> float | None:
    times = [e.get("timestamp_start") for e in response.get("evidence", []) if e.get("timestamp_start") is not None]
    return max(times) if times else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docs/eval/clips_spec.json")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="docs/eval/clips_results.json")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    client = TestClient(app)
    rows = []
    for clip, info in spec["clips"].items():
        path = Path(args.dir) / clip
        start = time.perf_counter()
        with path.open("rb") as handle:
            ingest = client.post("/api/ingest", files={"file": (clip, handle, "video/mp4")})
        ingest.raise_for_status()
        ingest_s = time.perf_counter() - start
        for check in info["checks"]:
            q_start = time.perf_counter()
            resp = client.post("/api/query", json={"question": check["question"]}).json()
            latency = time.perf_counter() - q_start
            seen = last_seen(resp)
            found = bool(resp.get("evidence"))
            if check["expect"] == "found":
                lo, hi = check["last_seen_between"]
                passed = found and seen is not None and lo <= seen <= hi
            else:
                passed = not found
            rows.append(
                {
                    "clip": clip,
                    "question": check["question"],
                    "expect": check["expect"],
                    "found": found,
                    "last_seen_s": round(seen, 2) if seen is not None else None,
                    "window": check.get("last_seen_between"),
                    "passed": passed,
                    "searched_on_demand": resp.get("searched_on_demand", False),
                    "answer": resp.get("answer", {}).get("text"),
                    "agent_status": resp.get("agent_status"),
                    "agent_trace": resp.get("agent_trace", []),
                    "latency_s": round(latency, 3),
                }
            )
            print(f"[{'PASS' if passed else 'FAIL'}] {clip}: {check['question']} -> found={found} seen={seen}")
        print(f"   ingest {ingest_s:.1f}s", flush=True)

    summary = {"passed": sum(r["passed"] for r in rows), "total": len(rows)}
    report = {
        "detector_mode": settings.detector_mode,
        "detector_labels": settings.detector_label_list,
        "confidence_threshold": settings.detection_confidence_threshold,
        "summary": summary,
        "results": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"{summary['passed']}/{summary['total']} passed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
