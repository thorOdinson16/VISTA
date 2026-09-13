# VISTA — Visual Intelligence & Spatial Temporal Archive

VISTA turns video into a persistent, queryable memory of objects, people,
events, and spatial relationships — with every answer grounded in visual
evidence.

> VISTA does not merely store video. It remembers what happened.

- Full design: [`VISTA_Architecture.md`](VISTA_Architecture.md)
- Requirements: [`VISTA_Requirements.md`](VISTA_Requirements.md)

---

## Pipeline

```
video → OpenCV ingestion & adaptive sampling
      → detection (cv2.dnn YOLO, or synthetic ColorDetector)
      → multi-object tracking (Kalman + IoU/appearance association)
      → temporal association / re-identification (ORB, histograms)
      → visual memory (objects, events, scene, relationships, evidence)
      → SQLite persistence
      → NL query → retrieval → evidence-grounded answer
      → UI (video seek, timeline, evidence, overlays)
```

OpenCV is used substantively: `cv2.dnn` inference, ORB feature
extraction/matching, Farneback optical flow, MOG2 background subtraction,
HSV histograms, Kalman filtering, and geometry.

---

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Node 18+ (frontend)

OpenCV 5 (`opencv-contrib-python>=5.0`) is installed automatically.

---

## Setup

```powershell
uv sync                 # core deps
uv sync --group dev     # tests + ruff
uv sync --group detection   # ultralytics, only for ONNX export
```

Copy the environment template and adjust:

```powershell
Copy-Item .env.example .env
```

Key settings (all prefixed `VISTA_`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VISTA_LLM_PROVIDER` | `rule-based` | `rule-based`, `openai`, `anthropic`, `ollama` |
| `VISTA_DETECTOR_MODE` | `synthetic` | `synthetic` demo detector or `real` YOLO |
| `VISTA_YOLO_MODEL_PATH` | *(empty)* | explicit ONNX path; else first `data/models/*.onnx` |
| `VISTA_SEMANTIC_RETRIEVAL_THRESHOLD` | `0.85` | min cosine similarity for semantic match |
| `VISTA_SAMPLING_STRATEGY` | `adaptive` | `adaptive`, `fixed`, `all` |

The app works fully offline with `rule-based` + `synthetic` (the default):
query parsing falls back to deterministic rules and answering to grounded
templates. Configuring an LLM provider enables free-text parsing, richer
answers, and semantic object retrieval.

---

## Real detector (YOLO via cv2.dnn)

1. Export a YOLO model to ONNX:

   ```powershell
   uv sync --group detection
   uv run python scripts/export_onnx.py --model yolov8s --out data/models/yolov8s.onnx
   ```

2. Set `VISTA_DETECTOR_MODE=real` in `.env` (and optionally
   `VISTA_YOLO_MODEL_PATH`).

With `real` selected and no model present, VISTA warns and falls back to
the synthetic detector.

---

## Run

Backend (FastAPI on `127.0.0.1:8000`):

```powershell
uv run python backend/run.py
```

Frontend (Vite on `5173`, proxying `/api`, `/video`, `/evidence`):

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173, upload a video, ask questions, and click
timeline/evidence items to seek the source video.

---

## Demo video

Generate the controlled scene from requirements §15 with ground truth:

```powershell
uv run python scripts/demo_video.py --out data/demo.mp4 --gt data/demo_gt.json
```

Then ingest `data/demo.mp4` through the UI or `POST /api/ingest`.

---

## Evaluation

```powershell
uv run python scripts/evaluate.py
```

Runs the pipeline against the demo ground truth and reports event
precision/recall/F1, tracking integrity, query-answer accuracy, and
throughput. Add `--strict` to exit non-zero when thresholds are missed.
`backend/tests/test_evaluation.py` enforces minimum event F1 and query
accuracy.

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/status` | Ingest status (objects, events, fps, duration) |
| `POST` | `/api/ingest` | Upload and index a video (multipart `file`) |
| `POST` | `/api/query` | Ask a natural-language question |
| `GET` | `/api/timeline` | Chronological events |
| `GET` | `/api/objects` | Object records |
| `GET` | `/api/scene?t=` | Scene snapshot (spatial relationships) |
| `GET` | `/api/similar?object_id=&k=` | Visually similar objects (OpenCV descriptors) |
| `GET` | `/video` | Source video stream |
| `GET` | `/evidence/<file>` | Evidence frames |

Example query:

```json
{ "question": "Who moved the red notebook?" }
```

The response separates **observed** facts from **inferred** conclusions,
includes a confidence value and a timeline, and references only stored
evidence (unsupported claims become explicit uncertainty).

---

## Tests & lint

```powershell
uv run ruff check .
uv run pytest
```

---

## Layout

```
backend/app/
  ingestion/   frame reading + adaptive (motion-aware) sampling
  perception/  detection, appearance descriptors, ORB, optical flow
  tracking/    association, Kalman tracking, re-identification
  memory/      object/event/scene/graph/evidence stores + SQLite persistence
  retrieval/   query parsing, temporal ops, vector index, evidence lookup
  reasoning/   answer synthesis + evidence grounding
  llm/         pluggable providers (rule-based/OpenAI/Anthropic/Ollama)
  pipeline/    end-to-end orchestration
  api/         FastAPI app
frontend/      React + Vite UI
scripts/       demo video generation, ONNX export, evaluation
```
