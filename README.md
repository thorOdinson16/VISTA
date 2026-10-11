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
| `VISTA_DETECTOR_MODE` | `real` | `real` (COCO YOLO ONNX), `owlv2` (open-vocabulary, recommended), `openvocab` (YOLO-World), `hybrid`, or `synthetic` (demo/tests) |
| `VISTA_OWLV2_MODEL` | `google/owlv2-base-patch16-ensemble` | OWLv2 weights (downloaded on first use, ~600 MB) |
| `VISTA_YOLO_MODEL_PATH` | *(empty)* | explicit ONNX path; else first `data/models/*.onnx` |
| `VISTA_YOLO_CLASS_NAMES_PATH` | *(empty)* | custom class names (JSON list or one per line) |
| `VISTA_DETECTOR_LABEL_MAP` | *(empty)* | JSON raw→canonical label map override |
| `VISTA_DETECTOR_LABELS` | canonical list | open-vocab text labels (comma-separated) |
| `VISTA_OPENVOCAB_MODEL` | `yolov8s-worldv2.pt` | YOLO-World weights |
| `VISTA_SEMANTIC_RETRIEVAL_THRESHOLD` | `0.85` | min cosine similarity for semantic match |
| `VISTA_SAMPLING_STRATEGY` | `adaptive` | `adaptive`, `fixed`, `all` |
| `VISTA_SCENE_CHANGE_MOTION_GATE` | `2.0` | flow magnitude above which camera motion suppresses scene-change |
| `VISTA_SALIENCY_FALLBACK` | `false` | opt-in: propose persistent unclassified regions when the detector finds nothing |
| `VISTA_SALIENCY_MIN_AREA` | `600` | min region area (source px) for a saliency proposal |

### Detector modes

`VISTA_DETECTOR_MODE=real` (the default) uses a YOLO ONNX model via
`cv2.dnn`. If no model is present, ingestion **fails with an explicit error**
rather than silently substituting the synthetic demo detector — the active
detector is always reported in the UI and `/api/status`. `synthetic` is only
for the generated demo and tests.

### Unknown objects (opt-in)

With `VISTA_SALIENCY_FALLBACK=true`, if the configured detector matches
nothing on a frame VISTA may propose *persistent, non-border* MOG2 regions as
`object` (unclassified) — only when the camera is not moving and the region
survives two sampled frames. This is off by default because MOG2 assumes a
static camera and can produce false objects on handheld footage. When off,
asking **"what is shown in the video?"** returns an honest "no configured
objects detected" plus a representative frame. Unrecognized content is never
given a fabricated label.

The app works fully offline with `rule-based` + `synthetic` (the default):
query parsing falls back to deterministic rules and answering to grounded
templates. Configuring an LLM provider enables free-text parsing, richer
answers, and semantic object retrieval.

---

## Real detector (YOLO via cv2.dnn)

`VISTA_DETECTOR_MODE=real` is the default. Fetch a model once:

```powershell
uv sync --group detection
uv run python scripts/fetch_model.py --model yolov8s   # or yolov8n
```

This writes `data/models/yolov8s.onnx`, which VISTA auto-discovers. Verify
the detector actually runs:

```powershell
uv run python scripts/verify_detector.py --source path/to/image_or_video.mp4
```

`real` mode with no model is an **explicit error**, never a silent fallback
to the synthetic detector. The synthetic detector is only used with
`VISTA_DETECTOR_MODE=synthetic` (the demo video and the test suite).

> The decoder supports YOLOv8/v11 detection exports (`4 + C` columns, no
> objectness) and performs class-aware NMS. Custom models need a
> class-names file via `VISTA_YOLO_CLASS_NAMES_PATH`.

### Detector modes

| Mode | Backend | Use |
| --- | --- | --- |
| `real` (default) | YOLO ONNX via `cv2.dnn` | everyday COCO objects |
| `owlv2` | OWLv2 (`transformers`) | **recommended for real footage**: arbitrary text labels, works on small / thin / low-contrast objects (chargers, cables, coins) |
| `openvocab` | Ultralytics YOLO-World | arbitrary text labels, faster but misses small or thin objects |
| `hybrid` | `real`, fall back to `openvocab` | best coverage; slower |
| `synthetic` | ColorDetector | demo video + tests only |

Open-vocabulary labels come from `VISTA_DETECTOR_LABELS` (comma-separated);
YOLO-World weights are fetched on first use into the Ultralytics cache.

### Open-vocabulary detection with OWLv2 (recommended)

COCO detectors only know 80 classes and will force anything else into the
nearest one (a coin becomes a `clock`, a charger becomes a `phone`).
`owlv2` detects whatever you name.

```powershell
uv sync --group dev --group detection --group owlv2
```

Create `.env` (it is gitignored, so every machine needs its own):

```
VISTA_DETECTOR_MODE=owlv2
VISTA_DETECTOR_LABELS=coin,key,wallet,phone,phone charger,charging cable,mug,bottle,laptop,person
VISTA_DETECTION_CONFIDENCE_THRESHOLD=0.2
```

Notes:

- **Speed.** OWLv2 runs at ~2 s/frame on CPU and is several times faster on
  a CUDA GPU. The detector uses CUDA automatically when available; the
  active device is shown in the UI (`OWLv2 (…, cuda)`).
- **First run** downloads the model (~600 MB) from Hugging Face.
- **Why not YOLO-World?** On thin or low-contrast objects (a white charger
  cable on a patterned sheet) YOLO-World scored ≤ 0.07 even with the exact
  label; OWLv2 scored 0.6+.
- **Confidence.** OWLv2 scores run lower than COCO YOLO; `0.2` is a good
  starting point. Lower it to find faint objects, raise it to cut noise.

#### Local GPU (CUDA) install

The lockfile pins CPU PyTorch so the same install works on ARM/Graviton.
To use a local NVIDIA GPU, install a CUDA build into the venv only:

```powershell
uv pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```

Match the CUDA tag to your driver (`cu128` / `cu130`). Then run with the
venv's interpreter, because `uv run` re-syncs the environment and will put
the CPU build back:

```powershell
.venv\Scripts\python backend/run.py
.venv\Scripts\python -m pytest
```

### Ask for anything (on-demand search)

Queries are not limited to the configured label list. Ask
"where is my charger?" and, if `charger` was never in the detector's
vocabulary, VISTA adds it, re-scans the video once, and answers. The label
is remembered for the session, so repeat questions are instant. The UI notes
when a re-scan happened. Compound nouns are kept intact ("phone charger" is
not reduced to "phone"). This applies to the `owlv2`, `openvocab` and
`hybrid` modes.

### Canonical vocabulary & label mapping

Detector labels are mapped to one canonical vocabulary
(`backend/app/perception/vocabulary.py`) so query terms, stored categories
and detector output agree: e.g. COCO `book` → `notebook`,
`dining table` → `desk`, `cup` → `mug`. Queries containing synonyms
("book", "table") resolve to the canonical objects. Override the raw→canonical
map with `VISTA_DETECTOR_LABEL_MAP` (JSON). Unmapped labels pass through
unchanged — an unknown detection is never silently renamed.

### Detection benchmark

```powershell
uv run python scripts/benchmark_detection.py --gt data/bench/gt.json
```

Ground-truth JSON:

```json
{ "images": [
  { "path": "data/bench/0001.jpg",
    "boxes": [ { "label": "person", "bbox": [x1, y1, x2, y2] } ] }
] }
```

Reports precision/recall/mAP@0.5 per class.

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
