#!/usr/bin/env bash
# Usage: infra/bench_box.sh base-arm|base-x86
#
# Baseline for the COOL comparison: a plain Ubuntu instance with *stock* pip
# OpenCV 5 and the same pinned dependencies, running the identical benchmark
# (scripts/bench_cv.py) and Objectron clips as the COOL instance. Results are
# copied to docs/eval/ and the instance is left running for you to stop or
# terminate (infra/stop.sh / infra/teardown.sh).
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE="${1:?base-arm|base-x86}"; mode_config "$MODE"
IP="$(instance_ip "$MODE")"
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/eval"
mkdir -p "$OUT"

until "${SSH[@]}" true 2>/dev/null; do echo "waiting for ssh..."; sleep 6; done

# Full pinned stack, including pip opencv/numpy (this is the stock install).
(cd "$ROOT" && uv export --frozen --no-dev --no-emit-project --no-hashes --group aws --group eval 2>/dev/null) \
  > "$STATE_DIR/requirements_full.txt"
(cd "$ROOT" && git archive --format=tar.gz -o infra/.state/src.tgz HEAD)
BENCH_CLIP="$(ls "$STATE_DIR"/bench_clip.MOV 2>/dev/null || true)"
[ -n "$BENCH_CLIP" ] || { echo "put the benchmark clip at infra/.state/bench_clip.MOV" >&2; exit 1; }
scp -i "$KEY_FILE" "$STATE_DIR/src.tgz" "$STATE_DIR/requirements_full.txt" "$BENCH_CLIP" \
  "$ROOT/data/models/yolov8n.onnx" "ubuntu@$IP:/tmp/"

"${SSH[@]}" bash -s "$MODE" <<'REMOTE'
set -euo pipefail
MODE="$1"
sudo apt-get update -qq >/dev/null
sudo apt-get install -y -qq python3.12-venv libgl1 libglib2.0-0 >/dev/null 2>&1
sudo mkdir -p /opt/vista /opt/vista-data/models /opt/vista-data/eval && sudo chown -R ubuntu /opt/vista /opt/vista-data
tar xzf /tmp/src.tgz -C /opt/vista
mv -f /tmp/yolov8n.onnx /opt/vista-data/models/
cd /opt/vista
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/pip install -q --disable-pip-version-check --no-deps -r /tmp/requirements_full.txt
export VISTA_DATA_DIR=/opt/vista-data VISTA_DETECTOR_MODE=real
.venv/bin/python - <<'PY'
import cv2; print("OpenCV", cv2.__version__, cv2.__file__)
PY
.venv/bin/python scripts/bench_cv.py --video /tmp/bench_clip.MOV --model /opt/vista-data/models/yolov8n.onnx \
  --repeats 5 --pipeline --label "$MODE" --out /opt/vista-data/eval/bench_$MODE.json
.venv/bin/python scripts/eval_objectron.py --classes cup bottle book laptop chair --per-class 4 \
  --cache /opt/vista-data/objectron --out /opt/vista-data/eval/objectron_${MODE}_2fps.json | tail -12
REMOTE

scp -q -i "$KEY_FILE" "ubuntu@$IP:/opt/vista-data/eval/bench_$MODE.json" \
  "ubuntu@$IP:/opt/vista-data/eval/objectron_${MODE}_2fps.json" "$OUT/"
echo "results copied to docs/eval/ ($MODE)"
