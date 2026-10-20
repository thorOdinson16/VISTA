#!/usr/bin/env bash
# The one remaining AWS session: bring the COOL box up on the final commit,
# re-run the evaluations that changed (MOT17 with the new tracker, the verify
# agent on held-out clips) on the COOL instance, then leave the app running for
# the screen recording. Afterwards run:  bash infra/teardown.sh --all
#
# Usage: bash infra/final_session.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=cool-arm

if [ -z "$(instance_id "$MODE")" ]; then
  echo "no $MODE instance on record: launching a fresh one (MOT17 data must be re-downloaded)"
  bash "$ROOT/infra/launch.sh" "$MODE"
else
  bash "$ROOT/infra/start.sh" "$MODE"
fi
bash "$ROOT/infra/deploy.sh" "$MODE"

IP="$(instance_ip "$MODE")"
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new "ubuntu@$IP")
"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
export PYTHONPATH=/opt/cool/python_3.12/site-packages:/opt/vista
export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib
export VISTA_DATA_DIR=/opt/vista-data VISTA_DETECTOR_MODE=real
cd /opt/vista
mkdir -p /opt/vista-data/eval
if [ -d /opt/vista-data/mot/MOT17/train ]; then
  .venv/bin/python scripts/eval_mot17.py --root /opt/vista-data/mot/MOT17/train \
    --seqs MOT17-09-FRCNN MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN \
    --sample-fps 2 --out /opt/vista-data/eval/mot17_cool_v2_2fps.json
else
  echo "MOT17 data not on the box; skipping MOT17"
fi
.venv/bin/python scripts/eval_objectron.py --classes cup bottle book laptop chair --per-class 4 \
  --cache /opt/vista-data/objectron --out /opt/vista-data/eval/objectron_cool_v2_2fps.json | tail -3
.venv/bin/python scripts/eval_agent.py --classes cup bottle book laptop chair --per-class 4 --seed 1 \
  --cache /opt/vista-data/objectron --out /opt/vista-data/eval/agent_cool_heldout.json | tail -14
REMOTE

scp -q -i "$KEY_FILE" "ubuntu@$IP:/opt/vista-data/eval/*_v2_*.json" "ubuntu@$IP:/opt/vista-data/eval/agent_cool_heldout.json" \
  "$ROOT/docs/eval/" 2>/dev/null || true
echo
echo "App is up: http://$IP:8000  (record the demo now)"
echo "Then run: bash infra/teardown.sh --all"
