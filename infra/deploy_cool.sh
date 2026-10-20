#!/usr/bin/env bash
# Usage: infra/deploy_cool.sh [mode]   (default: cool-arm)
#
# Deploy VISTA on a COOL (Cloud-Optimized OpenCV, OpenCV 5.x) instance.
# No Docker: COOL ships OpenCV + numpy under /opt/cool, so we create a Python
# 3.12 venv with the pinned deps from uv.lock *minus* pip's opencv/numpy and put
# COOL's site-packages on PYTHONPATH. The app runs as a systemd service.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE="${1:-cool-arm}"; mode_config "$MODE"
IP="$(instance_ip "$MODE")"
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

until "${SSH[@]}" true 2>/dev/null; do echo "waiting for ssh..."; sleep 6; done

# Pinned runtime requirements, without anything that would shadow COOL's cv2/numpy.
(cd "$ROOT" && uv export --frozen --no-dev --no-emit-project --no-hashes --group aws 2>/dev/null) \
  | grep -viE '^(opencv[-a-z]*|numpy)==' > "$STATE_DIR/requirements.txt"
echo "$(wc -l < "$STATE_DIR/requirements.txt") pinned requirements (opencv/numpy excluded)"

(cd "$ROOT" && git archive --format=tar.gz -o infra/.state/src.tgz HEAD)
(cd "$ROOT/frontend" && tar czf "$STATE_DIR/ui.tgz" dist)
scp -i "$KEY_FILE" "$STATE_DIR/src.tgz" "$STATE_DIR/ui.tgz" "$STATE_DIR/requirements.txt" \
  "$ROOT/data/models/yolov8n.onnx" "ubuntu@$IP:/tmp/"

"${SSH[@]}" bash -s "$BUCKET" "$REGION" <<'REMOTE'
set -euo pipefail
BUCKET="$1"; REGION="$2"
sudo apt-get update -qq >/dev/null
sudo apt-get install -y -qq python3.12-venv >/dev/null

sudo systemctl stop vista 2>/dev/null || true
sudo mkdir -p /opt/vista /opt/vista-data/models && sudo chown -R ubuntu /opt/vista /opt/vista-data
rm -rf /opt/vista/backend /opt/vista/scripts /opt/vista/frontend
tar xzf /tmp/src.tgz -C /opt/vista
mkdir -p /opt/vista/frontend && tar xzf /tmp/ui.tgz -C /opt/vista/frontend
mv -f /tmp/yolov8n.onnx /opt/vista-data/models/

cd /opt/vista
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/pip install -q --disable-pip-version-check -r /tmp/requirements.txt

sudo tee /etc/systemd/system/vista.service >/dev/null <<UNIT
[Unit]
Description=VISTA
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/vista
Environment=PYTHONPATH=/opt/cool/python_3.12/site-packages:/opt/vista
Environment=LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib
Environment=VISTA_DATA_DIR=/opt/vista-data
Environment=VISTA_HOST=0.0.0.0
Environment=VISTA_FRONTEND_DIR=/opt/vista/frontend/dist
Environment=VISTA_DETECTOR_MODE=real
Environment=VISTA_STORAGE=s3
Environment=VISTA_S3_BUCKET=$BUCKET
Environment=VISTA_AWS_REGION=$REGION
Environment=AWS_DEFAULT_REGION=$REGION
ExecStart=/opt/vista/.venv/bin/python backend/run.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now vista
sleep 4
systemctl is-active vista
PYTHONPATH=/opt/cool/python_3.12/site-packages LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib \
  .venv/bin/python -c "import cv2; print('OpenCV', cv2.__version__, cv2.__file__)"
REMOTE
echo "deployed: http://$IP:8000"
