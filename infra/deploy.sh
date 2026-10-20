#!/usr/bin/env bash
# Usage: infra/deploy.sh cool-arm|base-arm|base-x86|gpu -- ship the current commit, build the image, run it.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE="${1:?mode}"; mode_config "$MODE"
IP="$(instance_ip "$MODE")"
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")

until "${SSH[@]}" true 2>/dev/null; do echo "waiting for ssh..."; sleep 6; done

git archive --format=tar.gz -o "$STATE_DIR/src.tgz" HEAD
scp -i "$KEY_FILE" "$STATE_DIR/src.tgz" "ubuntu@$IP:/tmp/vista-src.tgz"

"${SSH[@]}" bash -s "$MODE" "$TORCH_INDEX" "$BUCKET" "$REGION" <<'REMOTE'
set -euo pipefail
MODE="$1"; TORCH_INDEX="$2"; BUCKET="$3"; REGION="$4"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo rm -rf /opt/vista && sudo mkdir -p /opt/vista /opt/vista-data && sudo chown ubuntu /opt/vista /opt/vista-data
tar xzf /tmp/vista-src.tgz -C /opt/vista
cd /opt/vista
sudo docker build -t vista --build-arg TORCH_INDEX="$TORCH_INDEX" .
sudo docker rm -f vista 2>/dev/null || true
GPU=""; [ "$MODE" = gpu ] && GPU="--gpus all"
sudo docker run -d --name vista --restart unless-stopped $GPU \
  -p 8000:8000 -v /opt/vista-data:/data \
  -e VISTA_STORAGE=s3 -e VISTA_S3_BUCKET="$BUCKET" \
  -e AWS_DEFAULT_REGION="$REGION" -e VISTA_AWS_REGION="$REGION" \
  vista
REMOTE
echo "deployed: http://$IP:8000"
