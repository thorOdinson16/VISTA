#!/usr/bin/env bash
# Usage: infra/start.sh cool-arm|base-arm|base-x86|gpu -- start a stopped instance (its public IP changes).
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE="${1:?mode}"
aws ec2 start-instances --instance-ids "$(instance_id "$MODE")" >/dev/null
aws ec2 wait instance-running --instance-ids "$(instance_id "$MODE")"
echo "$MODE: http://$(instance_ip "$MODE"):8000  (the container restarts automatically)"
