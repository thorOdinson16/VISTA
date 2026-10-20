#!/usr/bin/env bash
# Usage: infra/stop.sh [cool-arm|base-arm|base-x86|gpu|all] -- stop instances (no compute billing) so credits last.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODES="${1:-gpu}"; [ "$MODES" = all ] && MODES="cool-arm base-arm base-x86 gpu"
for m in $MODES; do
  id="$(instance_id "$m")"; [ -z "$id" ] && continue
  echo "$m: $(aws ec2 stop-instances --instance-ids "$id" --query 'StoppingInstances[0].CurrentState.Name' --output text)"
done
