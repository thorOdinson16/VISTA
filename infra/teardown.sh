#!/usr/bin/env bash
# Terminate instances. With --all also delete the bucket, IAM role and key pair.
set -euo pipefail
source "$(dirname "$0")/common.sh"
for m in gpu cpu-arm cpu-x86; do
  id="$(instance_id "$m")"; [ -z "$id" ] && continue
  aws ec2 terminate-instances --instance-ids "$id" >/dev/null && rm -f "$STATE_DIR/$m.id"
  echo "terminated $m ($id)"
done
if [ "${1:-}" = "--all" ]; then
  aws s3 rm "s3://$BUCKET" --recursive || true
  aws s3api delete-bucket --bucket "$BUCKET" || true
  aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE" || true
  aws iam delete-instance-profile --instance-profile-name "$PROFILE" || true
  aws iam delete-role-policy --role-name "$ROLE" --policy-name vista-s3 || true
  aws iam delete-role --role-name "$ROLE" || true
  aws ec2 delete-key-pair --key-name "$KEY_NAME" || true
fi
