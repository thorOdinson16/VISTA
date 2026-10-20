#!/usr/bin/env bash
# Usage: infra/launch.sh gpu|cpu-arm|cpu-x86   -- launch an instance (once per mode).
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODE="${1:?mode}"; mode_config "$MODE"
if [ -n "$(instance_id "$MODE")" ]; then
  echo "instance exists ($(instance_id "$MODE")); use infra/start.sh $MODE"; exit 0
fi
AMI="$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)"
SG="$(cat "$STATE_DIR/sg.id")"
ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG" \
  --iam-instance-profile "Name=$PROFILE" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=vista-$MODE},{Key=project,Value=vista}]" \
  --query 'Instances[0].InstanceId' --output text)"
echo "$ID" > "$STATE_DIR/$MODE.id"
aws ec2 wait instance-running --instance-ids "$ID"
echo "$MODE: $ITYPE $ID ip=$(instance_ip "$MODE")"
