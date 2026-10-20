#!/usr/bin/env bash
# One-time (idempotent) AWS setup: S3 bucket, instance IAM role, security group, key pair.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MYIP="$(curl -s https://checkip.amazonaws.com)"
echo "account=$ACCOUNT region=$REGION bucket=$BUCKET myip=$MYIP"

# --- S3 bucket (private) ---
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = us-east-1 ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# --- IAM role so EC2 reaches the bucket with no keys on the box ---
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name vista-s3 --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
   {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"},
   {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"}]}"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
  sleep 10
fi

# --- Security group: SSH + API only from this machine's IP ---
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG="$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
      --query 'SecurityGroups[0].GroupId' --output text)"
if [ "$SG" = None ]; then
  SG="$(aws ec2 create-security-group --group-name "$SG_NAME" --description VISTA --vpc-id "$VPC" \
        --query GroupId --output text)"
fi
echo "$SG" > "$STATE_DIR/sg.id"
for port in 22 8000; do
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port "$port" \
    --cidr "$MYIP/32" >/dev/null 2>&1 || true
done

# --- Key pair ---
if [ ! -f "$KEY_FILE" ]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  aws ec2 delete-key-pair --key-name "$KEY_NAME" 2>/dev/null || true
  aws ec2 create-key-pair --key-name "$KEY_NAME" --query KeyMaterial --output text > "$KEY_FILE"
  chmod 400 "$KEY_FILE"
fi
echo "provisioned: bucket=$BUCKET sg=$SG key=$KEY_FILE"
