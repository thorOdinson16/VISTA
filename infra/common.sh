# Shared settings for the VISTA AWS scripts (source me). Run from Git Bash.
export MSYS_NO_PATHCONV=1
export PATH="$PATH:/c/Program Files/Amazon/AWSCLIV2"
export AWS_PAGER=""

REGION="${VISTA_REGION:-$(aws configure get region)}"
REGION="${REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$REGION"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="vista-${ACCOUNT}-${REGION}"
ROLE="vista-ec2-role"
PROFILE="vista-ec2-profile"
SG_NAME="vista-sg"
KEY_NAME="vista-key"
KEY_FILE="$HOME/.ssh/vista-key.pem"
STATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.state"
mkdir -p "$STATE_DIR"

# mode -> instance type / AMI parameter / docker build arg
mode_config() {
  case "$1" in
    gpu)     ITYPE=g5g.xlarge
             AMI_PARAM=/aws/service/deeplearning/ami/arm64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id
             TORCH_INDEX=https://download.pytorch.org/whl/cu130 ;;
    cpu-arm) ITYPE=c7g.xlarge
             AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id
             TORCH_INDEX= ;;
    cpu-x86) ITYPE=c7i.xlarge
             AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
             TORCH_INDEX= ;;
    *) echo "mode must be gpu | cpu-arm | cpu-x86" >&2; exit 1 ;;
  esac
}

instance_id() { cat "$STATE_DIR/$1.id" 2>/dev/null || true; }
instance_ip() {
  aws ec2 describe-instances --instance-ids "$(instance_id "$1")" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}
