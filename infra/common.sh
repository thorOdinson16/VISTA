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

# mode -> instance type / AMI / root volume (GB) / docker build arg
#   cool-arm : Graviton5 (m9g) + COOL (Cloud-Optimized OpenCV) Marketplace AMI
#   base-arm : same hardware, plain Ubuntu + pip OpenCV  (COOL baseline)
#   base-x86 : x86 (c8i) + pip OpenCV                    (architecture baseline)
#   gpu      : optional Graviton2 + T4G GPU for OWLv2
COOL_AMI="ami-00f40acc34f3d7d39"   # Cloud Optimized OpenCV For AWS Graviton5 (m9g)
mode_config() {
  AMI=""; VOLUME=40; TORCH_INDEX=
  case "$1" in
    cool-arm) ITYPE=m9g.xlarge; AMI="$COOL_AMI"; VOLUME=80 ;;
    base-arm) ITYPE=m9g.xlarge
              AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id ;;
    base-x86) ITYPE=m8i.xlarge
              AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id ;;
    gpu)      ITYPE=g5g.xlarge; VOLUME=100
              AMI_PARAM=/aws/service/deeplearning/ami/arm64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id
              TORCH_INDEX=https://download.pytorch.org/whl/cu130 ;;
    *) echo "mode must be cool-arm | base-arm | base-x86 | gpu" >&2; exit 1 ;;
  esac
}

instance_id() { cat "$STATE_DIR/$1.id" 2>/dev/null || true; }
instance_ip() {
  aws ec2 describe-instances --instance-ids "$(instance_id "$1")" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}
