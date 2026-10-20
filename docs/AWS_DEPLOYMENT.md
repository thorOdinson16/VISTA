# VISTA on AWS

How VISTA is deployed on AWS for the OpenCV AI Competition 2026, why each
service is used, and how to reproduce the deployment from scratch. Measured
results live in [`BENCHMARK.md`](BENCHMARK.md) and [`EVALUATION.md`](EVALUATION.md).

## Architecture

```
                  you (browser, one allow-listed IP)
                              │  HTTP :8000
                              ▼
   ┌───────────────── EC2 m9g.xlarge  (AWS Graviton5, Arm64) ─────────────────┐
   │  AMI: "Cloud Optimized OpenCV For AWS Graviton5" (COOL, AWS Marketplace)  │
   │                                                                           │
   │  systemd service "vista"  (FastAPI + built React UI)                      │
   │    OpenCV 5.1 (COOL, KleidiCV) ── decode · adaptive sampling · optical    │
   │    flow · MOG2 · cv2.dnn YOLO inference · ORB re-ID · Kalman tracking     │
   │    visual memory (SQLite) ── retrieval ── evidence-grounded answers       │
   └───────────────┬───────────────────────────────────────────────────────────┘
                   │ IAM instance role (no keys on the box)
                   ▼
            Amazon S3 (private bucket)
            sessions/<id>/  video · memory.db · evidence/*.jpg
```

## AWS services and why

| Service | Role in VISTA | Why this and not something else |
|---|---|---|
| **EC2 on Graviton (m9g.xlarge)** | Runs the whole OpenCV pipeline and API | The workload is CPU video analysis; Graviton gives better price/performance than x86 and is the target of COOL. 4 vCPU / 16 GB is enough for 1080p footage. |
| **COOL AMI (AWS Marketplace)** | OpenCV 5.1 build with KleidiCV and Arm SVE/NEON tuning | Required for the *Best Use of COOL* path: COOL executes the core workload on the Arm instance. Usage-based, about $0.01/h on this size. |
| **Amazon S3** | Durable copy of source video, memory database and evidence frames | The instance is disposable; S3 keeps the archive. Local disk stays the working copy because OpenCV needs file paths. Public access blocked. |
| **IAM role + instance profile** | Lets the instance read/write only the VISTA bucket | No access keys are stored on the instance. Scope: `GetObject/PutObject/DeleteObject` on `bucket/*`, `ListBucket` on the bucket. |
| **EC2 security group** | Ports 22 and 8000 open to the operator's IP only | The API has no authentication, so it is not exposed to the internet. |
| **AWS Budgets** | $50 cost alert | Credit protection; it only notifies, so instances are also stopped by hand when idle. |

Not used, deliberately: a load balancer, containers/Kubernetes, Lambda, or a
managed database. The MVP is a single-session service; adding them would
hide the computer-vision contribution without adding evidence.

## Resource requirements

- Compute: m9g.xlarge (4 vCPU Neoverse-V3, 16 GB). A c8g/m8g also works with the Graviton4 COOL AMI **only if it ships OpenCV 5** (the Graviton4 v2 image we checked is OpenCV 4.14 and was rejected).
- Disk: 80 GB gp3 (the AMI snapshot size); a few GB are used.
- Network: egress only for dataset downloads; the API is allow-listed.
- Model: `yolov8n.onnx` (12 MB) for `cv2.dnn`. OWLv2 (open-vocabulary, PyTorch) is optional and not part of the COOL claim.
- Estimated cost: about $0.20/h for the instance (EC2 plus the COOL fee). Stopped instances cost only the EBS disk (about $0.2/day).

## Reproducing the deployment

Prerequisites: AWS CLI configured for a user that can create IAM roles, EC2 and S3 resources; an active subscription to the COOL listing for your Graviton generation; Git Bash (the scripts are bash) and `uv`.

```bash
bash infra/provision.sh          # bucket, IAM role/profile, security group (your IP), key pair
bash infra/launch.sh cool-arm    # m9g.xlarge from the COOL AMI
bash infra/deploy.sh cool-arm    # venv + pinned deps + systemd service, then prints the URL
bash infra/stop.sh cool-arm      # stop when idle (disk only is billed)
bash infra/start.sh cool-arm     # start again; the service comes back by itself (IP changes)
bash infra/teardown.sh --all     # remove instances, bucket, role and key pair
```

What `deploy.sh cool-arm` does (see `infra/deploy_cool.sh`):

1. Exports the pinned requirements from `uv.lock` **minus** `opencv-*` and `numpy`, so pip OpenCV can never shadow COOL's build.
2. Ships `git archive HEAD`, the built UI and the YOLO ONNX model.
3. Creates a Python 3.12 venv and installs the pins with `--no-deps`.
4. Installs a systemd unit that runs `backend/run.py` with
   `PYTHONPATH=/opt/cool/python_3.12/site-packages` and
   `LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib`, `VISTA_STORAGE=s3`.
5. Verifies that `cv2` imports from `/opt/cool` and reports `5.1.0-dev`.

Baseline instances for the COOL comparison use `infra/launch.sh base-arm|base-x86` and `infra/bench_box.sh <mode>` (stock pip OpenCV, identical benchmark).

## Evidence that COOL runs the workload

- Instance type `m9g.xlarge`, AMI `ami-00f40acc34f3d7d39` (us-east-1), Ubuntu 26.04.
- `cv2.__file__` is `/opt/cool/python_3.12/site-packages/cv2/__init__.py`, version `5.1.0-dev`, build `5.0.0-352-g565ed6d199`, custom HAL `KleidiCV 0.7.0`, baselines SVE/NEON/FP16/BF16.
- The running API process maps `libopencv_core.so.5.1.0` and `libopencv_imgproc.so.5.1.0` from `/opt/cool/cpp_sdk/lib` (checked in `/proc/<pid>/maps`); the venv contains no pip OpenCV or numpy.
- `scripts/bench_cv.py` records the build information and CPU in every result file.

## Security and data handling

- Bucket public access is blocked; instance credentials come from the role.
- The security group admits one IP; re-run `infra/provision.sh` after your IP changes.
- Uploaded video is stored under a random session id and mirrored to S3 as-is; delete the `sessions/<id>/` prefix to purge it. Third-party benchmark data (MOT17, Objectron) is downloaded at run time and not stored in the repository.
- After the competition: run `infra/teardown.sh --all`, cancel the Marketplace COOL subscription, and delete the IAM access key.

## Known limits

- Single session: one active video at a time (in-memory session state).
- No authentication on the API; it relies on the security group.
- The on-instance model is a closed-set COCO detector. Open-vocabulary detection (OWLv2) needs PyTorch and is run locally or on a GPU box, not on the COOL path.
