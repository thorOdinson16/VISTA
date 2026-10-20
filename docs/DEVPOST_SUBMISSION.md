# Devpost submission text (draft)

Fields marked **[YOU]** need information only the submitter has.

## Project name

VISTA: Visual Intelligence and Spatial Temporal Archive

## Tagline

Ask your video where things are. Every answer comes with the frame that proves it.

## Team

I'm Abhijna DS, an undergraduate at PES University in Bengaluru, India, interested in machine learning and computer vision. This is a solo entry: I designed and built everything, including the OpenCV pipeline, tracking and memory, the verify agent, the AWS deployment, the evaluation and the interface. I've taken part in hackathons before, including the Amazon ML Challenge 2026, where my team ranked 212 on the public leaderboard. Contact: abhijnads16@gmail.com

## Competition paths

Core entry, plus both optional paths: **Best Use of COOL** and **Agentic Vision**.

## What it does

VISTA turns video into a persistent, queryable memory of objects and events. Upload a clip and ask "where is my phone
charger?" or "what happened to the notebook?". An OpenCV 5 pipeline decodes and samples the video, detects, tracks and
re-identifies objects, and stores them with their evidence frames. Answers separate observed facts from inferences and cite
frames. If you ask about something the detector was never prompted with, VISTA adds the word and re-scans the video. A verify
agent checks each "where is it?" answer on a zoomed crop and falls back to a re-scan or to the human instead of guessing.

## How we built it

- **OpenCV 5**: `VideoCapture` decode, adaptive sampling with Farneback optical flow and MOG2, `cv2.dnn` YOLO inference,
  ORB and HSV-histogram re-identification, Kalman tracking, cubic zoom crops for verification.
- **AWS**: EC2 on **Graviton5** from the **Cloud Optimized OpenCV (COOL)** AMI (OpenCV 5.1 with KleidiCV); Amazon S3 for the
  video, memory database and evidence; an IAM instance role scoped to one bucket; a security group admitting one IP; AWS
  Budgets for cost control. Everything is reproducible with `infra/*.sh`.
- **Agentic Vision**: lookup, crop and re-detect, re-scan at denser sampling, then ask the human; each step is logged as a
  trace shown in the UI. The detector's output decides the next tool call or the request for approval. No language model.
- Python, FastAPI, React, SQLite. Dependencies are pinned in `uv.lock`.

## COOL path: what we measured

The same OpenCV workload on three instances (`docs/BENCHMARK.md`): COOL on Graviton5, stock pip OpenCV on the same Graviton5,
and stock pip OpenCV on x86.

| | Pipeline fps | Cost per million frames |
|---|---:|---:|
| COOL, Graviton5 | 20.1 | $2.85 |
| pip OpenCV, Graviton5 | 19.5 | $2.79 |
| pip OpenCV, x86 | 14.6 | $4.02 |

Graviton is about 1.33x faster and 31% cheaper per frame than x86. COOL is **within noise** of stock OpenCV on the same Arm
hardware (+2.8%): it is faster on ORB features and slower on decode and `cv2.dnn`, and the pip arm64 wheel already ships
KleidiCV. We report that as measured. Caveats: COOL is 5.1.0-dev versus pip 5.0.0, one clip, one instance per setup.

## Agentic Vision path

Workflow diagram: `docs/agent_workflow.png`. On 20 held-out Objectron clips the agent answered 17 automatically with 88%
precision (16 at 94% on the COOL instance), deferred one to the human and reported two as not found; its automatic accuracy (75%) is about equal to the plain
answer (80%), so its value is checkability and safe failure handling, not accuracy. A trace for every question, the
human-confirmation endpoint and the evaluation of failure handling are in `docs/EVALUATION.md`, section 4.

## Evaluation (with failures)

- **Objectron** (20 real phone videos, "where is the X?"): lead answer correct 75% with YOLOv8n on a laptop (55% before our ranking and
  tracker fix) and 90% with OWLv2; the same YOLOv8n re-run on the deployed COOL instance scored 60% (cause not isolated).
- **MOT17** (4 sequences): MOTA 0.15-0.53, IDF1 0.22-0.54; precision high, recall low on crowded 1080p scenes.
- **Hand-checked clips**: 2/5 with YOLOv8n, 4/5 with OWLv2 plus on-demand re-scan.
- **Failures we found**: a flat book missed on a shop shelf, a flashlight package reported as a notebook, crowded pedestrians
  missed by a nano detector, 132 false-positive tracks on one OWLv2 clip, and a printed photo of a face reported as a person.

## Challenges

We assumed a poor result came from one object fragmenting into many tracks; looking at the data showed it was persistent
false-positive tracks instead, so we corrected our own write-up. We also found that our first crop check distorted thin
objects, fixed it, and then evaluated on clips we had not looked at.

## Limitations and responsible use

Small evaluation samples; closed-set detector is weak on flat or unusual views; single session with no API authentication
(IP-restricted); the agent re-uses the same detector so it cannot catch a consistent mistake. VISTA stores video and
per-person tracks: process footage only with consent, purge a session by deleting its S3 prefix, and treat person
identities as suggestions that need review.

## Judge access

The AWS deployment is torn down after the recording to save credits; it is rebuilt from the repo in about 10-15 minutes
(`docs/AWS_DEPLOYMENT.md`). For judging, a live screen-share is available on request: email abhijnads16@gmail.com and the instance is started from the repo scripts.

## Links

- Code: https://github.com/thorOdinson16/VISTA (public)
- Video (5 minutes or less, shows the team): **[YOU]**
- Technical report: `docs/TECHNICAL_REPORT.md`
- Architecture: `docs/architecture.png`, agent workflow: `docs/agent_workflow.png`
