# Demo video script (5:00 maximum)

The rules ask for a public or unlisted video of **no more than five minutes** that shows **the team, the application
working, its architecture and its principal results**. This script is timed to 4:40 to leave margin.

## Before recording

- Start the COOL instance and deploy the final commit: `bash infra/start.sh cool-arm && bash infra/deploy.sh cool-arm`
  (or `infra/launch.sh cool-arm` first if the instance was terminated). Open `http://<ip>:8000` from the allow-listed IP.
- Have two clips ready: a desk scene with a laptop (about 5 s) and the phone-charger clip. Use **your own footage only**, with
  nobody identifiable in it (the repo's `video_clip1.mp4` contains a printed photo of a face: avoid it on camera).
- Have these tabs open: the app, `docs/architecture.png`, `docs/agent_workflow.png`, `docs/BENCHMARK.md` (rendered on GitHub),
  the AWS console on the EC2 instance page (shows `m9g.xlarge`, Graviton), and the S3 bucket listing.
- Record the screen plus a small camera view of the team (required). Record in 1080p, at 100% browser zoom.

## Script

| Time | On screen | Say |
|---|---|---|
| 0:00-0:25 | Team on camera, then title slide | "We are [names]. VISTA, the Visual Intelligence and Spatial Temporal Archive, turns video into a memory you can question: where did I leave my charger, and what proves it?" |
| 0:25-1:10 | App: drop the desk clip; processing, then the ready view | "I upload a clip. An OpenCV 5 pipeline decodes and samples it, detects and tracks objects, and builds a persistent memory. This is running on AWS, not on my laptop." |
| 1:10-2:00 | Ask "Where is the laptop?"; click the evidence frame; timeline | "The answer cites a frame. I click it and the video seeks to the moment. Every answer is grounded in stored evidence, and 'I cannot determine' is an allowed answer." |
| 2:00-2:50 | Ask "Where is the phone charger?" (not in the detector's list): show the on-demand notice | "The charger was never in the detector's vocabulary. VISTA adds the word, re-scans the video, and finds it. Anything you can name, it can look for." |
| 2:50-3:40 | Open the **Agent trace** under an answer; show a case that asks you to confirm | "This is the agentic part. The agent does not trust its own lookup. It crops the claimed box from the source frame, zooms in and re-runs OpenCV detection. What it sees decides what it does: accept, try another candidate, re-scan at higher sampling, or ask me. Here it could not verify, so it shows me the options and I pick. It never guesses a location." |
| 3:40-4:15 | `architecture.png`, then EC2 and S3 consoles | "On AWS: a Graviton5 EC2 instance from the Cloud Optimized OpenCV AMI runs OpenCV 5.1; S3 keeps the video, memory and evidence; an IAM role gives the instance access to that one bucket; a security group admits one IP. All of it is scripted in `infra/`." |
| 4:15-4:40 | `BENCHMARK.md` table, then `EVALUATION.md` summary | "We measured it. On Graviton the pipeline is about 1.33 times faster and 31 percent cheaper per frame than x86. COOL matches stock OpenCV on the same hardware, which we report as measured. On 20 real phone videos the lead answer is right 75 percent of the time with a small detector and 90 percent with an open-vocabulary one, and we list the failures. Thank you." |

## What not to claim

- Do not say COOL is faster than stock OpenCV on Arm: the measured difference is within noise (+2.8%).
- Do not say the agent improves accuracy: on held-out clips it makes answers checkable and safer, and its automatic accuracy is
  about equal to the plain answer (see `docs/EVALUATION.md`, section 4).
- Do not call tracking state of the art: MOT17 MOTA is 0.15-0.53 with a nano detector.

## After recording

Run `bash infra/teardown.sh --all` (the agreed plan: record, then terminate). Keep the Marketplace subscriptions until after
Nov 10 so the deployment can be rebuilt in about 10-15 minutes if a judge asks for a live session.
