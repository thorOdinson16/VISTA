"""Run the VISTA evaluation harness against the demo scene.

Generates the demo video + ground truth when missing, runs the full
pipeline, and reports event/tracking/query metrics. Use ``--strict`` to
exit non-zero when minimum thresholds are not met.

Usage:

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --strict --json data/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.evaluation.demo import generate_demo
from backend.app.evaluation.harness import format_report, run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="VISTA evaluation harness")
    parser.add_argument("--video", default="data/demo.mp4")
    parser.add_argument("--gt", default="data/demo_gt.json")
    parser.add_argument("--json", default=None, help="write the report as JSON")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when thresholds are not met",
    )
    args = parser.parse_args()

    if not Path(args.video).exists() or not Path(args.gt).exists():
        print("Demo assets missing; generating...")
        generate_demo(
            out=args.video, gt=args.gt, fps=args.fps, duration=args.duration
        )

    report = run_evaluation(args.video, args.gt)
    print(format_report(report))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")

    if args.strict and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
