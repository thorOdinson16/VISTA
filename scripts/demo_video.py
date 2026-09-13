"""Generate the synthetic demo video from section 15 of the requirements.

Thin CLI over :func:`backend.app.evaluation.demo.generate_demo`. Renders a
controlled scene (notebook, mug, Person A) with scripted movement, plus a
ground-truth JSON of events for evaluation/metrics. Objects are drawn as
solid colored shapes so the ``ColorDetector`` can locate them end-to-end
without a trained model.

Usage:

    uv run python scripts/demo_video.py --out data/demo.mp4 --gt data/demo_gt.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.evaluation.demo import generate_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VISTA demo video")
    parser.add_argument("--out", default="data/demo.mp4")
    parser.add_argument("--gt", default="data/demo_gt.json")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--duration", type=float, default=180.0)
    args = parser.parse_args()

    out_path, gt_path = generate_demo(
        out=args.out,
        gt=args.gt,
        fps=args.fps,
        size=tuple(args.size),
        duration=args.duration,
    )
    total_frames = int(args.duration * args.fps)
    print(f"Wrote {out_path} ({total_frames} frames) and {gt_path}")


if __name__ == "__main__":
    main()
