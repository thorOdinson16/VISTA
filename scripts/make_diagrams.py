"""Draw the architecture and agent-workflow diagrams (docs/*.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs"
BG, GOLD, TEXT, MUTED, PANEL, AWS = "#0b0b0b", "#d4af37", "#f1ece0", "#a49c88", "#16140f", "#ff9900"


def box(ax, x, y, w, h, title, sub="", edge=GOLD, fill=PANEL, fs=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=edge, facecolor=fill,
        )
    )
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            color=TEXT, fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", color=MUTED, fontsize=fs - 2.5)


def arrow(ax, x1, y1, x2, y2, label="", color=GOLD, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center", color=MUTED, fontsize=7.5)


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(0.3, h - 0.35, title, color=GOLD, fontsize=15, fontweight="bold", va="center")
    return fig, ax


def architecture() -> None:
    fig, ax = canvas(13, 7.2, "VISTA on AWS: OpenCV 5 (COOL) pipeline, memory and agent")
    # AWS EC2 container
    ax.add_patch(FancyBboxPatch((0.3, 0.7), 9.3, 5.4, boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=1.6, edgecolor=AWS, facecolor="none", linestyle="--"))
    ax.text(0.55, 5.85, "EC2 m9g.xlarge · AWS Graviton5 (Arm64) · COOL AMI (OpenCV 5.1, KleidiCV)",
            color=AWS, fontsize=9.5, fontweight="bold", va="center")

    box(ax, 0.6, 4.35, 2.1, 1.0, "Video in", "upload · phone clip", fs=9.5)
    box(ax, 3.0, 4.35, 2.6, 1.0, "OpenCV ingest", "decode · adaptive sampling", fs=9.5)
    box(ax, 5.9, 4.35, 3.4, 1.0, "Detect", "cv2.dnn YOLO  /  OWLv2 (open vocab)", fs=9.5)
    box(ax, 0.6, 2.85, 2.7, 1.0, "Track + re-ID", "Kalman · IoU · ORB · histograms", fs=9.5)
    box(ax, 3.6, 2.85, 2.7, 1.0, "Visual memory", "objects · events · scene · evidence", fs=9.5)
    box(ax, 6.6, 2.85, 2.7, 1.0, "Query + retrieval", "parse · rank sightings", fs=9.5)
    box(ax, 3.6, 1.0, 2.7, 1.0, "Verify agent", "crop check · re-scan · ask human", edge="#7fd1ae", fs=9.5)
    box(ax, 6.6, 1.0, 2.7, 1.0, "Grounded answer", "text + frame evidence + trace", fs=9.5)
    box(ax, 0.6, 1.0, 2.7, 1.0, "SQLite memory", "per-session store", fs=9.5)

    arrow(ax, 2.7, 4.85, 3.0, 4.85)
    arrow(ax, 5.6, 4.85, 5.9, 4.85)
    arrow(ax, 7.6, 4.35, 2.2, 3.85, "")
    arrow(ax, 3.3, 3.35, 3.6, 3.35)
    arrow(ax, 6.3, 3.35, 6.6, 3.35)
    arrow(ax, 7.95, 2.85, 7.95, 2.0)
    arrow(ax, 6.6, 3.0, 5.3, 2.0, "candidates", color="#7fd1ae")
    arrow(ax, 4.9, 2.0, 4.9, 2.85, "verified / rejected", color="#7fd1ae")
    arrow(ax, 3.6, 3.2, 3.3, 2.0, "")
    # S3 and UI outside the EC2 box
    box(ax, 10.1, 4.35, 2.6, 1.0, "Amazon S3", "video · memory.db · evidence", edge=AWS, fs=9.5)
    box(ax, 10.1, 2.85, 2.6, 1.0, "IAM instance role", "bucket-scoped, no keys", edge=AWS, fs=9.5)
    box(ax, 10.1, 1.0, 2.6, 1.0, "React UI", "video · timeline · trace", fs=9.5)
    arrow(ax, 9.6, 4.85, 10.1, 4.85, "mirror", color=AWS)
    arrow(ax, 11.4, 3.85, 11.4, 4.35, "", color=AWS)
    arrow(ax, 9.3, 1.5, 10.1, 1.5, "HTTP :8000")
    ax.text(0.3, 0.3, "Security group admits one IP · Budgets alert · infra/*.sh reproduces everything",
            color=MUTED, fontsize=8)
    fig.savefig(OUT / "architecture.png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def agent_workflow() -> None:
    fig, ax = canvas(13, 6.6, "Agentic Vision: OpenCV output decides the next action")
    box(ax, 0.4, 4.6, 2.4, 1.0, "Question", "\"Where is the X?\"", fs=10)
    box(ax, 3.3, 4.6, 2.6, 1.0, "lookup_memory", "ranked sightings", fs=10)
    box(ax, 6.4, 4.6, 2.9, 1.0, "inspect_crop", "zoom last box · re-detect", edge="#7fd1ae", fs=10)
    box(ax, 9.9, 4.6, 2.8, 1.0, "Answer", "verified · cites frame", fs=10)
    box(ax, 3.3, 2.5, 2.6, 1.0, "rescan (label)", "new vocabulary", fs=10)
    box(ax, 6.4, 2.5, 2.9, 1.0, "rescan (denser)", "6 fps sampling", fs=10)
    box(ax, 9.9, 2.5, 2.8, 1.0, "ask_human", "choose among sightings", edge=AWS, fs=10)
    box(ax, 9.9, 0.7, 2.8, 1.0, "Confirmed answer", "human decision logged", edge=AWS, fs=10)
    arrow(ax, 2.8, 5.1, 3.3, 5.1)
    arrow(ax, 5.9, 5.1, 6.4, 5.1)
    arrow(ax, 9.3, 5.1, 9.9, 5.1, "verified", color="#7fd1ae")
    arrow(ax, 4.6, 4.6, 4.6, 3.5, "nothing found")
    arrow(ax, 4.6, 2.5, 4.3, 4.6, "", color=MUTED)
    arrow(ax, 7.85, 4.6, 7.85, 3.5, "not verified")
    arrow(ax, 9.3, 3.0, 9.9, 3.0, "still unverified", color=AWS)
    arrow(ax, 7.85, 3.5, 6.0, 4.75, "look again", color=MUTED)
    arrow(ax, 11.3, 2.5, 11.3, 1.7, "", color=AWS)
    ax.text(0.4, 1.55, "Every tool call, its numeric observation and the decision it caused is returned as a trace.",
            color=MUTED, fontsize=9)
    ax.text(0.4, 1.15, "No candidate verified and no human answer: the agent says it cannot determine, it never guesses.",
            color=MUTED, fontsize=9)
    fig.savefig(OUT / "agent_workflow.png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    agent_workflow()
    print("wrote docs/architecture.png and docs/agent_workflow.png")
