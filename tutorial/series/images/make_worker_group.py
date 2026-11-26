"""Draw images/worker_group.png, the two dispatch rules of a worker group in notebook 2 §5. Run: python make_worker_group.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DRIVER = ("#574a8c", "#ede8f6")  # purple, as the driver in ray_actors.png
WORKER = ("#a83a3a", "#fbe7e7")  # red, as the TrainingWorker in ray_actors.png
ROWS_A = ("#2e6da4", "#e3eefa")  # blue: the first half of the batch
ROWS_B = ("#2e7d4f", "#e6f4ea")  # green: the second half
TEXT = "#333333"

figure, axes = plt.subplots(figsize=(16, 5.2), dpi=100)
axes.set_xlim(0, 20.5)
axes.set_ylim(0, 6.4)
axes.axis("off")


def box(x, y, width, height, colors, text, fontsize=14, bold=True):
    edge, fill = colors
    axes.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.12",
                                  linewidth=2.2, edgecolor=edge, facecolor=fill))
    axes.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize,
              fontweight="bold" if bold else "normal", color=edge)


def arrow(start, end, label=None, label_offset=0.25):
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=TEXT))
    if label:
        axes.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + label_offset, label, ha="center", va="center",
                  fontsize=11.5, color=TEXT, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


# ── one-to-all: the same call to every worker ────────────────────────────────
axes.text(3.6, 6.0, "one-to-all:  init_model()", ha="center", va="center", fontsize=15, fontweight="bold", color=TEXT)
box(0.3, 2.3, 2.2, 1.2, DRIVER, "driver")
box(5.0, 3.9, 2.4, 1.0, WORKER, "worker 0")
box(5.0, 0.9, 2.4, 1.0, WORKER, "worker 1")
arrow((2.5, 3.2), (5.0, 4.3), "same call", label_offset=0.5)
arrow((2.5, 2.6), (5.0, 1.5), "same call", label_offset=-0.5)
axes.text(3.6, 0.25, "returns [result 0, result 1]", ha="center", va="center", fontsize=12, color=TEXT, style="italic")

# Divider
axes.plot([8.4, 8.4], [0.2, 6.2], color="#cccccc", linewidth=1.5)

# ── data-parallel: split the batch, one piece per worker, then concatenate ──
axes.text(14.5, 6.0, "data-parallel:  compute_log_probs(batch of 128 rows)", ha="center", va="center",
          fontsize=15, fontweight="bold", color=TEXT)
box(9.0, 3.9, 2.6, 1.0, ROWS_A, "rows 0–63", fontsize=13)
box(9.0, 0.9, 2.6, 1.0, ROWS_B, "rows 64–127", fontsize=13)
axes.text(10.3, 2.9, "chunk(2)", ha="center", va="center", fontsize=12, color=TEXT, family="monospace")
box(13.0, 3.9, 2.4, 1.0, WORKER, "worker 0")
box(13.0, 0.9, 2.4, 1.0, WORKER, "worker 1")
arrow((11.6, 4.4), (13.0, 4.4))
arrow((11.6, 1.4), (13.0, 1.4))
box(16.0, 3.9, 1.6, 1.0, ROWS_A, "64 rows", fontsize=12, bold=False)
box(16.0, 0.9, 1.6, 1.0, ROWS_B, "64 rows", fontsize=12, bold=False)
arrow((15.4, 4.4), (16.0, 4.4))
arrow((15.4, 1.4), (16.0, 1.4))
# concat: both halves, back in order
axes.add_patch(FancyBboxPatch((18.6, 1.6), 1.7, 2.5, boxstyle="round,pad=0,rounding_size=0.12",
                              linewidth=2.2, edgecolor=TEXT, facecolor="white"))
axes.add_patch(FancyBboxPatch((18.75, 2.95), 1.4, 1.05, boxstyle="round,pad=0,rounding_size=0.08",
                              linewidth=0, facecolor=ROWS_A[1]))
axes.add_patch(FancyBboxPatch((18.75, 1.7), 1.4, 1.05, boxstyle="round,pad=0,rounding_size=0.08",
                              linewidth=0, facecolor=ROWS_B[1]))
axes.text(19.45, 3.47, "0–63", ha="center", va="center", fontsize=12, color=ROWS_A[0])
axes.text(19.45, 2.22, "64–127", ha="center", va="center", fontsize=12, color=ROWS_B[0])
arrow((17.6, 4.4), (18.6, 3.6))
arrow((17.6, 1.4), (18.6, 2.2))
axes.text(19.45, 0.9, "concat:\n128 rows", ha="center", va="center", fontsize=12, color=TEXT, family="monospace")

figure.savefig(Path(__file__).with_name("worker_group.png"), bbox_inches="tight", facecolor="white")
