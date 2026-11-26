"""Draw images/ray_actors.png, the driver and two Ray actors in notebook 2 §4. Run: python make_ray_actors.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DRIVER = ("#574a8c", "#ede8f6")   # purple, as the policy in the notebook 1 loop diagram
ROLLOUT = ("#2e6da4", "#e3eefa")  # blue, as SAMPLE
TRAINING = ("#a83a3a", "#fbe7e7") # red, as UPDATE
FILE = ("#9a9a9a", "#f1f1f1")
TEXT = "#333333"

figure, axes = plt.subplots(figsize=(15, 6), dpi=100)
axes.set_xlim(0, 18.6)
axes.set_ylim(0, 7)
axes.axis("off")


def box(x, y, width, height, colors, title, subtitle, title_color=None):
    edge, fill = colors
    axes.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.15",
                                  linewidth=2.5, edgecolor=edge, facecolor=fill))
    axes.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", fontsize=15, fontweight="bold",
              color=title_color or edge)
    axes.text(x + width / 2, y + height * 0.3, subtitle, ha="center", va="center", fontsize=12.5, color=TEXT)


def arrow(start, end, label=None, label_y=None, rad=0.0):
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2, color=TEXT,
                                   connectionstyle=f"arc3,rad={rad}"))
    if label:
        axes.text((start[0] + end[0]) / 2, label_y, label, ha="center", va="center", fontsize=12, color=TEXT,
                  family="monospace", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


# The driver: this notebook's process, which runs the training loop
box(0.3, 1.2, 3.4, 4.6, DRIVER, "driver", "this notebook:\nthe training loop,\nno GPU work")

# One GPU, shared by the two actors
gpu_x = 9.4
axes.add_patch(FancyBboxPatch((gpu_x - 0.4, 0.7), 4.8, 5.6, boxstyle="round,pad=0,rounding_size=0.2",
                              linewidth=1.5, edgecolor="#777777", facecolor="none", linestyle="--"))
axes.text(gpu_x + 2.0, 6.55, "one GPU", ha="center", va="center", fontsize=13, color="#555555", style="italic")
box(gpu_x, 3.9, 4.0, 1.9, ROLLOUT, "RolloutWorker", "vLLM\n(num_gpus=0.3)")
box(gpu_x, 1.2, 4.0, 1.9, TRAINING, "TrainingWorker", "policy + Adam\n(num_gpus=0.6)")

# Calls from the driver, and their results, through Ray
arrow((3.7, 5.25), (gpu_x, 5.25), "generate.remote(prompts)", 5.6)
arrow((gpu_x, 4.45), (3.7, 4.45), "samples", 4.1)
arrow((3.7, 2.55), (gpu_x, 2.55), "compute_log_probs.remote(batch)\nupdate.remote(batch)", 3.1)
arrow((gpu_x, 1.75), (3.7, 1.75), "log-probs, metrics", 1.4)
axes.text((3.7 + gpu_x) / 2, 0.55, "calls and results travel through Ray's object store", ha="center", va="center",
          fontsize=12, color=TEXT, style="italic")

# Weight sync: through a file in RAM; only its path goes through Ray
file_x = gpu_x + 5.0
box(file_x, 2.85, 3.6, 1.3, FILE, "/dev/shm/…", "model.safetensors", title_color=TEXT)
arrow((gpu_x + 4.0, 2.0), (file_x + 1.8, 2.85), rad=0.3)
axes.text(gpu_x + 4.3, 2.55, "writes", ha="left", va="center", fontsize=12, color=TEXT, family="monospace")
arrow((file_x + 1.8, 4.15), (gpu_x + 4.0, 5.0), rad=0.3)
axes.text(gpu_x + 4.3, 4.45, "reads", ha="left", va="center", fontsize=12, color=TEXT, family="monospace")
axes.text(file_x + 1.8, 1.4, "weight sync: only the\npath travels through Ray", ha="center", va="center",
          fontsize=12, color=TEXT, style="italic")

figure.savefig(Path(__file__).with_name("ray_actors.png"), bbox_inches="tight", facecolor="white")
