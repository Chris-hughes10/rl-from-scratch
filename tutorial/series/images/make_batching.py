"""Draw images/static_batching.png and images/continuous_batching.png for notebook 2 §1. Run: python make_batching.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# edge, fill: the same palette as the notebook 1 images
SEQUENCES = {
    "A": ("#2e6da4", "#e3eefa"),  # blue
    "B": ("#2e7d4f", "#e6f4ea"),  # green
    "C": ("#b07d12", "#fff4e0"),  # amber
    "D": ("#a83a3a", "#fbe7e7"),  # red
    "E": ("#574a8c", "#ede8f6"),  # purple
}
IDLE = ("#9a9a9a", "#f1f1f1")
TEXT = "#333333"

NUM_STEPS = 8
# One string per batch slot: the sequence generating a token at each step, "·" when the slot is idle
STATIC = ["AAAAAAAA", "BBB·····", "CCCCC···"]
CONTINUOUS = ["AAAAAAAA", "BBBDDDDD", "CCCCCEEE"]

cell, gap = 0.9, 0.12
panel_width = NUM_STEPS * (cell + gap)
label_width = 1.6


def draw(slots, title, caption, filename):
    figure, axes = plt.subplots(figsize=(7.5, 4.4), dpi=100)
    axes.set_xlim(0, label_width + panel_width + 0.2)
    axes.set_ylim(-1.9, len(slots) * (cell + gap) + 1.2)
    axes.axis("off")
    axes.set_aspect("equal")
    top = (len(slots) - 1) * (cell + gap)

    for slot_index, tokens in enumerate(slots):
        row_y = top - slot_index * (cell + gap)
        axes.text(label_width - 0.3, row_y + cell / 2, f"slot {slot_index}", ha="right", va="center", fontsize=15, color=TEXT)
        for step, token in enumerate(tokens):
            edge, fill = SEQUENCES.get(token, IDLE)
            x = label_width + step * (cell + gap)
            axes.add_patch(FancyBboxPatch((x, row_y), cell, cell, boxstyle="round,pad=0,rounding_size=0.12",
                                          linewidth=2, edgecolor=edge, facecolor=fill))
            axes.text(x + cell / 2, row_y + cell / 2, token, ha="center", va="center", fontsize=16, fontweight="bold", color=edge)

    middle = label_width + panel_width / 2
    axes.text(middle, top + cell + 0.5, title, ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    # Time runs left to right: one generated token per step
    axes.add_patch(FancyArrowPatch((label_width, -0.45), (label_width + panel_width - gap, -0.45), arrowstyle="-|>",
                                   mutation_scale=20, linewidth=2, color=TEXT))
    axes.text(middle, -0.85, "time: one token per step", ha="center", va="center", fontsize=13, color=TEXT)
    axes.text(middle, -1.5, caption, ha="center", va="center", fontsize=14, color=TEXT, style="italic")

    figure.savefig(Path(__file__).with_name(filename), bbox_inches="tight", facecolor="white")
    plt.close(figure)


draw(STATIC, "static batching (model.generate)", "finished slots sit idle until A is done", "static_batching.png")
draw(CONTINUOUS, "continuous batching (vLLM)", "D and E take the slots as soon as B and C end", "continuous_batching.png")
