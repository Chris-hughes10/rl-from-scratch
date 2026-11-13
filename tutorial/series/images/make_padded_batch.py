"""Draw images/padded_batch.png, the padded-batch layout in notebook 1 §1.7. Run: python make_padded_batch.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROMPT = ("#2e6da4", "#e3eefa")    # edge, fill: blue, as SAMPLE in the loop diagram
RESPONSE = ("#2e7d4f", "#e6f4ea")  # green, as REWARD in the loop diagram
PADDING = ("#9a9a9a", "#f1f1f1")
TEXT = "#333333"

# One row per sample: (left padding, prompt tokens, response tokens, right padding)
rows = [
    (2, 3, 4, 0),
    (0, 5, 2, 2),
    (1, 4, 3, 1),
]
prompt_width = 5
response_width = 4

cell, gap = 0.9, 0.12
left_margin = 2.0
figure, axes = plt.subplots(figsize=(11, 4.6), dpi=100)
axes.set_xlim(0, left_margin + (prompt_width + response_width) * (cell + gap) + 0.6 + 0.4)
axes.set_ylim(-1.9, len(rows) * (cell + gap) + 1.3)
axes.axis("off")
axes.set_aspect("equal")


def column_x(column):
    split = 0.6 if column >= prompt_width else 0.0  # a small gap between the prompt and response halves
    return left_margin + column * (cell + gap) + split


def draw_cell(column, row_y, kind):
    edge, fill = {"P": PROMPT, "R": RESPONSE, "·": PADDING}[kind]
    x = column_x(column)
    axes.add_patch(FancyBboxPatch((x, row_y), cell, cell, boxstyle="round,pad=0,rounding_size=0.12",
                                  linewidth=2, edgecolor=edge, facecolor=fill))
    axes.text(x + cell / 2, row_y + cell / 2, kind, ha="center", va="center", fontsize=16, fontweight="bold", color=edge)


top = (len(rows) - 1) * (cell + gap)
for row_index, (left_pad, prompt_len, response_len, right_pad) in enumerate(rows):
    row_y = top - row_index * (cell + gap)
    axes.text(left_margin - 0.3, row_y + cell / 2, f"sample {row_index}", ha="right", va="center", fontsize=15, color=TEXT)
    kinds = ["·"] * left_pad + ["P"] * prompt_len + ["R"] * response_len + ["·"] * right_pad
    for column, kind in enumerate(kinds):
        draw_cell(column, row_y, kind)

# Half labels above the grid
label_y = top + cell + 0.45
prompt_mid = (column_x(0) + column_x(prompt_width - 1) + cell) / 2
response_mid = (column_x(prompt_width) + column_x(prompt_width + response_width - 1) + cell) / 2
axes.text(prompt_mid, label_y, "prompts (left-padded)", ha="center", va="center", fontsize=15, color=PROMPT[0], fontweight="bold")
axes.text(response_mid, label_y, "responses (right-padded)", ha="center", va="center", fontsize=15, color=RESPONSE[0], fontweight="bold")

# The column every response starts in
start_x = column_x(prompt_width)
axes.plot([start_x - 0.3, start_x - 0.3], [-0.25, top + cell + 0.15], linestyle="--", linewidth=2, color=TEXT)
axes.add_patch(FancyArrowPatch((start_x + 0.45, -1.15), (start_x + 0.45, -0.15), arrowstyle="-|>", mutation_scale=20,
                               linewidth=2, color=TEXT))
axes.text(start_x + 0.45, -1.45, "every response starts in this column (prompt_width)", ha="center", va="center",
          fontsize=14, color=TEXT)

figure.savefig(Path(__file__).with_name("padded_batch.png"), bbox_inches="tight", facecolor="white")
