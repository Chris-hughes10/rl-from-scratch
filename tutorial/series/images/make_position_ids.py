"""Draw images/position_ids.png, the position-IDs example in notebook 1 §1.7. Run: python make_position_ids.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROMPT = ("#2e6da4", "#e3eefa")    # as in padded_batch.png
PADDING = ("#9a9a9a", "#f1f1f1")
WRONG = "#a83a3a"                  # red, as UPDATE in the loop diagram
RIGHT = "#2e7d4f"                  # green
MUTED = "#9a9a9a"
TEXT = "#333333"

tokens = ["·", "·", "P", "P", "P", "P", "P", "P"]
num_padding = tokens.count("·")
default_positions = list(range(len(tokens)))                                             # the column number
position_ids = [0] * num_padding + list(range(len(tokens) - num_padding))                # counting real tokens only

cell, gap = 0.9, 0.12
left_margin = 4.6
figure, axes = plt.subplots(figsize=(12, 4.2), dpi=100)
axes.set_xlim(0, left_margin + len(tokens) * (cell + gap) + 4.4)
axes.set_ylim(-2.3, 1.6)
axes.axis("off")
axes.set_aspect("equal")


def column_x(column):
    return left_margin + column * (cell + gap)


def row_label(y, text, color=TEXT):
    axes.text(left_margin - 0.35, y, text, ha="right", va="center", fontsize=15, color=color)


# The tokens of one left-padded row
token_y = 0.0
row_label(token_y + cell / 2, "token")
for column, kind in enumerate(tokens):
    edge, fill = PROMPT if kind == "P" else PADDING
    axes.add_patch(FancyBboxPatch((column_x(column), token_y), cell, cell, boxstyle="round,pad=0,rounding_size=0.12",
                                  linewidth=2, edgecolor=edge, facecolor=fill))
    axes.text(column_x(column) + cell / 2, token_y + cell / 2, kind, ha="center", va="center", fontsize=16,
              fontweight="bold", color=edge)
    axes.text(column_x(column) + cell / 2, token_y + cell + 0.35, str(column), ha="center", va="center", fontsize=13, color=MUTED)
row_label(token_y + cell + 0.35, "column", MUTED)
axes.text(column_x(len(tokens)) + 0.1, token_y + cell / 2, "…", ha="left", va="center", fontsize=18, color=TEXT)


def number_row(y, label, values, color, note):
    row_label(y, label, color)
    for column, value in enumerate(values):
        value_color = MUTED if tokens[column] == "·" else color  # padding is hidden by the attention mask
        axes.text(column_x(column) + cell / 2, y, str(value), ha="center", va="center", fontsize=17, fontweight="bold",
                  color=value_color)
    axes.text(column_x(len(tokens)) + 0.6, y, note, ha="left", va="center", fontsize=14, color=color)


number_row(-0.85, "default positions", default_positions, WRONG, "✗ every position 2 too high")
number_row(-1.75, "position_ids", position_ids, RIGHT, "✓ counting real tokens only")

figure.savefig(Path(__file__).with_name("position_ids.png"), bbox_inches="tight", facecolor="white")
