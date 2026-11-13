"""Draw images/token_level_rewards.png, verl's per-token reward layout in notebook 1 §2. Run: python make_token_level_rewards.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESPONSE = ("#2e7d4f", "#e6f4ea")  # as in padded_batch.png
PADDING = ("#9a9a9a", "#f1f1f1")
MUTED = "#9a9a9a"
TEXT = "#333333"

tokens = ["R", "R", "R", "R", "·", "·"]
last_real = max(column for column, kind in enumerate(tokens) if kind == "R")
rewards = [1 if column == last_real else 0 for column in range(len(tokens))]  # a response with reward 1

cell, gap = 0.9, 0.12
left_margin = 5.0
figure, axes = plt.subplots(figsize=(10, 3.6), dpi=100)
axes.set_xlim(0, left_margin + len(tokens) * (cell + gap) + 0.4)
axes.set_ylim(-2.6, 1.3)
axes.axis("off")
axes.set_aspect("equal")


def column_x(column):
    return left_margin + column * (cell + gap)


def row_label(y, text):
    axes.text(left_margin - 0.35, y, text, ha="right", va="center", fontsize=15, color=TEXT)


# The response tokens of one right-padded row
token_y = 0.0
row_label(token_y + cell / 2, "response")
for column, kind in enumerate(tokens):
    edge, fill = RESPONSE if kind == "R" else PADDING
    axes.add_patch(FancyBboxPatch((column_x(column), token_y), cell, cell, boxstyle="round,pad=0,rounding_size=0.12",
                                  linewidth=2, edgecolor=edge, facecolor=fill))
    axes.text(column_x(column) + cell / 2, token_y + cell / 2, kind, ha="center", va="center", fontsize=16,
              fontweight="bold", color=edge)

# One reward per token: our reward on the last real token, 0 everywhere else
reward_y = -0.75
row_label(reward_y, "token_level_rewards")
for column, value in enumerate(rewards):
    color = RESPONSE[0] if column == last_real else MUTED
    axes.text(column_x(column) + cell / 2, reward_y, str(value), ha="center", va="center", fontsize=17, fontweight="bold",
              color=color)

# Label the reward
reward_x = column_x(last_real) + cell / 2
axes.add_patch(FancyArrowPatch((reward_x, -1.85), (reward_x, -1.1), arrowstyle="-|>", mutation_scale=20, linewidth=2,
                               color=TEXT))
axes.text(reward_x, -2.15, "our reward, on the last real token", ha="center", va="center", fontsize=14, color=TEXT)

figure.savefig(Path(__file__).with_name("token_level_rewards.png"), bbox_inches="tight", facecolor="white")
