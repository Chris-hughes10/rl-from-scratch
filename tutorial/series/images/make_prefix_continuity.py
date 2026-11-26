"""Draw images/prefix_continuity.png, merging two model calls into one row, in notebook 3 §4.
Run: python make_prefix_continuity.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

PROMPT = ("#2e6da4", "#e3eefa")       # blue, as in padded_batch.png
MODEL = ("#2e7d4f", "#e6f4ea")        # green: the model's tokens, trained
ENVIRONMENT = ("#b07d12", "#fff4e0")  # amber: the tool's tokens, context only
TEXT = "#333333"
MUTED = "#777777"

P1, R1, T1, R2 = 3.0, 2.6, 2.6, 2.6  # widths of the four pieces
label_width = 2.6
row_height = 0.9

figure, axes = plt.subplots(figsize=(14, 6.4), dpi=100)
axes.set_xlim(0, label_width + P1 + R1 + T1 + R2 + 2.8)
axes.set_ylim(-0.8, 6.5)
axes.axis("off")


def piece(x, y, width, edge, fill, text):
    axes.add_patch(FancyBboxPatch((label_width + x + 0.05, y), width - 0.1, row_height,
                                  boxstyle="round,pad=0,rounding_size=0.1", linewidth=2.2, edgecolor=edge, facecolor=fill))
    axes.text(label_width + x + width / 2, y + row_height / 2, text, ha="center", va="center", fontsize=13,
              fontweight="bold", color=edge)


def bracket(start, end, y, text, color=MUTED):
    # A thin bracket under a row: which part is the prompt, and which the response
    left, right = label_width + start + 0.1, label_width + end - 0.1
    axes.plot([left, left, right, right], [y + 0.12, y, y, y + 0.12], color=color, linewidth=1.5)
    axes.text((left + right) / 2, y - 0.25, text, ha="center", va="center", fontsize=12, color=color)


def row_label(y, text):
    axes.text(label_width - 0.3, y + row_height / 2, text, ha="right", va="center", fontsize=14, color=TEXT)


# call 1: prompt P1, response R1
y1 = 4.6
row_label(y1, "call 1")
piece(0, y1, P1, *PROMPT, "P1")
piece(P1, y1, R1, *MODEL, "R1")
bracket(0, P1, y1 - 0.15, "prompt")
bracket(P1, P1 + R1, y1 - 0.15, "response")

# call 2: prompt P1 + R1 + T1, response R2. R1 is now part of the prompt, so it has a prompt (blue) outline
y2 = 2.4
row_label(y2, "call 2")
piece(0, y2, P1, *PROMPT, "P1")
piece(P1, y2, R1, PROMPT[0], MODEL[1], "R1")
piece(P1 + R1, y2, T1, *ENVIRONMENT, "T1 (tool)")
piece(P1 + R1 + T1, y2, R2, *MODEL, "R2")
bracket(0, P1 + R1 + T1, y2 - 0.15, "prompt")
bracket(P1 + R1 + T1, P1 + R1 + T1 + R2, y2 - 0.15, "response")

# Prefix continuity: call 2's prompt starts with exactly call 1's prompt and response
axes.add_patch(Rectangle((label_width - 0.05, y2 - 0.08), P1 + R1 + 0.1, y1 + row_height - y2 + 0.16, fill=False,
                         linestyle="--", linewidth=1.8, edgecolor=MUTED))
axes.text(label_width + (P1 + R1) / 2, y1 + row_height + 0.4, "the same tokens: call 2's prompt starts with\n"
          "exactly call 1's prompt + response", ha="center", va="center", fontsize=12.5, color=TEXT, style="italic")

# merged row: P1 as the prompt, then R1, T1 and R2 as one response
y3 = 0.2
row_label(y3, "merged row")
piece(0, y3, P1, *PROMPT, "P1")
piece(P1, y3, R1, *MODEL, "R1")
piece(P1 + R1, y3, T1, *ENVIRONMENT, "T1")
piece(P1 + R1 + T1, y3, R2, *MODEL, "R2")
bracket(0, P1, y3 - 0.15, "prompt")
for start, width, mask, colors in [(P1, R1, 1, MODEL), (P1 + R1, T1, 0, ENVIRONMENT), (P1 + R1 + T1, R2, 1, MODEL)]:
    axes.text(label_width + start + width / 2, y3 - 0.4, f"mask {mask}", ha="center", va="center", fontsize=12.5,
              color=colors[0], family="monospace")
axes.text(label_width + P1 + R1 + T1 + R2 + 0.3, y3 + row_height / 2, "one row:\nP1 is processed\nonce, not twice",
          ha="left", va="center", fontsize=12.5, color=TEXT, style="italic")

figure.savefig(Path(__file__).with_name("prefix_continuity.png"), bbox_inches="tight", facecolor="white")
