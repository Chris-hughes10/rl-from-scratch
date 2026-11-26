"""Draw images/multi_turn_sequence.png, one multi-turn episode as a single sequence, in notebook 3 §1.
Run: python make_multi_turn.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROMPT = ("#2e6da4", "#e3eefa")       # blue, as in padded_batch.png
MODEL = ("#2e7d4f", "#e6f4ea")        # green: the model's tokens, trained (as in show_episode)
ENVIRONMENT = ("#b07d12", "#fff4e0")  # amber: the tool's tokens, context only (as in show_episode)
TEXT = "#333333"

# (width, colours, title, content, response_mask)
segments = [
    (2.6, PROMPT, "prompt", "", None),
    (4.6, MODEL, "model", "<tool_call>…</tool_call>", 1),
    (5.8, ENVIRONMENT, "environment (the tool)", "<tool_response>…</tool_response>", 0),
    (3.8, MODEL, "model", "… \\boxed{N}", 1),
]

figure, axes = plt.subplots(figsize=(15, 3.4), dpi=100)
axes.set_xlim(-0.1, sum(width for width, *_ in segments) + 0.4)
axes.set_ylim(-1.6, 2.1)
axes.axis("off")

x = 0.0
for width, (edge, fill), title, content, mask in segments:
    axes.add_patch(FancyBboxPatch((x + 0.05, 0), width - 0.1, 1.3, boxstyle="round,pad=0,rounding_size=0.12",
                                  linewidth=2.2, edgecolor=edge, facecolor=fill))
    axes.text(x + width / 2, 0.88 if content else 0.65, title, ha="center", va="center", fontsize=13.5,
              fontweight="bold", color=edge)
    if content:
        axes.text(x + width / 2, 0.4, content, ha="center", va="center", fontsize=12, color=edge, family="monospace")
    if mask is not None:
        axes.text(x + width / 2, -0.45, f"response_mask = {mask}", ha="center", va="center", fontsize=13,
                  color=edge, family="monospace")
        axes.text(x + width / 2, -1.0, "trained" if mask else "context only: not trained", ha="center", va="center",
                  fontsize=12.5, color=edge, style="italic")
    x += width

# The response: everything after the prompt
response_start = segments[0][0]
axes.annotate("", xy=(response_start + 0.05, 1.65), xytext=(x - 0.05, 1.65),
              arrowprops={"arrowstyle": "<->", "color": TEXT, "linewidth": 1.8})
axes.text((response_start + x) / 2, 1.9, "the response: one sequence, generated over several turns", ha="center",
          va="center", fontsize=13, color=TEXT)

figure.savefig(Path(__file__).with_name("multi_turn_sequence.png"), bbox_inches="tight", facecolor="white")
