"""Draw images/grpo_loop.png, the GRPO loop diagram in notebook 1. Run: python make_grpo_loop.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ARROW_GREY = "#444444"
POLICY_PURPLE = "#574a8c"

figure, axes = plt.subplots(figsize=(16, 9.2), dpi=100)
axes.set_xlim(0, 16)
axes.set_ylim(0, 9.2)
axes.axis("off")


def box(x, y, width, height, title, subtitle, edge, fill, rounding=0.18):
    axes.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                  linewidth=3, edgecolor=edge, facecolor=fill))
    axes.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=21, fontweight="bold", color=edge)
    axes.text(x + width / 2, y + height * 0.30, subtitle, ha="center", va="center", fontsize=15, color="#333333", linespacing=1.4)


def arrow(start, end, color=ARROW_GREY):
    axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=24, linewidth=3, color=color, shrinkA=0, shrinkB=0))


def label(x, y, text, color="#555555", ha="center"):
    axes.text(x, y, text, ha=ha, va="center", fontsize=15, color=color)


# The four steps
box(2.3, 6.5, 5.5, 1.7, "1. SAMPLE", "n attempts per question", "#2e6da4", "#e3eefa")
box(10.1, 6.5, 5.5, 1.7, "2. REWARD", "is each attempt correct?\n1 if yes, 0 if no", "#2e7d4f", "#e6f4ea")
box(10.1, 0.8, 5.5, 1.7, "3. ADVANTAGE", "better or worse than the\naverage for this question?", "#b07d12", "#fff4e0")
box(2.3, 0.8, 5.5, 1.7, "4. UPDATE", "advantage > 0: tokens more likely\nadvantage < 0: tokens less likely", "#a83a3a", "#fbe7e7")

# The policy, used by SAMPLE (to generate) and by UPDATE (to score, then to change)
policy_x, policy_y, policy_width, policy_height = 3.45, 3.95, 3.2, 1.3
axes.add_patch(FancyBboxPatch((policy_x, policy_y), policy_width, policy_height, boxstyle="round,pad=0,rounding_size=0.5",
                              linewidth=3, edgecolor=POLICY_PURPLE, facecolor="#ede8f6"))
axes.text(policy_x + policy_width / 2, policy_y + 0.86, "policy  π$_θ$", ha="center", va="center", fontsize=22,
          fontweight="bold", color=POLICY_PURPLE)
axes.text(policy_x + policy_width / 2, policy_y + 0.36, "(the LLM weights θ)", ha="center", va="center", fontsize=15,
          color=POLICY_PURPLE)

left_lane, right_lane = 4.45, 5.65  # x positions of the two arrows in each policy link
policy_top, policy_bottom = policy_y + policy_height, policy_y

# Each step's use of the policy is numbered (i), (ii) in order; in both pairs (i) is the left arrow.
# SAMPLE ⇄ policy
arrow((left_lane, 6.5), (left_lane, policy_top + 0.05), POLICY_PURPLE)
label(left_lane - 0.2, (6.5 + policy_top) / 2, "1(i) prompt", POLICY_PURPLE, ha="right")
arrow((right_lane, policy_top), (right_lane, 6.45), POLICY_PURPLE)
label(right_lane + 0.2, (6.5 + policy_top) / 2, "1(ii) n attempts", POLICY_PURPLE, ha="left")

# UPDATE ⇄ policy
arrow((left_lane, policy_bottom), (left_lane, 2.55), POLICY_PURPLE)
label(left_lane - 0.2, (2.5 + policy_bottom) / 2, "4(i) how likely is\neach token?", POLICY_PURPLE, ha="right")
arrow((right_lane, 2.5), (right_lane, policy_bottom - 0.05), POLICY_PURPLE)
label(right_lane + 0.2, (2.5 + policy_bottom) / 2, "4(ii) adjust θ", POLICY_PURPLE, ha="left")

# The flow between the steps
axes.text(1.85, 7.35, "question", ha="right", va="center", fontsize=19, style="italic", color="#222222")
arrow((1.95, 7.35), (2.25, 7.35))
arrow((7.8, 7.35), (10.05, 7.35))
label(8.95, 7.75, "n attempts")
arrow((12.85, 6.5), (12.85, 2.55))
label(13.0, 4.6, "n rewards", ha="left")
arrow((10.1, 1.65), (7.85, 1.65))
label(8.95, 2.05, "n advantages")

figure.savefig(Path(__file__).with_name("grpo_loop.png"), bbox_inches="tight", facecolor="white")
