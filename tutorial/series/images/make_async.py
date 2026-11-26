"""Draw images/async_timeline.png, synchronous vs one-step-off asynchronous training in notebook 2 §3.
Run: python make_async.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

GENERATE = ("#2e6da4", "#e3eefa")  # blue, as SAMPLE in the notebook 1 loop diagram
UPDATE = ("#a83a3a", "#fbe7e7")    # red, as UPDATE in the notebook 1 loop diagram
IDLE = ("#9a9a9a", "#f7f7f7")
TEXT = "#333333"

GEN_TIME, UPDATE_TIME = 2.0, 1.5
lane_height, lane_gap = 0.8, 0.25
label_width = 2.2

# (lane, start, end, kind, label); lanes: 0 = rollout, 1 = training
synchronous, asynchronous = [], []
time = 0.0
for step in range(1, 4):
    synchronous.append((0, time, time + GEN_TIME, "gen", f"gen {step}"))
    synchronous.append((1, time, time + GEN_TIME, "idle", ""))
    synchronous.append((1, time + GEN_TIME, time + GEN_TIME + UPDATE_TIME, "update", f"update {step}"))
    synchronous.append((0, time + GEN_TIME, time + GEN_TIME + UPDATE_TIME, "idle", ""))
    time += GEN_TIME + UPDATE_TIME
sync_end = time

for step in range(1, 5):
    asynchronous.append((0, (step - 1) * GEN_TIME, step * GEN_TIME, "gen", f"gen {step}"))
asynchronous.append((1, 0, GEN_TIME, "idle", ""))
for step in range(1, 4):
    start = step * GEN_TIME  # update k starts as soon as batch k is generated
    asynchronous.append((1, start, start + UPDATE_TIME, "update", f"update {step}"))
    asynchronous.append((1, start + UPDATE_TIME, start + GEN_TIME, "idle", ""))
async_end = 3 * GEN_TIME + UPDATE_TIME

figure, axes = plt.subplots(figsize=(13, 6.2), dpi=100)
axes.set_xlim(-0.2, label_width + sync_end + 0.4)
axes.set_ylim(-0.6, 6.4)
axes.axis("off")


def draw_timeline(blocks, base_y, title, finish_time):
    axes.text(0, base_y + 2 * (lane_height + lane_gap) + 0.25, title, ha="left", va="center", fontsize=15, fontweight="bold", color=TEXT)
    for lane, name in enumerate(["rollout", "training"]):
        lane_y = base_y + (1 - lane) * (lane_height + lane_gap)
        axes.text(label_width - 0.25, lane_y + lane_height / 2, name, ha="right", va="center", fontsize=14, color=TEXT)
    for lane, start, end, kind, label in blocks:
        lane_y = base_y + (1 - lane) * (lane_height + lane_gap)
        edge, fill = {"gen": GENERATE, "update": UPDATE, "idle": IDLE}[kind]
        axes.add_patch(FancyBboxPatch((label_width + start + 0.04, lane_y), end - start - 0.08, lane_height,
                                      boxstyle="round,pad=0,rounding_size=0.1", linewidth=2 if kind != "idle" else 1.2,
                                      edgecolor=edge, facecolor=fill, linestyle="-" if kind != "idle" else "--",
                                      hatch="//" if kind == "idle" else None))
        if label:
            axes.text(label_width + (start + end) / 2, lane_y + lane_height / 2, label, ha="center", va="center",
                      fontsize=13, fontweight="bold", color=edge)
    # When the third update finishes
    finish_x = label_width + finish_time
    axes.plot([finish_x, finish_x], [base_y - 0.15, base_y + 2 * lane_height + lane_gap + 0.1], linestyle=":", linewidth=2, color=TEXT)
    axes.text(finish_x + 0.08, base_y - 0.3, "3 updates done", ha="left", va="center", fontsize=12, color=TEXT)


draw_timeline(synchronous, 3.6, "synchronous", sync_end)
draw_timeline(asynchronous, 0.4, "one-step-off asynchronous", async_end)

# Staleness: batch 2 is generated while update 1 runs, so it uses the weights from before update 1
gen2_x = label_width + 1.5 * GEN_TIME
async_rollout_top = 0.4 + lane_height + lane_gap + lane_height
note_x = label_width + 4 * GEN_TIME + 0.35
axes.add_patch(FancyArrowPatch((note_x, async_rollout_top - 0.1), (gen2_x, async_rollout_top + 0.05),
                               arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color=TEXT, connectionstyle="arc3,rad=0.35"))
axes.text(note_x + 0.1, async_rollout_top - 0.45, "batch 2 is generated with the\nweights from before update 1",
          ha="left", va="center", fontsize=12, color=TEXT, style="italic")

# Legend for the idle blocks
axes.add_patch(FancyBboxPatch((label_width + sync_end - 1.6, 6.05), 0.5, 0.3, boxstyle="round,pad=0,rounding_size=0.05",
                              linewidth=1.2, edgecolor=IDLE[0], facecolor=IDLE[1], linestyle="--", hatch="//"))
axes.text(label_width + sync_end - 1.0, 6.2, "idle", ha="left", va="center", fontsize=12, color=TEXT)

figure.savefig(Path(__file__).with_name("async_timeline.png"), bbox_inches="tight", facecolor="white")
