"""Draw images/gateway.png, how our gateway records one rollout, in notebook 3 §3. Run: python make_gateway.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AGENT = ("#b07d12", "#fff4e0")    # amber
GATEWAY = ("#574a8c", "#ede8f6")  # purple
VLLM = ("#2e6da4", "#e3eefa")     # blue, as the rollout engine elsewhere
RECORD = ("#2e7d4f", "#e6f4ea")   # green: what we keep for training
TEXT = "#333333"

columns = {"agent": 2.0, "gateway": 8.5, "vllm": 15.0}
figure, axes = plt.subplots(figsize=(15, 7.2), dpi=100)
axes.set_xlim(-0.6, 17.6)
axes.set_ylim(-0.9, 8.6)
axes.axis("off")


def header(x, colors, title, subtitle):
    edge, fill = colors
    axes.add_patch(FancyBboxPatch((x - 1.9, 7.3), 3.8, 1.1, boxstyle="round,pad=0,rounding_size=0.15",
                                  linewidth=2.5, edgecolor=edge, facecolor=fill))
    axes.text(x, 8.02, title, ha="center", va="center", fontsize=15, fontweight="bold", color=edge)
    axes.text(x, 7.6, subtitle, ha="center", va="center", fontsize=12, color=TEXT)
    axes.plot([x, x], [-0.7, 7.3], color=edge, linewidth=1.5, linestyle="--")  # lifeline


def message(start, end, y, label, dashed=False):
    axes.add_patch(FancyArrowPatch((start, y), (end, y), arrowstyle="-|>", mutation_scale=18, linewidth=2,
                                   color=TEXT, linestyle="--" if dashed else "-"))
    axes.text((start + end) / 2, y + 0.12, label, ha="center", va="bottom", fontsize=11.5, color=TEXT,
              family="monospace", linespacing=1.4)


def note(x, y, text, colors, width=4.6):
    edge, fill = colors
    axes.add_patch(FancyBboxPatch((x - width / 2, y - 0.45), width, 0.9, boxstyle="round,pad=0,rounding_size=0.1",
                                  linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=3))
    axes.text(x, y, text, ha="center", va="center", fontsize=11.5, color=edge, family="monospace", linespacing=1.4,
              zorder=4)


header(columns["agent"], AGENT, "agent", "rollout r42")
header(columns["gateway"], GATEWAY, "our gateway", "one URL per rollout")
header(columns["vllm"], VLLM, "vLLM server", "the policy")

agent, gateway, vllm = columns["agent"], columns["gateway"], columns["vllm"]
message(agent, gateway, 6.5, "POST /rollout/r42/v1/chat/completions")
message(gateway, vllm, 5.35, 'forward, plus "also return\ntoken ids and log-probs"')
message(vllm, gateway, 4.2, "response + token ids\n+ log-probs", dashed=True)
note(gateway, 3.35, 'record under "r42":\nprompt_ids, response_ids, log_probs', RECORD, width=6.4)
message(gateway, agent, 1.95, "normal OpenAI response\n(text + tool_calls)", dashed=True)
axes.text((agent + gateway) / 2, 1.55, "the extra fields are simply ignored", ha="center", va="center", fontsize=11.5,
          color=TEXT, style="italic")
message(agent, gateway, 0.65, 'POST /rollout/r42/events {"reward": 1}')
note(gateway, -0.25, 'RECORDS["r42"]["reward"] = 1.0', RECORD, width=6.4)

figure.savefig(Path(__file__).with_name("gateway.png"), bbox_inches="tight", facecolor="white")
