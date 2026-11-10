# TODO

## Split `rl_training_from_scratch.ipynb` into a series

The notebook (205 cells) teaches three things for three audiences: RL for LLMs (§1), the systems
behind a trainer (§2–§6), and training real agents (§7–§9). Keep the "write it, then swap in the
library" spine, but split it into notebooks that each run on their own:

| # | Notebook | From the current notebook | Needs |
|---|---|---|---|
| 1 | **GRPO from scratch** | §1, §2 up to §2.3 (vLLM, weight sync, trainer-vs-engine mismatch) | one GPU, a few minutes |
| 2 | **From our trainer to verl** | §3–§6, §9.1 | one GPU, Ray |
| 3 | **Training real agents** | §7, §8.1–§8.3, §9.2 | vLLM server, Agent Lightning services |
| 4 | **Rewards: exact checks and LLM judges** | §1.1's reward discussion, §8.4 | a judge (Foundry deployment, or the local 7B: ~15 GB) |
| 5 | **Calc-X end to end** (the current `rl_training_walkthrough.ipynb`, trimmed) | see below | the full stack |

- **The judges are core content, not an appendix.** Notebook 4 generates its own rollouts to
  grade at the start, so it runs without notebook 3:
  - Start a vLLM server with `Qwen2.5-1.5B-Instruct` (same flags as §8.1) and run the Calc-X agent
    on about 32 questions × 4 attempts = 128 rollouts, the same number §8.4 grades now. With the
    untrained model, roughly 13% pass the exact check, enough for disagreements to show up.
  - The agent can't talk to vLLM directly: it asks for model `"auto"`, must post its reward to
    `AGL_EVENT_URL` (it raises if that fails), and its final answer is only visible in the HTTP
    traffic. §8.1's 60-line recording gateway handles all three. Move it, with `launch_agent` and
    `run_agents`, into `tutorial/generated/recording_gateway.py`, so notebooks 3 and 4 import the
    same code; notebook 3 still explains it line by line.
  - Grade with the agent's last model call (its final answer) and the reward it posted, exactly as
    §8.4 does with `RECORDS` today.
  - Stop the policy's vLLM server before starting a local 7B judge, as now; with a Foundry judge
    nothing else is needed on the GPU.
  - Keep the rule that **the judge never goes through the gateway** (§8.4 explains why); with the
    gateway imported from a module, say so where it's set up.
- Notebooks 2–4 need §1's code without re-running §1: write the trainer pieces
  (`make_padded_batch`, `select_rows`, `compute_log_probs`, `grpo_advantages`, `ppo_clip_loss`,
  `ppo_update`, `rl_step`) to `tutorial/generated/rl_core.py` with `%%writefile`, as
  `arith_task.py` already is.
- Each notebook opens with a short "where we are" recap and the prerequisites it assumes.
- Update `tools/cpu_check.py` (it finds cells by prefix in one notebook) and the README's notebook list.

## Where `rl_training_walkthrough.ipynb` (Part 2) fits

It teaches the same stack **top-down**: it starts from verl + Agent Lightning on Calc-X, plays the
agent by hand through the Gateway's REST API, and ends with a real training run. Most of its middle
duplicates the from-scratch notebook:

| Walkthrough | Duplicates |
|---|---|
| §2 policy, `response_log_probs`, scoring responses | from-scratch §1.2–§1.4 |
| §6 `DataProto`, `RolloutAdapter`, trajectory vs transition | §3, §8.3 |
| §7 advantages, Dr. GRPO, zero-signal groups | §1.5 |
| §8 PPO-clip by hand, training/inference mismatch | §1.5, §2.3 |
| §9 single controller over Ray | §5–§6 |

Unique to it, and worth keeping: §4 "be the agent" (one rollout driven by hand through the Gateway's
API), §5.3 GRPO groups on real agent rollouts, §7.4 per-rollout loss normalisation
(`normalize_advantages_by_rollout`), §9.3 the config tree (`build_config`), §10 a longer real
training run and how to read its metrics.

Plan: make it notebook 5, cut the duplicated sections down to a line and a link to where the
series explains them, and keep the unique parts.

**Fix regardless of the split:** its §7 Dr. GRPO text (cells 66 and 68) says std normalisation gives
a 1-of-4 and a 3-of-4 success "the same magnitude". Both groups have std 0.5, so they are scaled
identically, and within a group std changes nothing. What std does change is the weighting *across*
groups with different std (lopsided groups such as 1-of-8 are scaled up about 2.8×, an even
4-of-8 group about 1.9×). See the corrected explanation in the from-scratch notebook's §1.5.

## Other follow-ups

- `azureml/verl_from_scratch/` and `azureml/README.md` still point at `tutorial/verl_from_scratch.ipynb`,
  which no longer exists; point them at the current notebook(s).
- Possible §1.3 addition: measure log-probs of a left-padded row with and without `position_ids`
  (correct: identical to unpadded; shifted by 15: mean |Δ| ≈ 0.016, about 4× the trainer-vs-vLLM
  mismatch from §2.3).
- Re-run the from-scratch notebook top to bottom so the new explanation cells get execution counts
  and the training outputs are fresh. Check the prose that quotes numbers (e.g. §8.4's 22% vs 13%)
  against the new outputs.
- Transformers is 4.57.6; vLLM 0.24 will drop v4 support. Test v5 (≥ 5.5.1) with verl 0.8.0 in a
  separate venv before upgrading `scripts/setup_verl.sh`.
