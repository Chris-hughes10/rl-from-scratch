# TODO

## Publish notebook 1 (`tutorial/series/grpo_manual.ipynb`): website first, then Medium

Do this once the notebook's edits are finished. The site (https://chris-hughes10.github.io/, built with Quarto 1.11)
is the original. Medium is a copy for distribution, with its canonical link pointing back to the site.

**Still needed from Chris:** the public GitHub URL of the notebook. The draft post uses the placeholder
`https://github.com/Chris-hughes10/REPO-TBC`; re-sync with `--code-url` once the repo is live. (No headline image:
not needed for a technical post.)

### 1. Finish the notebook (done 2026-11-16: outputs are from a full run, accuracy 0.765 → 0.925)

- [x] **Re-run top to bottom** (about 15 min on one H100) so every output is fresh and has an execution count.
- [x] **Check every number quoted in the prose against the new outputs.** Training results vary from run to run (bf16 non-determinism), so these will drift:
  - §1.5: the average change in log π (−1.47 / +0.73).
  - §1.7: the bf16 mismatch (0.42 over 107 tokens, 0.15 max). The activation memory (45–56 GiB vs 6–8 GiB), and the per-micro-batch weighting (×1.26 / ×0.75).
  - §1.8:
    - "Each step takes between 3 and 8 seconds… about 5 to 6 minutes".
    - Reward "from around 0.6 to around 0.9".
    - Length "from about 150 tokens to about 135".
    - Held-out accuracy "0.765 to 0.860".
    - "more than half of each step's samples are wasted".
  - §1.9: the ratios "typically range from about 0.5 to 1.5", with "usually well under 1%" outside [0.8, 1.28]. The 3-seed experiment is read from `results/reuse_experiment.jsonl`, not re-run.
  - §2: the timings "update about 1 s, π_old about 0.1 s, generation about 6 s".
- [x] **Verify the citations added from memory:** Dr. GRPO (Liu et al., 2025, arXiv 2503.20783) and DAPO (Yu et al., 2025, arXiv 2503.14476).
- [x] **Add cross-links to earlier posts:**
  - §1.9, where GRPO borrows PPO's ratio and clipping: *Understanding PPO* (https://medium.com/@chris.p.hughes10/understanding-ppo-a-game-changer-in-ai-decision-making-explained-for-rl-newcomers-913a0bc98d2b) and the simple-ppo repo (https://github.com/Chris-hughes10/simple-ppo). Possibly also a line near the GRPO intro in §1.5.
  - §1.7's bf16 explanation: *How Computers Store Numbers* (https://medium.com/@chris.p.hughes10/how-computers-store-numbers-foundations-of-floating-point-for-ai-engineers-1259f39f9f1b).
- [x] **Check any remaining references to notebook 2 or "the series"** (the "What we built" link was removed on 2026-11-13), so the article stands alone.
- [x] **Check the `medium-*` tags on new code cells** (see [tools/notebook_to_medium.md](tools/notebook_to_medium.md) §1). Rule: cells that define a function or class get `medium-code`, everything else `medium-screenshot`. Display-only helpers are the exception, e.g. the appendix's step 2 table.
- [x] **Run `.venv/bin/python tools/check_series.py`.** The only expected failure is `MISMATCH rl_core.py:rl_step`.

### 2. Website post (Quarto)

The draft is at `../Chris-hughes10.github.io/myblog/posts/grpo-from-scratch/` (created 2026-11-16, not yet committed in
the website repo). `tools/notebook_to_quarto.py` builds it from the notebook:

- it copies the notebook to `index.ipynb`, minus the title and subtitle lines (Quarto shows them from the front matter), and minus warnings (stderr, and vLLM's log lines)
- it adds a "Run it yourself" line linking to the code
- it copies the 4 images
- it writes `_metadata.yml` with the front matter, but only if that file doesn't exist yet, so edits made there survive re-syncing

`.ipynb` posts render from their saved outputs, with no execution and no Python needed (checked with Quarto 1.11.9, the
site's version, which is what the site's CI uses).

- [x] **Create the draft** with `python3 tools/notebook_to_quarto.py tutorial/series/grpo_manual.ipynb --slug grpo-from-scratch`. It has `draft: true`, so with the site's `draft-mode: unlinked` it renders but isn't listed or linked.
- [ ] **After any further edits to the notebook, re-run the same command.** It overwrites `index.ipynb` and the images, and keeps `_metadata.yml`.
- [ ] **Review the draft:** `cd myblog && quarto preview`, then open `/posts/grpo-from-scratch/`. In `_metadata.yml`, check the description, the date and `reading-time: 80`. The reading time is an estimate from 13.5k words of prose plus code.
- [ ] **Once the repo is public,** re-sync with `--code-url <URL>`.
- [ ] **To publish:** set the date in `_metadata.yml`, delete the `draft: true` line, then commit and push the website repo. CI renders and deploys it. Check that it appears under Technical Blog Posts, and **note the post's URL** (https://chris-hughes10.github.io/posts/grpo-from-scratch/), which step 3 needs for the canonical link.

### 3. Medium copy

- [ ] **Export:** `uv run tools/notebook_to_medium.py tutorial/series/grpo_manual.ipynb --site-url https://chris-hughes10.github.io/posts/grpo-from-scratch/ --code-url <REPO URL>`. `--site-url` adds an italic line under the subtitle: "first published on my website, where the code can be copied and the tables and equations render properly", plus the GitHub link. It writes to `outputs/medium/grpo_manual/`; the full workflow is in [tools/notebook_to_medium.md](tools/notebook_to_medium.md).
- [ ] **Paste the rendered page, not the HTML source.**
  1. Serve the folder with `python3 -m http.server 8765 --bind 127.0.0.1`.
  2. Open `http://localhost:8765/article.html` through VS Code's port forwarding.
  3. Select all, copy, and paste into a new story.
- [ ] **Add the images.** Download `images/` and drag each one onto its `[IMAGE nn: …]` placeholder, using `preview.html` to check placement.
- [ ] **Format the title and subtitle in Medium.** The export pastes them as a heading and an italic line. Turn them into Medium's title and subtitle (select each, then T for the title and the smaller t for the subtitle), so they're used in previews and search.
- [ ] **Check the "first published on my website" line** under the subtitle has the real repo URL, not the `REPO-TBC` placeholder. Fix any relative links the export script lists.
- [ ] **In the story's settings:**
  - Under **Advanced settings**, set *Customize canonical link* to the site post's URL.
  - Set the SEO title and description, e.g. "GRPO from scratch in PyTorch, checked against verl".
  - Add up to 5 tags: Reinforcement Learning, LLM, Machine Learning, PyTorch, Deep Learning.
- [ ] **Publish.**

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
