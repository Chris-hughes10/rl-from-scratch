# TODO

## Article 1 (`tutorial/series/grpo_manual.ipynb`): publish

The notebook is finished. The full checklist is in [docs/publishing.md](docs/publishing.md). Remaining steps:

- [ ] Rename the GitHub repo `al-explore` → `rl-from-scratch`, update the local remote, and make it public
      (publishing.md §0).
- [ ] Review the website draft (`myblog/posts/grpo-from-scratch/`, re-synced 2026-11-26), then publish it (§2).
- [ ] Medium copy (§3).

## Article 2 (`rl_systems.ipynb`) and article 3 (`rl_agents.ipynb`)

Both run end to end with saved outputs. They have been rebuilt around "break it on purpose" experiments, but the prose
still needs the same editing pass that article 1 had. Before publishing each one:

- [ ] Edit the text.
- [ ] Re-run the notebook and check the numbers quoted in the prose.
- [ ] Tag the code cells for the Medium export.
- [ ] Follow docs/publishing.md.

## Appendices

`appendix_llm_judge.ipynb` (judge vs exact check on Calc-X rollouts) and `appendix_judge_reward.ipynb` (exact check vs
a Foundry judge as the reward). Decide whether they are published as one post, two, or kept in the repo only.

## Other

- Transformers is 4.57.6, and vLLM 0.24 will drop v4 support. Test v5 (≥ 5.5.1) with verl 0.8.0 in a separate venv
  before upgrading `scripts/setup_verl.sh`.
