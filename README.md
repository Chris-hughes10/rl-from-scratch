# RL for LLMs, from scratch

A series of articles, each one a runnable Jupyter notebook, about how production reinforcement-learning setups for LLMs
work. Each piece is written in plain PyTorch first, run, and broken on purpose to see why it exists. Then it is
swapped for its library version and checked against what we built. [verl](https://github.com/volcengine/verl) and
[Agent Lightning](https://github.com/microsoft/agent-lightning) are the worked examples, not the subject.

| # | Article | Status |
|---|---|---|
| 1 | **[GRPO from Scratch: Reinforcement Learning for LLMs in Plain PyTorch](tutorial/series/grpo_manual.ipynb)**: sampling, log-probs, group-relative advantages, padded batches and PPO clipping, used to train Qwen2.5-0.5B on arithmetic and checked against verl | Published: [read it on the website](https://chris-hughes10.github.io/posts/grpo-from-scratch/) |
| 2 | [From a Training Loop to a Production RL Setup](tutorial/series/rl_systems.ipynb): inference engine, weight sync, the batch that travels between processes, worker groups, then plain verl | 🚧 Work in progress |
| 3 | [Training Agents with RL](tutorial/series/rl_agents.ipynb): multi-turn tool calls, then an existing agent program (Agent Lightning's Calc-X) trained with verl + Agent Lightning | 🚧 Work in progress |
| – | Appendices: [grading rollouts with an LLM judge](tutorial/series/appendix_llm_judge.ipynb), [an LLM judge as the reward](tutorial/series/appendix_judge_reward.ipynb) | 🚧 Work in progress |

Articles 2 and 3 and the appendices run end to end, but their text is still being written and may change a lot.

Each notebook is saved with its outputs, so you can read it on GitHub without running anything. To run one, you need
**one NVIDIA GPU with about 80 GB of memory**. On an H100, article 1 takes about 15 minutes and the others 5–10
minutes each.

## Setup

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 13.0, and Python 3.12.

```bash
uv sync                                   # Agent Lightning, the Calc-X agent's dependencies, notebook dependencies
bash scripts/setup_verl.sh 0.8.0 cu130    # verl 0.8.0, vLLM, torch cu130, and flash-attn compiled from source (10–30 min)

# Optional: pin the exact package set the notebooks were run with
uv pip install --no-deps -r azureml/calc_x/environment/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

# The Jupyter kernel the notebooks use
.venv/bin/python -m ipykernel install --user --name agl-rl-tutorials --display-name "agl-rl-tutorials (3.12.x)"

# Article 3 only: the Calc-X dataset (~2 MB), from the Google Drive link in Agent Lightning's docs
make calc-x-data
```

**Don't run `uv sync` again after `setup_verl.sh`.** It removes the GPU stack, which isn't in `pyproject.toml`. If
you did, re-run `setup_verl.sh`. If `from vllm import LLM` fails with a protobuf error, run
`uv pip install "protobuf>=6.33.5,<7"`.

There is no prebuilt flash-attn wheel for torch 2.11 + CUDA 13.0, so `setup_verl.sh` compiles it. To reuse a wheel
you built before, run `FLASH_ATTN_WHEEL=path/to/flash_attn-….whl bash scripts/setup_verl.sh 0.8.0 cu130`
(`make flash-attn-wheel exp=calc_x` builds one).

## Running

Open a notebook in `tutorial/series/` with the `agl-rl-tutorials` kernel and run it top to bottom. The notebooks start
and stop their own background services (vLLM, Ray, the Agent Lightning Gateway and Controller) and write their working
files to `/tmp/rl-from-scratch`.

Each article runs on its own: the code an article needs from earlier ones is imported from `tutorial/lib/`.
`tools/check_series.py` checks that those modules still match the notebooks that explain them:

```bash
.venv/bin/python tools/check_series.py
```

## Layout

```
tutorial/series/           the articles (notebooks), their images/ and saved results/ from longer runs
tutorial/lib/              code shared between articles, kept in sync with the notebooks by tools/check_series.py
tutorial/generated/        files the notebooks write with %%writefile (git-ignored)
examples/calc_x/           the Calc-X agent and training entrypoint, copied from Agent Lightning v1.0.2
scripts/setup_verl.sh      installs the verl GPU stack
tools/                     the series check, scripts that produce the saved results, and the publishing converters
debug/, .vscode/           article 3's components in separate debuggable processes, without Ray (VS Code launch configs)
azureml/, Makefile         full Calc-X training as an Azure ML job or in Docker (see azureml/README.md)
run_calc_x_local.sh        full Calc-X training on this machine (a few hours on one GPU)
docs/publishing.md         how an article goes from notebook to website and Medium
```

## License

MIT ([LICENSE](LICENSE)). `examples/calc_x/` and `scripts/setup_verl.sh` come from Agent Lightning and are under its
MIT license ([LICENSE-agent-lightning](LICENSE-agent-lightning)).
