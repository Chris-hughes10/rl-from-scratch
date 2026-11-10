# RL training for LLM agents: verl + Agent Lightning, hands on

Notebooks that take apart how [verl](https://github.com/volcengine/verl) and
[Agent Lightning](https://github.com/microsoft/agent-lightning) train an LLM agent with reinforcement learning, using
Agent Lightning's Calc-X example (an AutoGen agent with an MCP calculator tool).

Two notebooks, meant to be read in order:

1. **[Part 1: rl_training_from_scratch.ipynb](tutorial/rl_training_from_scratch.ipynb)** — bottom-up. GRPO built in
   plain PyTorch (log-probs, advantages, PPO-clip, and a check that an update did what it should), then vLLM,
   `DataProto`, verl's algorithm functions, Ray, worker groups, a multi-turn tool loop, and a token-recording gateway
   for a real agent, each swapped for its library version (verl, then Agent Lightning) and checked against what we
   built, ending with plain verl and verl + Agent Lightning end to end.
2. **[Part 2: rl_training_walkthrough.ipynb](tutorial/rl_training_walkthrough.ipynb)** — top-down. One complete RL step
   done by hand with the real components (vLLM, the Agent Lightning Gateway and Controller, the real agent), from
   rollouts to `DataProto`, GRPO advantages and a PPO update, then a short run of the real trainer.

All of them need one GPU with about 80 GB of memory.

## Layout

```
tutorial/                  the notebooks
examples/calc_x/           Calc-X agent and training entrypoint, copied from Agent Lightning v1.0.2
scripts/setup_verl.sh      installs the verl GPU stack (from Agent Lightning, plus a prebuilt flash-attn option)
debug/rollout_step.py      one rollout + training batch without Ray, for stepping through in a debugger
tools/                     generators for the from-scratch notebooks, and CPU-only checks of their pure-Python cells
azureml/                   AzureML jobs: Calc-X training, and end-to-end execution of the from-scratch notebook
Makefile, config.env       AzureML / Docker commands (`make help`)
run_calc_x_local.sh        full Calc-X training run on this machine, without W&B
```

`agentlightning` itself is installed from PyPI (`agentlightning==1.0.2`). The agent doesn't need it at all: it only
reads its OpenAI base URL and reward endpoint from the environment.

## Setup

Requires [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 13.0, and Python 3.12.

```bash
uv sync                                   # agentlightning, the Calc-X agent's deps, notebook deps
bash scripts/setup_verl.sh 0.8.0 cu130    # verl 0.8.0, vLLM 0.22, torch cu130, and flash-attn compiled from source (10-30 min)

# Optional: pin to the exact package set the notebooks were verified with (also used by the AzureML image)
uv pip install --no-deps -r azureml/calc_x/environment/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

# Jupyter kernel the notebooks expect
.venv/bin/python -m ipykernel install --user --name agl-rl-tutorials --display-name "Python (agl-rl-tutorials .venv)"

# Calc-X dataset (train/test/test_mini parquet, ~2 MB) from the Google Drive link in Agent Lightning's docs
make calc-x-data
```

Don't re-run `uv sync` after `setup_verl.sh`: it would remove the GPU stack, which isn't in `pyproject.toml`.

**flash-attn.** There is no prebuilt flash-attn wheel for torch 2.11 + CUDA 13.0, so `setup_verl.sh` compiles it.
That's all a local setup needs. The AzureML image can't compile it (too slow on the image builder), so it installs a
wheel from `azureml/calc_x/environment/wheels/` (gitignored, 113 MB). Build that once, after the setup above, with
`make flash-attn-wheel exp=calc_x` (about 10 min on 40 cores). Once the wheel exists, a new environment can skip the
compile: `FLASH_ATTN_WHEEL=azureml/calc_x/environment/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl bash scripts/setup_verl.sh 0.8.0 cu130`.

## Running

Open a notebook with the "Python (agl-rl-tutorials .venv)" kernel and run it top to bottom. The notebooks start and
stop their own background services (vLLM, the Gateway, the Controller).

Full Calc-X training (a few hours on one GPU): `./run_calc_x_local.sh`. AzureML: see
[azureml/README.md](azureml/README.md).

After editing the from-scratch notebook, check its pure-Python parts without a GPU:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/cpu_check.py
```

## License

`examples/calc_x/` and `scripts/setup_verl.sh` come from Agent Lightning and are under its MIT license
([LICENSE-agent-lightning](LICENSE-agent-lightning)).
