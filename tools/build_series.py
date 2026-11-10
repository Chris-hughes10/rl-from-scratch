"""Build the tutorial series (tutorial/series/*.ipynb) and its shared modules (tutorial/lib/) from the two
source notebooks, which are only read:

    tutorial/rl_training_from_scratch.ipynb  ->  notebooks 1-4, and lib/arith_task.py, rl_core.py, verl_core.py,
                                                 recording_gateway.py (functions copied verbatim from its cells)
    tutorial/rl_training_walkthrough.ipynb   ->  notebook 5

Usage: .venv/bin/python tools/build_series.py
"""

import ast
import copy
import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = json.load(open(REPO / "tutorial/rl_training_from_scratch.ipynb"))
WALK = json.load(open(REPO / "tutorial/rl_training_walkthrough.ipynb"))
OUT = REPO / "tutorial/series"
LIB = REPO / "tutorial/lib"


# ── cell helpers ─────────────────────────────────────────────────────────────
def text(nb, i: int) -> str:
    return "".join(nb["cells"][i]["source"])


def take(nb, i: int, *patches: tuple[str, str], expect: str | None = None) -> dict:
    """Copy cell i (outputs dropped), applying exact (old, new) text patches."""
    cell = copy.deepcopy(nb["cells"][i])
    src = text(nb, i)
    if expect is not None:
        assert src.startswith(expect), (i, expect, src[:60])
    for old, new in patches:
        assert src.count(old) == 1, (i, src.count(old), old[:70])
        src = src.replace(old, new)
    cell["source"] = src.splitlines(keepends=True)
    cell["id"] = uuid.uuid4().hex[:8]
    if cell["cell_type"] == "code":
        cell["outputs"], cell["execution_count"] = [], None
    return cell


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def save(name: str, cells: list[dict]) -> None:
    nb = {"cells": cells, "metadata": copy.deepcopy(SRC["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote tutorial/series/{name}: {len(cells)} cells")


def find(nb, prefix: str) -> int:
    hits = [i for i in range(len(nb["cells"])) if text(nb, i).startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


# ── extracting top-level definitions verbatim, for the shared modules ────────
def top_level(src: str, names: list[str]) -> str:
    """The source of the named top-level functions/classes/assignments in `src`, decorators included."""
    tree = ast.parse(src)
    lines = src.splitlines()
    found = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            found[name] = "\n".join(lines[start - 1 : node.end_lineno])
    missing = [n for n in names if n not in found]
    assert not missing, missing
    return "\n\n\n".join(found[n] for n in names)


def write_module(name: str, src: str) -> None:
    ast.parse(src)  # must be valid Python
    (LIB / name).write_text(src.rstrip() + "\n")
    print(f"wrote tutorial/lib/{name}")


S = lambda prefix: text(SRC, find(SRC, prefix))  # noqa: E731  (source of the from-scratch cell starting with prefix)

ARITH = S("%%writefile tutorial/generated/arith_task.py").split("\n", 1)[1]
write_module("arith_task.py", ARITH)

write_module("rl_core.py", '''
"""Notebook 1's trainer as a module, so notebooks 2-4 can import it without re-running notebook 1.

Every function here is copied verbatim from notebook 1 (tools/check_series.py checks that they match);
read them there, where each one is explained step by step.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
from vllm import SamplingParams, TokensPrompt

from arith_task import BOXED, make_problems, reward_fn
from common import DEVICE, MODEL, SHM

TRAIN = make_problems(4000, seed=1)
EVAL = make_problems(200, seed=2)
INSTRUCTION = r"Show brief working, then give the final answer as \\boxed{N}."
LR = 2e-6

hf_logging.set_verbosity_error()

''' + top_level(S("from transformers import AutoModelForCausalLM"), ["tokenizer", "PAD_ID", "STOP_IDS", "chat_prompt_ids", "decode_tokens"])
    + "\n\n\n" + top_level(S("def load_policy("), ["load_policy", "make_optimizer"])
    + "\n\n\n" + top_level(S("def hf_generate("), ["hf_generate", "truncate_at_stop"])
    + "\n\n\n" + top_level(S("def make_padded_batch("), ["make_padded_batch"])
    + "\n\n\n" + top_level(S("def select_rows("), ["select_rows"])
    + "\n\n\n" + top_level(S("def micro_batch_log_probs("), ["micro_batch_log_probs", "compute_log_probs"])
    + "\n\n\n" + top_level(S("def grpo_advantages("), ["grpo_advantages"])
    + "\n\n\n" + top_level(S("def ppo_clip_loss("), ["ppo_clip_loss"])
    + "\n\n\n" + top_level(S("def ppo_update("), ["ppo_update"])
    + "\n\n\n" + top_level(S("def rl_step("), ["rl_step"])
    + "\n\n\n" + top_level(S("def print_step_metrics("), ["print_step_metrics"])
    + "\n\n\n" + top_level(S("def vllm_generate("), ["vllm_generate"])
    + "\n\n\n" + top_level(S("import safetensors.torch"), ["save_for_inference", "push_weights", "evaluate"])
    + "\n\n\n" + top_level(S("def plot_histories("), ["plot_histories"]))

write_module("verl_core.py", '''
"""Notebook 2's verl-flavoured helpers (§4), so notebook 3 can import them. Copied verbatim from notebook 2."""

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, get_policy_loss_fn

from arith_task import reward_fn
from rl_core import decode_tokens, make_padded_batch

''' + top_level(S("from omegaconf import OmegaConf"), ["verl_policy_loss", "verl_ppo_loss"])
    + "\n\n\n" + top_level(S("def build_batch("), ["build_batch", "add_advantages", "compute_batch_metrics"]))

gateway_src = S("import uvicorn")
gateway_head = gateway_src.split("\ngateway_server = uvicorn.Server")[0]
write_module("recording_gateway.py", '''
"""Notebook 3's recording gateway (§8.1) and agent launcher, so notebook 4 can import them.

Copied verbatim from notebook 3, which explains every line. The gateway forwards an agent's chat requests to the
policy's vLLM server and records the exact token IDs and log-probs under the rollout's ID.
A judge model must never be called through it: everything on these URLs is recorded as the policy's own output.
"""

import os
import subprocess
import sys
import threading
import time

import httpx

from common import CALC_X_MODEL, REPO, WORK

VLLM_PORT = 8110
VLLM_ROOT = f"http://127.0.0.1:{VLLM_PORT}"

''' + gateway_head.strip() + '''


def start_gateway() -> "uvicorn.Server":
    """Serve the gateway from a background thread of this process; returns the server (set .should_exit to stop it)."""
    server = uvicorn.Server(uvicorn.Config(gateway, host="127.0.0.1", port=GATEWAY_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server


(WORK / "agents").mkdir(parents=True, exist_ok=True)


''' + top_level(S("import uuid"), ["launch_agent", "run_agents"]))


# ── shared notebook pieces ───────────────────────────────────────────────────
SERIES_TABLE = r'''
| # | Notebook | What it builds | Sections |
|---|---|---|---|
| 1 | [GRPO from scratch](1_grpo_from_scratch.ipynb) | an RL trainer in plain PyTorch, then a real inference engine (vLLM) and weight sync | §0–§2 |
| 2 | [From our trainer to verl](2_from_trainer_to_verl.ipynb) | the same trainer, rebuilt from verl's pieces: `DataProto`, `core_algos`, Ray actors, worker groups; then plain verl | §3–§6 |
| 3 | [Training real agents](3_training_real_agents.ipynb) | multi-turn tool use, recording a real agent at the HTTP boundary, Agent Lightning, and verl + Agent Lightning end to end | §7–§10 |
| 4 | [Rewards: exact checks and LLM judges](4_rewards_and_judges.ipynb) | what a reward measures, and an LLM judge next to an exact check | own numbering |
| 5 | [Calc-X end to end](5_calc_x_end_to_end.ipynb) | one training step driven by hand through Agent Lightning, then a real training run | own numbering |

Notebooks 1–3 share one set of section numbers, so a reference like "§1.5" or "§8.1" always points to the same place, whichever notebook it appears in. Each notebook runs on its own: code from earlier notebooks is imported from `tutorial/lib/`, where it is copied verbatim.
'''.strip()


def setup_cell(extra_imports: str = "") -> dict:
    return code(f'''
import gc
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "examples" / "calc_x").is_dir())
sys.path.insert(0, str(REPO / "tutorial/lib"))  # the series' shared code
from common import (CALC_X_DATA, CALC_X_MODEL, DEVICE, GENERATED, MODEL, SEED, SERVICES, SHM, VENV_BIN, WORK,  # noqa: E402
                    follow_log, gpu_free_gib, log_tail, setup, start_service, stop_service, wait_until_healthy)

setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
{extra_imports}
''')


def section_refs(cell: dict, notebook: int) -> dict:
    """Point references to the judge section (now notebook 4) at that notebook, and drop references to exercises that don't exist."""
    if cell["cell_type"] != "markdown":
        return cell
    src = "".join(cell["source"])
    src = src.replace("in §8.4", "in [notebook 4](4_rewards_and_judges.ipynb)").replace("(§8.4)", "([notebook 4](4_rewards_and_judges.ipynb))")
    for n in (5, 7, 8):
        src = src.replace(f" (exercise {n})", "")
    cell["source"] = src.splitlines(keepends=True)
    return cell


# ═════════════════════════════════════════════════════════════════════════════
# Notebook 1: GRPO from scratch (§0–§2)
# ═════════════════════════════════════════════════════════════════════════════
nb1 = [md(r'''
# RL training for LLM agents — 1. GRPO from scratch

### Building an RL trainer from nothing but PyTorch

Frameworks like [verl](https://github.com/volcengine/verl) make it possible to train LLMs with reinforcement learning (RL) across many GPUs, but they are also large, layered codebases: Ray workers, FSDP, an inference engine with weight sync, agent loops, a few hundred config options. When something goes wrong, or when we want to train an agent that doesn't fit the framework's mould, it is hard to know which of those layers matters, and why it exists at all.

Jeremy Howard's [*What is `torch.nn` really?*](https://pytorch.org/tutorials/beginner/nn_tutorial.html) has a great answer to this kind of problem: train a model with nothing but tensors, then replace one hand-written piece at a time with its library equivalent, until only idiomatic library code is left. After every swap the code still runs, and we know exactly what each library piece does, because we wrote it first. This series does the same for RL training of LLMs:

''' + SERIES_TABLE + r'''

**In this notebook** we build the trainer itself, and then make its slowest part fast:

| § | We write | Then replace it with | The problem that forces the swap |
|---|---|---|---|
| 1 | GRPO in plain PyTorch: sample, score, log-probs, advantages, PPO-clip loss | | |
| 2 | | `model.generate` → **vLLM** | generation is most of the step time; now we need **weight sync** |

**What we need to know already.** PyTorch, and roughly how an LLM is fine-tuned. Everything RL-specific (policies, log-probs, rewards, advantages, PPO) is introduced when we first need it.

**Hardware and time.** One GPU with about 40 GB of free memory, and a few minutes on an H100 (plus a 1 GB model download). The policy is `Qwen2.5-0.5B-Instruct`, small enough to train in minutes.
'''),
    md(r'''
## 0. Setup

Paths, the model name and the random seed live in a small shared module, `tutorial/lib/common.py`, so that every notebook in the series uses the same ones. `setup()` creates the working folders (logs and data go to `/tmp/rl-from-scratch/`; weights are handed from trainer to inference engine through `/dev/shm/rl-from-scratch/`, which lives in RAM, not on disk), seeds the random number generators, and checks that the GPU is free.
'''),
    setup_cell()]
i_writefile = find(SRC, "%%writefile tutorial/generated/arith_task.py")
for i in range(find(SRC, "## 1. GRPO in plain PyTorch"), find(SRC, "## 3. Swap 2")):
    if i == i_writefile:
        continue
    if text(SRC, i).startswith("**Bundling into a module.**"):
        nb1.append(md(r'''
**Bundling into a module.** Later stages (Ray workers, verl's reward workers, the other notebooks in this series) run in other processes and need to import this code. It lives in `tutorial/lib/arith_task.py`, copied verbatim from the three cells above: nothing new.
'''))
        continue
    if text(SRC, i).startswith("import importlib"):
        nb1.append(code(r'''
from arith_task import BOXED, make_problems, reward_fn, safe_eval

# From here on we use the module versions. Quick sanity check:
TRAIN = make_problems(4000, seed=1)
EVAL = make_problems(200, seed=2)
INSTRUCTION = r"Show brief working, then give the final answer as \boxed{N}."

print(f"{len(TRAIN)} training problems, {len(EVAL)} eval problems")
print("first problem:", TRAIN[0])
'''))
        continue
    nb1.append(section_refs(take(SRC, i), 1))
nb1 += [md(r'''
## What we built, and what comes next

| We wrote | It became | Why |
|---|---|---|
| `hf_generate` | vLLM | generation speed: paged KV cache, continuous batching, CUDA graphs |
| (nothing: one copy of the weights) | `save_for_inference` + `reload_weights` | two copies of the weights must be kept in sync after every update |

We now have a complete RL trainer that learns, with a real inference engine. Every function we wrote is also in `tutorial/lib/rl_core.py`, copied verbatim, so the next notebooks can import it instead of re-running this one.

**Next:** [notebook 2](2_from_trainer_to_verl.ipynb) rebuilds this trainer from verl's pieces, one swap at a time, until it *is* verl. First, free the GPU:
'''),
    code(r'''
llm.llm_engine.engine_core.shutdown()
del llm, policy, optimizer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU free: {gpu_free_gib():.0f} GiB")
''')]
save("1_grpo_from_scratch.ipynb", nb1)


# ═════════════════════════════════════════════════════════════════════════════
# Notebook 2: From our trainer to verl (§3–§6)
# ═════════════════════════════════════════════════════════════════════════════
RL_CORE_IMPORTS = '''
from rl_core import (EVAL, INSTRUCTION, LR, PAD_ID, STOP_IDS, TRAIN, chat_prompt_ids, compute_log_probs,  # noqa: E402
                     decode_tokens, evaluate, grpo_advantages, load_policy, make_optimizer, make_padded_batch,
                     plot_histories, ppo_clip_loss, ppo_update, print_step_metrics, push_weights, rl_step,
                     save_for_inference, select_rows, tokenizer, vllm_generate)
from arith_task import BOXED, make_problems, reward_fn, safe_eval  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402
'''.strip()

nb2 = [md(r'''
# RL training for LLM agents — 2. From our trainer to verl

### Rebuilding the notebook 1 trainer from verl's pieces, one swap at a time

''' + SERIES_TABLE + r'''

**Where we are.** [Notebook 1](1_grpo_from_scratch.ipynb) built a complete GRPO trainer in plain PyTorch (§1) and moved generation to vLLM, with weight sync after every update (§2). It works, but everything runs in **one process**: one policy, one vLLM engine, one Python loop. That's fine for a 0.5B model on one GPU, and impossible for a 70B model on 64 GPUs.

**In this notebook** we replace our pieces with verl's, one at a time, and check after every swap that nothing changed except *whose* code runs:

| § | We write | Then replace it with | The problem that forces the swap |
|---|---|---|---|
| 3 | a dict of padded tensors | verl `DataProto` | a batch needs to travel between processes |
| 4 | our advantage and loss functions | `verl.trainer.ppo.core_algos` | (none: same maths, battle-tested code, and we check they agree) |
| 5 | one process doing everything | **Ray** actors: a rollout engine and a trainer | they need different resources and lifetimes |
| 6 | our own `WorkerGroup` (split → call → gather) | verl `RayWorkerGroup` + `@register(Dispatch...)` | one actor per GPU, N GPUs |
| 6.3 | | **plain verl**, on our arithmetic task | everything above, wired together |

**What we need to know already.** Notebook 1, or at least §1.3–§1.6: the batch layout, log-probs, GRPO advantages, and `rl_step`. Its code is imported here from `tutorial/lib/rl_core.py`.

**Hardware and time.** One GPU with about 80 GB, and about 5 minutes on an H100, most of it in §6.3.
'''),
    md(r'''
## 0. Setup

The same setup as notebook 1, plus notebook 1's trainer, imported from `tutorial/lib/rl_core.py`.
'''),
    setup_cell(RL_CORE_IMPORTS),
    md(r'''
**Where notebook 1 left off.** §3 and §4 work on a real training batch, so we recreate notebook 1's end state: the policy and its optimizer, a vLLM engine sharing the GPU, and one `rl_step` (16 questions × 8 attempts), after which the new weights are pushed into vLLM. `batch` and `samples` are that step's batch and raw samples.
'''),
    code(r'''
policy = load_policy()
optimizer = make_optimizer(policy, LR)
llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=4096, seed=SEED)
step_rng = random.Random(SEED)

metrics, batch, samples = rl_step(policy, optimizer, lambda prompts, samples_per_prompt: vllm_generate(llm, prompts, samples_per_prompt),
                                  step_rng.sample(TRAIN, 16), samples_per_prompt=8)
push_weights(policy, llm)
print_step_metrics(0, metrics)
print(f"batch: {batch['responses'].shape[0]} rows; keys: {list(batch)}")
'''),
]
for i in range(find(SRC, "## 3. Swap 2"), find(SRC, "## 7. Multi-turn")):
    src = text(SRC, i)
    if src.startswith("def build_batch("):
        nb2.append(take(SRC, i, ("    print_step_metrics(len(vllm_history) + step, metrics)", "    print_step_metrics(step + 1, metrics)")))
    elif src.startswith("def ray_rl_step("):
        nb2.append(take(SRC, i, ('plot_histories({"§2 vLLM, one process": vllm_history, "§5 Ray actors": ray_history})',
                                 'plot_histories({"§4 one process": verl_fn_history, "§5 Ray actors": ray_history})')))
    elif src.startswith("The driver loop is the §4 loop"):
        nb2.append(take(SRC, i, ("These actors start from the base model (the §2 policy is gone), so the reward starts low again.",
                                 "These actors start from the base model (the §4 policy is gone), so the reward starts lower again.")))
    else:
        nb2.append(section_refs(take(SRC, i), 2))

VERL_SINGLE_TURN_REWARD = r'''
%%writefile tutorial/generated/verl_reward.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
from arith_task import reward_fn  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return reward_fn(solution_str, int(ground_truth))
'''
nb2 += [md(r'''
### 6.3 The real thing: plain verl on our task

Everything from §1 to §6 is now verl's job: Ray worker groups, the FSDP actor, colocated vLLM with weight sync, `DataProto`, GRPO, PPO-clip. Let's hand it the task from notebook 1, unchanged, and let verl run the whole loop. What *we* provide is the part that's specific to our task, in verl's format:

* **the reward**, as a function `compute_score(data_source, solution_str, ground_truth, extra_info)`: our `reward_fn`, called on the decoded response;
* **the data**, as parquet rows with a chat `prompt`, a `data_source` name, and the answer in `reward_model.ground_truth`.

verl starts its own Ray cluster and its own vLLM, so first we stop ours: the rollout actor's vLLM engine, then Ray itself.
'''),
    code(r'''
ray.get(rollout.shutdown.remote())  # stop vLLM's engine process before its parent actor goes away
ray.shutdown()
time.sleep(5)
print(f"GPU free: {gpu_free_gib():.0f} GiB")
'''),
    code(VERL_SINGLE_TURN_REWARD),
    code(r'''
import datasets


def to_verl_rows(problems):
    return [
        {"data_source": "arith", "prompt": [{"role": "user", "content": f"{problem['question']} {INSTRUCTION}"}],
         "reward_model": {"style": "rule", "ground_truth": str(problem["answer"])}, "extra_info": {"index": index}}
        for index, problem in enumerate(problems)
    ]


datasets.Dataset.from_list(to_verl_rows(TRAIN[:512])).to_parquet(str(WORK / "arith_single_train.parquet"))
datasets.Dataset.from_list(to_verl_rows(EVAL[:64])).to_parquet(str(WORK / "arith_single_val.parquet"))

SINGLE_TURN_METRICS = WORK / "single_turn_verl_metrics.jsonl"
SINGLE_TURN_METRICS.unlink(missing_ok=True)
verl_cmd = [
    sys.executable, "-m", "verl.trainer.main_ppo",
    "algorithm.adv_estimator=grpo",
    f"data.train_files={WORK / 'arith_single_train.parquet'}", f"data.val_files={WORK / 'arith_single_val.parquet'}",
    "data.train_batch_size=16", "data.max_prompt_length=256", "data.max_response_length=256", "data.dataloader_num_workers=0",
    f"actor_rollout_ref.model.path={MODEL}",
    "actor_rollout_ref.model.use_remove_padding=True", "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    f"actor_rollout_ref.actor.optim.lr={LR}",
    "actor_rollout_ref.actor.ppo_mini_batch_size=8", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
    "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.actor.clip_ratio_low=0.2", "actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.4", "actor_rollout_ref.rollout.n=8",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
    "actor_rollout_ref.rollout.agent.num_workers=2",
    f"reward.custom_reward_function.path={GENERATED / 'verl_reward.py'}", "reward.custom_reward_function.name=compute_score",
    "reward.num_workers=2",
    "trainer.n_gpus_per_node=1", "trainer.nnodes=1", "trainer.total_training_steps=10",
    "trainer.val_before_train=True", "trainer.test_freq=10", "trainer.save_freq=-1",
    "trainer.logger=[console,file]", "trainer.project_name=rl_training_series", "trainer.experiment_name=single_turn_verl",
    f"trainer.rollout_data_dir={WORK / 'single_turn_rollouts'}", f"trainer.validation_data_dir={WORK / 'single_turn_val'}",
]
print(" \\\n  ".join(verl_cmd[2:]))
'''),
    md(r'''
Every line of that config maps onto something we built:

| Config | Our code |
|---|---|
| `algorithm.adv_estimator=grpo` | `grpo_advantages` / `compute_grpo_outcome_advantage` (§1.5, §4) |
| `actor.clip_ratio_low/high`, `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu` | `ppo_clip_loss` and `ppo_update`'s mini/micro-batch loops (§1.5) |
| `rollout.name=vllm`, `gpu_memory_utilization`, `n` | `RolloutWorker` and `vllm_generate(samples_per_prompt=...)` (§2, §5) |
| `trainer.n_gpus_per_node`, `nnodes` | the resource pool (§6.2) |
| `reward.custom_reward_function` | `reward_fn` |
| `model.use_remove_padding=True` | (new) instead of padding rows to a rectangle (§1.3), verl packs all the real tokens of a batch into one long sequence, so no compute is spent on padding |

**Watch the units.** `data.train_batch_size=16` is in **prompts**: each step takes 16 questions, and `rollout.n=8` samples each one 8 times, giving 128 rows. `ppo_mini_batch_size=8` is also in prompts, and verl multiplies it by `n` too: 8 × 8 = 64 rows. So each training step is **two** optimizer steps, exactly like our §2 loop (128 rows, mini-batches of 64), and from the second one on the PPO ratio is no longer 1. `ppo_micro_batch_size_per_gpu`, by contrast, counts rows.

We run 10 steps, validating on 64 held-out problems before and after. Most of the time is start-up (Ray, the FSDP actor, vLLM with CUDA graphs). Selected log lines are printed; the full log is in `/tmp/rl-from-scratch/logs/single_turn_verl.log`.
'''),
    code(r'''
start_service("single_turn_verl", verl_cmd, cwd=WORK, env={"VERL_FILE_LOGGER_PATH": str(SINGLE_TURN_METRICS), "WANDB_MODE": "disabled"})
exit_code = follow_log("single_turn_verl", r"step:\d+ |Initial validation|Training Progress")
print(f"\nverl exited with code {exit_code}")
if exit_code != 0:
    raise RuntimeError(log_tail("single_turn_verl", 60))
'''),
    md(r'''
verl's file logger writes one JSON line per step. `read_metrics` keeps the metrics we ask for, by name pattern, as a table with one column per step (step 0 is the validation run before training):
'''),
    code(r'''
def read_metrics(path: Path, patterns: list[str]) -> pd.DataFrame:
    """verl's file logger writes one JSON line per step; keep the metrics whose names match any pattern, in pattern order."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    all_names = sorted({name for record in records for name in record["data"]})
    keep = [name for pattern in patterns for name in all_names if re.fullmatch(pattern, name)]
    keep = list(dict.fromkeys(keep))  # drop duplicates, keep order
    rows = [{"step": record["step"], **{name: record["data"].get(name) for name in keep}} for record in records]
    return pd.DataFrame(rows).groupby("step").last()


read_metrics(SINGLE_TURN_METRICS, [
    r"val-core/.*mean@1", r"critic/score/mean", r"actor/pg_loss", r"actor/grad_norm", r"actor/ppo_kl", r"actor/pg_clipfrac",
    r"actor/entropy", r"response_length/mean", r"timing_s/(gen|old_log_prob|update_actor|update_weights|step)",
]).T
'''),
    md(r'''
**Reading the metrics.** verl logs a few hundred metrics per step; every one in the table above is something we computed by hand:

| Metric | What it tells us | Where we built it |
|---|---|---|
| `val-core/…/mean@1` | accuracy on held-out problems (greedy): the number we actually care about | `evaluate` (§2.1) |
| `critic/score/mean` | mean training reward (verl logs it under `critic/` even without a critic) | `reward` in `rl_step` (§1.6) |
| `actor/pg_loss`, `actor/grad_norm` | the update itself. The loss *value* means little; its gradient carries the signal | `ppo_update` (§1.5) |
| `actor/ppo_kl`, `actor/pg_clipfrac` | how far the policy moved within the step, and the fraction of tokens whose gradient was clipped. With two mini-batches per step they're no longer exactly 0, but should stay small | the ratio and clip in `ppo_clip_loss` (§1.5) |
| `actor/entropy` | how spread out the policy's next-token distribution is. Collapsing entropy means the model has stopped exploring | |
| `response_length/mean` | how the model's behaviour is changing | `len` in `print_step_metrics` |
| `timing_s/*` | where the time goes: generation, old log-probs, update, weight sync | the timings in every `rl_step` |

`trainer.rollout_data_dir` makes verl dump every episode it generated as JSONL: the decoded prompt and response, the ground truth and the score. Here are the training scores step by step, and one response from the last step:
'''),
    code(r'''
for path in sorted((WORK / "single_turn_rollouts").glob("*.jsonl"), key=lambda path: int(path.stem)):  # one file per training step
    scores = [json.loads(line)["score"] for line in path.read_text().splitlines()]
    print(f"training step {path.stem:>2}: {len(scores)} episodes, mean score {np.mean(scores):.2f}")

last_step = max((WORK / "single_turn_rollouts").glob("*.jsonl"), key=lambda path: int(path.stem))
example = json.loads(last_step.read_text().splitlines()[0])
print("\n" + "─" * 100 + "\n" + example["output"] + f"\n{'─' * 100}\nground truth {example['gts']} | score {example['score']}")
'''),
    md(r'''
## What we built, and what each library is for

| We wrote | It became | Why the library version |
|---|---|---|
| dict of padded tensors | `DataProto` | splits, concatenates and travels between processes |
| `grpo_advantages`, `ppo_clip_loss` | `core_algos` | the same maths, plus dual-clip, many estimators and losses, aggregation modes |
| one process | Ray actors | separate roles, resources and lifetimes |
| `MyWorkerGroup` | `RayWorkerGroup` + `@register(Dispatch...)` | one worker per GPU, N GPUs, ranks and rendezvous handled |
| our training loop | `RayPPOTrainer` (`verl.trainer.main_ppo`) | all of the above, wired together and configured from one tree |

For a **single-turn** task like this one, that's the whole story: verl alone is the right tool, and our only code is the reward and the data.

**Next:** [notebook 3](3_training_real_agents.ipynb) moves on to agents, where the model calls tools over several turns, and then to agents that already exist as their own programs, which is where Agent Lightning comes in.
'''),
]
save("2_from_trainer_to_verl.ipynb", nb2)


# ═════════════════════════════════════════════════════════════════════════════
# Notebook 3: Training real agents (§7–§10)
# ═════════════════════════════════════════════════════════════════════════════
VERL_CORE_IMPORTS = RL_CORE_IMPORTS + '''
from verl import DataProto  # noqa: E402
from verl_core import add_advantages, build_batch, compute_batch_metrics, verl_ppo_loss  # noqa: E402
import httpx  # noqa: E402
'''
nb3 = [md(r'''
# RL training for LLM agents — 3. Training real agents

### Multi-turn tool use, recording an unmodified agent, and verl + Agent Lightning end to end

''' + SERIES_TABLE + r'''

**Where we are.** [Notebook 1](1_grpo_from_scratch.ipynb) built a GRPO trainer from scratch, with vLLM for generation. [Notebook 2](2_from_trainer_to_verl.ipynb) rebuilt it from verl's pieces (`DataProto`, `core_algos`, Ray actors, worker groups) and ran plain verl on our arithmetic task. All of that was **single-turn**: one prompt in, one response out.

**In this notebook** the model becomes an agent:

| § | We write | Then replace it with | The problem that forces the swap |
|---|---|---|---|
| 7 | a token-level multi-turn tool loop | verl's `ToolAgentLoop`, then… a **real** agent | real agents speak HTTP and own their control flow |
| 8 | a 60-line HTTP gateway that records tokens | Agent Lightning's Gateway + Controller | rollouts as processes, state, failures, merging calls into rows |
| 9 | | plain `verl` and `verl` + **Agent Lightning**, end to end | |

**Where Agent Lightning fits.** For agents we're willing to (re)write as a verl agent loop, verl alone is the right tool (§7.2, §9.1). [Agent Lightning](https://github.com/microsoft/agent-lightning) starts paying off the moment the agent is an existing program we don't want to rewrite (§7.3). This isn't a tutorial on its API (its [docs](https://microsoft.github.io/agent-lightning/) cover that); we build the idea ourselves in §8, then swap in the real thing and look at what it does that our version doesn't.

**What we need to know already.** Notebook 1's trainer (§1.3–§1.6, §2.1) and notebook 2's verl helpers (§4); both are imported here from `tutorial/lib/`.

**Hardware and time.** One GPU with about 80 GB, and about 10 minutes on an H100 (plus a 3 GB download for the 1.5B model used from §8 on).
'''),
    md(r'''
## 0. Setup

The same setup as the other notebooks, plus notebook 1's trainer (`rl_core`) and notebook 2's verl helpers (`verl_core`), imported from `tutorial/lib/`.
'''),
    setup_cell(VERL_CORE_IMPORTS),
    md(r'''
§7 trains on multi-turn tool episodes with the setup from the end of notebook 1: the policy and its optimizer in this process, and a vLLM engine sharing the GPU. (Notebook 2 moved these into Ray actors; the tool loop works the same either way, so we keep it simple here.)
'''),
    code(r'''
policy = load_policy()
optimizer = make_optimizer(policy, LR)
llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=4096, seed=SEED)
step_rng = random.Random(SEED)
print(f"GPU free: {gpu_free_gib():.0f} GiB")
'''),
]
i_judge_start = find(SRC, "### 8.4 A reward that needs a model")
i_judge_end = find(SRC, "## 9. The real thing")
for i in range(find(SRC, "## 7. Multi-turn"), len(SRC["cells"])):
    if i_judge_start <= i < i_judge_end:
        continue  # the judges are notebook 4
    src = text(SRC, i)
    if src.startswith("def run_episodes("):
        nb3.append(take(SRC, i, ("        turn_outputs = ray.get(rollout.generate.remote(contexts, 1, max_tokens))",
                                 "        turn_outputs = vllm_generate(llm, contexts, 1, max_tokens)")))
    elif src.startswith("def tool_rl_step("):
        nb3.append(take(SRC, i,
                        ('    batch.batch["old_log_probs"] = ray.get(trainer.compute_log_probs.remote(batch))',
                         '    batch.batch["old_log_probs"] = compute_log_probs(policy, batch.batch)'),
                        ("    stats = ray.get(trainer.update.remote(batch))",
                         '    stats = ppo_update(\n        policy, optimizer, batch.batch,\n        old_log_probs=batch.batch["old_log_probs"], advantages=batch.batch["advantages"], loss_fn=verl_ppo_loss,\n    )'),
                        ("    weights_path = ray.get(trainer.save_weights.remote())\n    ray.get(rollout.load_weights.remote(weights_path))",
                         "    push_weights(policy, llm)")))
    elif src.startswith("| `run_episodes` (§7.1) | verl `ToolAgentLoop` |"):
        nb3.append(take(SRC, i, ("| `rollout.generate.remote(prompt_ids + response_ids)` |", "| `vllm_generate(llm, prompt_ids + response_ids, 1)` |")))
    elif src.startswith("## 8. Swap 6"):
        nb3.append(take(SRC, i, ("First, stop Ray: its actors hold the GPU.",
                                 "First, free the GPU: the in-process vLLM engine and the policy from §7 go away.")))
    elif src.startswith("ray.get(rollout.shutdown.remote())"):
        nb3.append(take(SRC, i, ('ray.get(rollout.shutdown.remote())  # stop vLLM\'s engine process before its parent actor goes away\nray.shutdown()\n',
                                 "llm.llm_engine.engine_core.shutdown()\ndel llm, policy, optimizer\ngc.collect()\ntorch.cuda.empty_cache()\n")))
    elif src.startswith("Every rollout that made more than one call breaks continuity"):
        nb3.append(section_refs(take(SRC, i, (
            "The policy, our vLLM server and Agent Lightning's services have done their job, so let's free the GPU.",
            "The policy, our vLLM server and Agent Lightning's services have done their job, so let's free the GPU. (The LLM judges that read these same rollouts are [notebook 4](4_rewards_and_judges.ipynb).)")), 3))
    elif src.startswith("%%writefile tutorial/generated/verl_calc_tool.py") or src.startswith("%%writefile tutorial/generated/verl_reward.py"):
        nb3.append(take(SRC, i, ("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
                                 'sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))')))
    elif src.startswith("## 10. What we built"):
        nb3.append(take(SRC, i, ("**Next:** [Part 2](rl_training_walkthrough.ipynb) uses verl + Agent Lightning end-to-end on the full Calc-X task: a manual rollout walkthrough, real training launch, and reading the metrics.",
                                 "**Next:** [notebook 4](4_rewards_and_judges.ipynb) looks at the reward itself: what an exact check really measures, and what changes when a model does the grading. [Notebook 5](5_calc_x_end_to_end.ipynb) drives one Calc-X training step through Agent Lightning by hand, then launches a real training run.")))
    else:
        nb3.append(section_refs(take(SRC, i), 3))
save("3_training_real_agents.ipynb", nb3)


# ═════════════════════════════════════════════════════════════════════════════
# Notebook 4: Rewards — exact checks and LLM judges
# ═════════════════════════════════════════════════════════════════════════════
nb4 = [md(r'''
# RL training for LLM agents — 4. Rewards: exact checks and LLM judges

### What a reward really measures, and what changes when a model does the grading

''' + SERIES_TABLE + r'''

In RL, the reward is the **only** training signal. Everything the earlier notebooks built (advantages, the PPO loss, the whole verl and Agent Lightning machinery) exists to push the policy toward whatever the reward function returns. So whatever the reward gets wrong, the policy learns.

**In this notebook:**

| § | What we do |
|---|---|
| 1 | Look at what makes rewards hard to get right, using the Calc-X agent's own exact check |
| 2 | Generate rollouts to grade: run the unmodified Calc-X agent 128 times, recording its answers |
| 3 | Put an **LLM judge** next to the exact check on the same rollouts, and compare |
| 4 | Read every disagreement |
| 5 | Probe the judge with responses designed to fool it |
| 6 | Where the judge lives in a real training stack |

**What we need to know already.** What a reward is for (notebook 1, §1.1 and §1.5). §2 reuses the recording gateway from notebook 3 (§8.1), imported from `tutorial/lib/recording_gateway.py`; this notebook only uses it, so notebook 3 isn't required.

**Hardware and time.** One GPU with about 80 GB, and about 5 minutes. The judge is either a model deployed on Azure AI Foundry (set `FOUNDRY_ENDPOINT` and `FOUNDRY_DEPLOYMENT`, see §3), or, without one, `Qwen2.5-7B-Instruct` served locally by vLLM (a 15 GB download).
'''),
    md(r'''
## 0. Setup

The same setup as the other notebooks. The tokenizer is the Calc-X policy's, which we need to read the recorded responses.
'''),
    setup_cell('''
import uuid  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained(CALC_X_MODEL)
sys.path.insert(0, str(REPO / "examples/calc_x"))  # for eval_utils: the Calc-X agent's own answer check
'''.strip()),
    md(r'''
## 1. Why rewards are hard to get right

Rewards like the ones in this series are:

* **sparse**: one number at the very end of a response, or of a whole multi-turn episode;
* **outcome-based**: they say *whether* an attempt worked, not *which* tokens were good or bad;
* **literal**: the policy optimizes exactly what we check, not what we meant. If the format is part of the check, the format is part of what gets learned; in notebook 1 (§2.2) and notebook 3 (§7.1, §9.1), most of the early improvement was the model learning to box its answer.

The Calc-X agent's check is a good example of the last point. It extracts the text between `### ANSWER:` and `###` and compares it with the reference using `scalar_are_results_same` (from `examples/calc_x/eval_utils.py`): exact match, multiple-choice letter match, or numeric equality within 1% relative tolerance (parsed with sympy):
'''),
    take(WALK, find(WALK, "from eval_utils import scalar_are_results_same")),
    md(r'''
Notice `("105", "A") -> 0`. If the question is multiple choice and the model answers with the *value* instead of the *letter*, it gets nothing, even when its maths is right. And `"about 42"` scores 0 against `"42"`. Both are defensible rules, but they are *rules we chose*, and the policy will learn them as literally as it learns the arithmetic.

Many tasks can't be checked with a few lines of Python at all ("is this explanation correct?", "did the agent leave the database in the right state?"). The reward then comes from an **online verifier**: a test runner in a sandbox, an external API, or another model. The rest of this notebook looks at the last of these, an **LLM judge**, next to Calc-X's exact check.
'''),
    md(r'''
## 2. Rollouts to grade

To compare two graders we need something to grade: real Calc-X agent responses. We run the unmodified agent (the same one notebook 3 trains, `examples/calc_x/calc_agent.py`) on 32 questions, 4 attempts each.

The agent can't talk to vLLM directly. It asks for a model called `"auto"`, it must post its reward to an event URL (it raises an error if that fails), and its final answer is only visible in its HTTP traffic. Notebook 3's 60-line **recording gateway** (§8.1) handles all three: it forwards the agent's requests to vLLM under the real model name, accepts the reward event, and records every call's tokens under the rollout's ID. We import it from `tutorial/lib/recording_gateway.py`.

First, the policy's vLLM server, with the flags from notebook 3, §8.1 (no weight sync this time: we only generate):
'''),
    code(r'''
from recording_gateway import GATEWAY_URL, RECORDS, VLLM_PORT, VLLM_ROOT, run_agents, start_gateway

start_service("vllm", [
    str(VENV_BIN / "vllm"), "serve", CALC_X_MODEL, "--host", "127.0.0.1", "--port", str(VLLM_PORT),
    "--gpu-memory-utilization", "0.3", "--max-model-len", "8192", "--seed", str(SEED),
    "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--generation-config", "vllm",
])
wait_until_healthy("vllm", f"{VLLM_ROOT}/health")
gateway_server = start_gateway()
print("gateway up:", httpx.get(f"{GATEWAY_URL}/healthz").json())
'''.replace("from recording_gateway", "import httpx\n\nfrom recording_gateway", 1)),
    md(r'''
**The judge never goes through this gateway.** Everything on a rollout's gateway URL is recorded as training data for the policy: if the judge were called there, its prompt and verdict would be trained on as if the policy had written them. (The gateway even rewrites the model name to the policy's, so the "judge" would actually be the policy grading itself.) A judge needs its own endpoint, which the agent or the trainer calls directly. That's easy to forget once the gateway is just an import, so it's worth saying where it's set up.

Now the rollouts: 32 questions × 4 attempts, all at once, each agent in its own process. Each rollout gets its own ID, and the gateway files everything it records under that ID.
'''),
    code(r'''
calc_train = pd.read_parquet(CALC_X_DATA / "train.parquet")
calc_rng = random.Random(SEED)
questions = calc_train.iloc[calc_rng.sample(range(len(calc_train)), 32)]

jobs = []  # (rollout_id, question, reference result), what run_agents needs
for q_index in range(len(questions)):
    question = questions.iloc[q_index]
    for _ in range(4):  # 4 attempts per question, each with its own id
        jobs.append((f"q{q_index}-{uuid.uuid4().hex[:6]}", question["question"], question["result"]))

start = time.time()
exit_codes = run_agents(jobs)
print(f"{len(jobs)} rollouts in {time.time() - start:.0f}s; exit codes: {pd.Series(exit_codes).value_counts().to_dict()}")
rewards = [RECORDS.get(rollout_id, {}).get("reward") for rollout_id, _, _ in jobs]
print(f"rollouts that reported a reward: {sum(r is not None for r in rewards)}; mean exact-check reward: {np.mean([r or 0.0 for r in rewards]):.3f}")
'''),
    md(r'''
The policy is the untrained 1.5B model sampling at temperature 1, so most attempts fail the exact check. That's fine: we're here to compare graders, not to train. The policy's vLLM server has done its job; stop it, so a local judge has the GPU to itself.
'''),
    code(r'''
gateway_server.should_exit = True
stop_service("vllm")
time.sleep(5)
print(f"GPU free: {gpu_free_gib():.0f} GiB")
'''),
]
nb4.append(take(SRC, i_judge_start,
                ("### 8.4 A reward that needs a model: LLM judges", "## 3. A reward that needs a model: LLM judges"),
                ("So far every reward has been a few lines of Python. Calc-X's agent extracts the text between `### ANSWER:` and `###` and compares it with the reference using `scalar_are_results_same` (exact match, multiple-choice letter, or numeric equality within 1%). Many tasks can't be checked like that, and the reward has to come from another model: an **LLM judge** that reads the question, a reference answer and the response, and gives a verdict.",
                 "An **LLM judge** reads the question, a reference answer and the response, and gives a verdict."),
                ("Let's put one next to the exact check, on the rollouts our agent already produced: the 96 from §8.2 and the 32 from §8.3.",
                 "Let's put one next to the exact check, on the 128 rollouts from §2."),
                ("**The judge must never be called through the gateway.** Every request on a rollout's gateway URL is recorded as training data for the policy, so the judge's prompt and verdict would be trained on as if the policy had written them. Our gateway even rewrites the model name to the policy's, so the \"judge\" would actually be the policy grading itself. A judge needs its own endpoint, which the agent (or the trainer) calls directly. With a hosted judge, the separation is hard to miss.\n\n", "")))
nb4.append(take(SRC, i_judge_start + 1))
nb4.append(take(SRC, i_judge_start + 2))
nb4.append(take(SRC, i_judge_start + 3, (
    '''graded = []
for rollout_id, question, result in gateway_jobs:
    recorded = RECORDS.get(rollout_id, {})
    if recorded.get("calls") and recorded.get("reward") is not None:
        graded.append({"source": "§8.2", "question": question, "reference": result,
                       "response": tokenizer.decode(recorded["calls"][-1]["response_ids"], skip_special_tokens=True), "exact": recorded["reward"]})
for rollout in completed:
    if rollout.triplets and rollout.final_reward is not None:
        graded.append({"source": "§8.3", "question": rollout.input["question"], "reference": rollout.input["result"],
                       "response": tokenizer.decode(rollout.triplets[-1].response["token_ids"], skip_special_tokens=True), "exact": rollout.final_reward})''',
    '''graded = []
for rollout_id, question, result in jobs:
    recorded = RECORDS.get(rollout_id, {})
    if recorded.get("calls") and recorded.get("reward") is not None:  # the agent finished and reported its exact-check reward
        final_response = tokenizer.decode(recorded["calls"][-1]["response_ids"], skip_special_tokens=True)  # its last model call
        graded.append({"question": question, "reference": result, "response": final_response, "exact": recorded["reward"]})''')))
nb4.append(md(r'''
Compare the two mean rewards and the table. The rows are the exact check's verdict and the columns the judge's: the diagonal is where they agree, and the off-diagonal cells are the interesting ones.

When we ran this with a hosted reasoning model as the judge (`gpt-6.1-sol` on Azure AI Foundry), it was more generous than the exact check: it marked about 22% of the rollouts correct, against 13%, and it never rejected an answer the exact check accepted, so every disagreement was the judge saying "correct" where the exact check said "wrong". The 128 calls took about 300 seconds of judge time between them, but only 7 seconds of wall-clock time.

Let's list every disagreement: the reference, the final answer as the exact check parses it from `### ANSWER: … ###`, and the first line of the judge's reasoning.
'''))
nb4.append(take(SRC, i_judge_start + 5, expect='ANSWER = re.compile'))
nb4.append(md(r'''
## 4. Reading the disagreements

Read down the table, together with the questions (which include the multiple-choice options). Which disagreements are judge mistakes, and which are places where the exact check is more literal than we meant it to be?

With the hosted judge, none of them looked like judge mistakes:

* most were the right **value instead of the option letter**: `7` where option E is 7, `22 sec` for option B. The exact check only accepts the letter;
* one was a **formatting slip**, a stray backtick after `9` that broke the exact check's parsing;
* one gave the right answer in a sentence **without the `### ANSWER` format** at all;
* one was genuinely ambiguous: `C) 54 %`, where the correct option is D) 54%. The judge accepted it because our prompt said to accept "the correct letter or its value". Is that right? It depends on what we want the model to learn, and the judge can only follow the rules we write down. A reward measures what we *wrote*, not what we *meant*, whether it's a regular expression or a prompt.

**The judge's own quality matters too.** With the fallback, the local 7B judge, the picture was very different. In our runs it marked about 41% of the rollouts correct against 11% for the exact check, and in about two-thirds of the disagreements it was simply wrong: it stretched the "rounding within 1%" we asked for (`0.02` "is equivalent to 69/100 when rounded to two decimal places") and mapped arbitrary values to option letters. A policy trained against it would have been rewarded for a wrong answer in about one rollout in five, which is plenty of signal to learn from. A judge's errors become the policy's training signal, so its accuracy matters as much as anything else in the reward.

## 5. Can the policy game it?

The disagreements above happened by accident. A policy under RL pressure searches for them on purpose: any response the judge rewards and shouldn't is a direction the gradient will find. Let's probe the judge with a few hand-written responses to one question, as notebook 1 (§1.4) probed the policy: a correct answer, a wrong one, wrong answers dressed up to look right, and two that the exact check and the judge might see differently.
'''))
nb4.append(take(SRC, i_judge_start + 7, expect="probe = next("))
nb4.append(md(r'''
With the hosted judge, the judge held up against the obvious tricks: a hedge, and a note asking it to mark the answer correct. The two responses where the judge and the exact check disagreed were about definitions rather than mistakes:

* With **two answers**, the exact check reads the first `### ANSWER` and the judge, told to grade "the final answer", reads the last. Whichever of them we trained with, the policy would learn its convention, and anything downstream had better use the same one.
* Our prompt ignores the answer format, so a policy trained against this judge would have no reason to keep writing `### ANSWER: … ###`, which matters if something downstream parses it.

## 6. Where the judge lives in each stack

* **With Agent Lightning**, the reward is whatever number the agent posts at the end, so the judge call goes in the agent's own code, in place of `scalar_are_results_same`.
* **In plain verl**, the custom `compute_score` (notebook 2, §6.3) can call a judge endpoint just like `judge_verdict` does, or verl can host the judge itself: `reward.reward_model` serves a model with vLLM inside the training job, and `enable_resource_pool` gives it GPUs of its own.

Either way, rollouts run concurrently, so their judge calls arrive concurrently too, and the judge serves them in parallel: that's what makes a slow verifier affordable (compare the wall-clock time in §3 with the sum of the call latencies).

A common pattern in practice: use programmatic checks wherever they're possible, and a judge only for what can't be checked mechanically, with its prompt treated as part of the reward design, and probed like this before training against it.
'''))
nb4.append(take(SRC, i_judge_start + 9, expect='stop_service("judge")'))
save("4_rewards_and_judges.ipynb", nb4)


# ═════════════════════════════════════════════════════════════════════════════
# Notebook 5: Calc-X end to end (the walkthrough, trimmed to what the series doesn't cover)
# ═════════════════════════════════════════════════════════════════════════════
W = lambda prefix: find(WALK, prefix)  # noqa: E731
nb5 = [md(r'''
# RL training for LLM agents — 5. Calc-X end to end

### One training step driven by hand through Agent Lightning, then a real training run

''' + SERIES_TABLE + r'''

**Where we are.** Notebooks 1–3 built every piece of an agent RL trainer ourselves and swapped each one for verl's or Agent Lightning's, checking that nothing changed. Notebook 4 looked at the reward. This notebook is the practitioner's view: the real stack, on the full Calc-X task, used the way we'd use it for real.

We play the trainer, one HTTP call at a time: register the model, create a rollout, act as the agent, report a reward, and read back exactly what the Gateway recorded. Then we let the Controller run the real agent in groups, turn the traces into a verl batch, compute advantages the way Agent Lightning does, look at the config, and launch a real training run.

| § | Topic | Key idea |
|---|---|---|
| 0 | The big picture | who does what (verl vs. Agent Lightning), as messages between processes |
| 1 | The task | Calc-X data, and the reward (notebook 4 looks at it in depth) |
| 2 | Rollout engine + Gateway | start vLLM and the Agent Lightning Gateway |
| 3 | Be the agent ourselves | do one rollout by hand and see exactly what is recorded |
| 4 | Real agent rollouts | run the Calc-X agent many times per question (a GRPO *group*) |
| 5 | Traces → `DataProto` | Agent Lightning's `RolloutAdapter`, and where the reward goes |
| 6 | Advantages | per rollout, zero-signal groups, and per-rollout loss normalisation |
| 7 | How verl does it at scale | the training step inside the trainer, and the config tree |
| 8 | A short real training run | launch the actual trainer and read its metrics |

**What we need to know already.** Notebooks 1–3: this one doesn't re-explain log-probs, advantages, PPO, `DataProto` or the gateway idea. Where a topic was covered, we point back to it ("notebook 1, §1.5").

**Hardware:** one GPU with about 80 GB of memory. Background services (vLLM, Gateway, Controller) are started *from* the notebook and logged to `/tmp/agl-tutorial/`.
''')]

import re  # noqa: E402

MAJOR = {3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 7, 10: 8, 11: 9, 12: 10}  # walkthrough section -> notebook 5 section
KEEP = "@@S@@"  # stands for "§" in text that must not be renumbered (references into other notebooks)


def renumbered(src: str) -> str:
    """Renumber the walkthrough's sections for notebook 5 (its §2 and §8 are covered by notebook 1 instead)."""
    src = re.sub(r"§(\d+)", lambda m: "§" + str(MAJOR.get(int(m.group(1)), m.group(1))), src)
    src = re.sub(r"^(#{2,3} )(\d+)", lambda m: m.group(1) + str(MAJOR.get(int(m.group(2)), m.group(2))), src, flags=re.M)
    return src.replace(KEEP, "§")


def walk_cell(where: str | int, *patches: tuple[str, str]) -> dict:
    """A walkthrough cell (by index, or by its unique prefix), with exact patches applied first
    (use KEEP for "§" in references to other notebooks), then renumbered."""
    cell = take(WALK, where if isinstance(where, int) else W(where), *patches)
    cell["source"] = renumbered("".join(cell["source"])).splitlines(keepends=True)
    return cell


big_picture = text(WALK, W("## 0. The big picture"))
big_picture = big_picture.split("### Vocabulary we will see everywhere")[0] + "### Who does what in this repo?" + big_picture.split("### Who does what in this repo?")[1]
big_picture = big_picture.replace("RL for LLMs\n\nIn supervised", "RL for LLMs\n\n(Notebook 1 explains this loop in detail; here it is for reference.) In supervised")
big_picture = big_picture.replace("(§4.6–§8)", "(§4.6–§9)")  # the update (old §8) is now done by the trainer in §10, which renumbers to §8
nb5.append(md(renumbered(big_picture)))
nb5.append(walk_cell("### 0.1 Setup"))
nb5.append(walk_cell("import os\nimport sys\nimport time"))
nb5.append(walk_cell("The next cell is plumbing"))
nb5.append(walk_cell("import atexit"))
nb5.append(walk_cell("## 1. The task: Calc-X, and the reward"))
nb5.append(walk_cell("import numpy as np\nimport pandas as pd"))
nb5.append(walk_cell("from pprint import pprint"))
nb5.append(md(r"""
The reward is the agent's own exact check: it extracts the text between `### ANSWER:` and `###` and compares it with `result` using `scalar_are_results_same` (from `examples/calc_x/eval_utils.py`): exact match, multiple-choice letter match, or numeric equality within 1%. [Notebook 4](4_rewards_and_judges.ipynb) looks at what that check really measures, and puts an LLM judge next to it. A few examples:
"""))
nb5.append(walk_cell("from eval_utils import scalar_are_results_same"))
nb5.append(md(r"""
## 2. The tokenizer, and the rollout engine

The policy is `Qwen2.5-1.5B-Instruct`. Notebook 1 (§1.2–§1.4) covered what a policy is and how log-probs are computed from it; here we only need its tokenizer, to read the token IDs the Gateway records.
"""))
nb5.append(walk_cell("from transformers import AutoModelForCausalLM, AutoTokenizer",
                     ("from transformers import AutoModelForCausalLM, AutoTokenizer", "from transformers import AutoTokenizer")))
start_vllm_md = text(WALK, W("The whole job of RL training is to change these numbers"))
start_vllm_md = start_vllm_md.split("## 3. Start the rollout engine (vLLM) and the Agent Lightning Gateway\n\n")[1]
start_vllm_md = start_vllm_md.replace("(a Python library for running code across many processes and machines; §9.2 covers how verl uses it)",
                                      f"([notebook 2](2_from_trainer_to_verl.ipynb), {KEEP}5–{KEEP}6, builds it by hand and then with verl)")
nb5.append(md(renumbered(start_vllm_md)))
nb5.append(walk_cell("if port_in_use(VLLM_PORT):"))
nb5.append(walk_cell("Next, the **Gateway** (`agl-server`)."))
nb5.append(walk_cell("if port_in_use(AGL_PORT):"))

DATAPROTO_OLD = """Everything that flows between verl components is a `verl.DataProto`:

* `.batch`: a `TensorDict` of tensors that all share the first dimension (one entry per **training row**);
* `.non_tensor_batch`: a dict of numpy object arrays with the same first dimension (IDs, strings);
* `.meta_info`: a plain dict of batch-level info (temperature, step, …).

We can index it like a tensor (`batch[[0, 3, 5]]`), concatenate it, and `union` it with another `DataProto` (that's how `old_log_probs` get attached later). verl's workers receive and return `DataProto` objects over Ray."""
DATAPROTO_NEW = f"Everything that flows between verl components is a `verl.DataProto`: tensors in `.batch`, ids and strings in `.non_tensor_batch`, batch-level values in `.meta_info` ([notebook 2](2_from_trainer_to_verl.ipynb), {KEEP}3)."
ADV_OLD = """The basic policy gradient is

$$\\nabla_\\theta J = \\mathbb{E}\\Big[\\, A \\cdot \\nabla_\\theta \\log \\pi_\\theta(y \\mid x) \\Big],$$

that is, push up the log-prob of a response in proportion to its **advantage** A. If we used the raw reward (A = r), every correct answer would be pushed up equally, whether the question was trivial or hard, and wrong answers (r = 0) would teach nothing. Subtracting a **baseline** (what we expected to get) fixes this and greatly reduces variance.

PPO learns a separate *critic* network to predict that baseline. **GRPO** drops the critic and uses the group itself: with n rollouts for the same question,

$$A_i = \\frac{r_i - \\operatorname{mean}(r_1..r_n)}{\\operatorname{std}(r_1..r_n) + \\varepsilon}$$

and every response token of rollout *i* gets the same A_i. Let's compute it by hand from the batch."""
ADV_NEW = "Notebook 1 (§1.5) explains why GRPO compares each attempt with its group: A_i = (r_i − mean) / (std + ε), the same for every response token of attempt *i*. Here we compute it by hand from the real agent batch, then check it against what Agent Lightning computes."

for i in range(W("## 4. Be the agent"), W("## 8. The policy update")):
    src = text(WALK, i)
    if src.startswith("**With vs. without std normalization") or src.startswith("adv_no_std = compute_rollout_level_advantage(") \
            or src.startswith("With std normalization, different groups"):
        continue  # the Dr. GRPO comparison: covered (and corrected) in notebook 1, §1.5
    prefix = i
    if src.startswith("## 6. From traces to a verl training batch"):
        nb5.append(walk_cell(prefix, (DATAPROTO_OLD, DATAPROTO_NEW)))
    elif src.startswith("## 7. Advantages"):
        nb5.append(walk_cell(prefix, (ADV_OLD, ADV_NEW)))
    elif src.startswith("Since the model samples at temperature 1"):
        nb5.append(walk_cell(prefix, ("That is §1's \"the model optimizes exactly what we check\" in action.",
                                      "That is the \"the model optimizes exactly what we check\" point from notebook 4 in action.")))
    else:
        nb5.append(walk_cell(prefix))

nb5.append(md(r"""
## The update itself

The next step is the policy update: old log-probs, the PPO-clip loss, the backward pass and the optimizer step. [Notebook 1](1_grpo_from_scratch.ipynb) built every piece of it by hand (§1.4–§1.7), including the check that it worked, and its §2.3 showed why the trainer recomputes the old log-probs instead of trusting vLLM's. Here we let the real trainer do it, in §8; §7 first shows where each piece lives inside it.
"""))

for i in range(W("## 9. How verl does all of this at scale"), len(WALK["cells"])):
    src = text(WALK, i)
    prefix = i
    if src.startswith("## 9. How verl does all of this at scale"):
        nb5.append(walk_cell(prefix,
                             ("| §8.2 old log-probs |", "| old log-probs (notebook 1, §1.4) |"),
                             ("| §8.6 ref log-probs |", "| ref log-probs (not used by Calc-X) |"),
                             ("| §8.3 update |", "| update (notebook 1, §1.5) |"),
                             ("| §8.6 weight sync |", "| weight sync (notebook 1, §2.1) |"),
                             ("""verl uses a **single-controller** design. One driver process (the `fit()` loop above) holds the *logic*, while the *heavy compute* happens in **worker groups**: sets of Ray actors, one per GPU, that each own a shard of a model. When the driver calls `self.actor_rollout_wg.compute_log_prob(batch)`, verl splits the `DataProto` across the workers, runs the method on every GPU, and gathers the result. Scaling from 1 to 64 GPUs changes the resource pool, not the algorithm code. That's why this notebook's hand-made version and the real trainer look so similar.""",
                              f"""verl uses a **single-controller** design: one driver process (the `fit()` loop above) holds the *logic*, while the *heavy compute* happens in **worker groups** of Ray actors, one per GPU. When the driver calls `self.actor_rollout_wg.compute_log_prob(batch)`, verl splits the `DataProto` across the workers, runs the method on every GPU, and gathers the result. [Notebook 2](2_from_trainer_to_verl.ipynb) ({KEEP}5–{KEEP}6) builds exactly this, by hand and then with verl's `RayWorkerGroup`.""")))
    elif src.startswith("from examples.calc_x.train_calc_agent import build_config"):
        nb5.append(walk_cell(prefix, ("from examples.calc_x.train_calc_agent import build_config",
                                      "from omegaconf import OmegaConf\n\nfrom examples.calc_x.train_calc_agent import build_config")))
    elif src.startswith("### 10.1 Reading the metrics"):
        nb5.append(walk_cell(prefix,
                             ("| the update itself (§8.3) |", "| the update itself (notebook 1, §1.5) |"),
                             ("| how far the policy moved within a step (§8.5); should stay small |", "| how far the policy moved within a step (notebook 1, §1.5); should stay small |"),
                             ("| training-vs-inference log-prob mismatch (§8.2) |", "| training-vs-inference log-prob mismatch (notebook 1, §2.3) |"),
                             ("where the time goes (rollouts, §8.2, §8.3, weight sync)", "where the time goes (rollouts, old log-probs, update, weight sync)")))
    else:
        nb5.append(walk_cell(prefix))
save("5_calc_x_end_to_end.ipynb", nb5)
