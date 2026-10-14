"""Builds tutorial/verl_from_scratch.ipynb from the cell sources below."""

import json
import sys
import uuid

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md(r'''
# What is verl, *really*?

### Building an RL trainer for LLM agents from scratch, then swapping in verl and Agent Lightning piece by piece

Jeremy Howard's [*What is `torch.nn` really?*](https://pytorch.org/tutorials/beginner/nn_tutorial.html) trains a model with nothing but tensors, then replaces one hand-written piece at a time with its PyTorch equivalent until only idiomatic library code is left. After every swap the code still runs, and you know exactly what each library piece does because you wrote it first.

This notebook does the same for RL training of LLMs. Each stage ends with a **working training loop** and swaps exactly one thing:

| § | You write | Then replace it with | The problem that forces the swap |
|---|---|---|---|
| 1 | GRPO in plain PyTorch: sample, score, advantage, PPO-clip loss, AdamW | | |
| 2 | | `model.generate` → **vLLM** | generation is most of the step time; now you need **weight sync** |
| 3 | a dict of padded tensors | verl `DataProto` | a batch needs to travel between processes |
| 4 | your advantage and loss functions | `verl.trainer.ppo.core_algos` | (none: same maths, battle-tested code, and you check they agree) |
| 5 | one process doing everything | **Ray** actors: a rollout engine and a trainer | they need different resources and lifetimes |
| 6 | your own `WorkerGroup` (split → call → gather) | verl `RayWorkerGroup` + `@register(Dispatch...)` | one actor per GPU, N GPUs |
| 7 | a token-level multi-turn tool loop | verl's `ToolAgentLoop`, then… a **real** agent | real agents speak HTTP and own their control flow |
| 8 | a 60-line HTTP gateway that records tokens | Agent Lightning's Gateway + Controller | rollouts as processes, state, failures, scale |
| 9 | | plain `verl` and `verl` + **Agent Lightning**, end to end | |

**Where Agent Lightning fits.** Stages 1–6 don't need it, and the notebook says so: for single-turn tasks, or agents you're willing to rewrite as a verl agent loop, verl alone is the right tool. Agent Lightning starts paying off in §7, the moment the agent is an existing program you don't want to rewrite.

**Prerequisites.** This notebook builds on [rl_training_walkthrough.ipynb](rl_training_walkthrough.ipynb), which dissects the finished system top-down. Log-probs, GRPO advantages, the PPO-clip objective and the Gateway's role are explained there, and only recapped here.

**Hardware and time.** One GPU with about 80 GB, and roughly an hour end to end, mostly in §2 and §9. The policy is `Qwen2.5-0.5B-Instruct`, small enough that every stage trains in minutes. **Stop anything else using the GPU first** (for example the walkthrough notebook's vLLM server).
''')

md(r'''
## 0. Setup

Paths, the model name, and a few helpers for background processes (used from §8 on). Generated files and logs go to `/tmp/verl-from-scratch/`, and weights are handed from trainer to inference engine through `/dev/shm/verl-from-scratch/` (RAM, not disk).
''')

code(r'''
import gc
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "examples" / "calc_x").is_dir())
WORK = Path("/tmp/verl-from-scratch")      # generated files, datasets, logs
SHM = Path("/dev/shm/verl-from-scratch")   # weight hand-off from trainer to inference engine
for d in (WORK, SHM, WORK / "logs"):
    d.mkdir(parents=True, exist_ok=True)
VENV_BIN = Path(sys.executable).parent
CALC_X_DATA = Path(os.environ.get("CALC_X_DATA", REPO / "examples/calc_x/data"))

# Child processes (Ray workers, vLLM, agents) inherit these.
sys.path[:0] = [str(WORK), str(REPO)]
os.environ["PATH"] = f"{VENV_BIN}:{os.environ['PATH']}"
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [str(WORK), str(REPO), os.environ.get("PYTHONPATH", "")] if p)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

free, total = torch.cuda.mem_get_info()
print(f"repo: {REPO}\npython: {sys.executable}\nGPU: {torch.cuda.get_device_name(0)}, {free / 2**30:.0f} of {total / 2**30:.0f} GiB free")
if free < 0.85 * total:
    print("WARNING: something else is using this GPU. Stop it before continuing, or later stages may run out of memory.")
''')

code(r'''
import httpx

SERVICES: dict[str, subprocess.Popen] = {}


def start_service(name: str, cmd: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None) -> subprocess.Popen:
    stop_service(name)
    with open(WORK / "logs" / f"{name}.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    SERVICES[name] = proc
    print(f"started {name} (pid {proc.pid}), log: {WORK / 'logs' / f'{name}.log'}")
    return proc


def stop_service(name: str, timeout: float = 60) -> None:
    proc = SERVICES.pop(name, None)
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(f"stopped {name}")


def log_tail(name: str, n: int = 30) -> str:
    path = WORK / "logs" / f"{name}.log"
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


def wait_until_healthy(name: str, url: str, timeout: float = 900) -> None:
    start = time.time()
    while time.time() - start < timeout:
        proc = SERVICES.get(name)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{name} exited with code {proc.returncode}:\n{log_tail(name)}")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                print(f"{name} is healthy after {time.time() - start:.0f}s")
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{name} not healthy after {timeout}s:\n{log_tail(name)}")


def follow_log(name: str, pattern: str) -> int:
    """Print matching lines of a service's log until it exits; return its exit code."""
    proc, rx, t0 = SERVICES[name], re.compile(pattern), time.time()
    with open(WORK / "logs" / f"{name}.log", errors="replace") as log:
        while True:
            line = log.readline()
            if not line:
                if proc.poll() is not None:
                    return proc.returncode
                time.sleep(1)
                continue
            if rx.search(line):
                line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                print(f"[{time.time() - t0:5.0f}s] {line[:220]}{' …' if len(line) > 220 else ''}")


def gpu_free_gib() -> float:
    return torch.cuda.mem_get_info()[0] / 2**30


print("helpers ready")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 1. GRPO in plain PyTorch

### 1.1 The task

We need a task where a 0.5B model is sometimes right and sometimes wrong, so that comparing attempts carries signal. Three-term integer arithmetic does the job. The reward is 1 if the last `\boxed{...}` in the answer holds the right integer, else 0.

The task lives in a small module rather than in the notebook, because later stages (Ray workers, verl's reward workers) run in other processes and need to import it. `%%writefile` writes the cell to disk *and* shows it to you.

`safe_eval` will also be the agent's **calculator tool** in §7. It walks the expression's syntax tree and allows only numbers and `+ - * /`: text written by a model must never reach Python's `eval`.
''')

code(r'''
%%writefile /tmp/verl-from-scratch/arith_task.py
"""The toy task used throughout the notebook: integer arithmetic with a 0/1 reward."""

import ast
import operator
import random
import re

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """Evaluate `+ - * /` arithmetic by walking the syntax tree; anything else is rejected."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {expression!r}")

    return ev(ast.parse(expression, mode="eval"))


def make_problems(n: int, seed: int, hard: bool = False) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        if hard:  # too big to multiply reliably in your head: a calculator helps
            expr = f"{rng.randint(100, 999)} * {rng.randint(100, 999)} - {rng.randint(1000, 9999)}"
        else:
            expr = f"{rng.randint(2, 99)} {rng.choice('+-*')} {rng.randint(2, 99)} {rng.choice('+-*')} {rng.randint(2, 30)}"
        problems.append({"question": f"What is {expr}?", "answer": int(safe_eval(expr))})
    return problems


BOXED = re.compile(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}")


def reward_fn(text: str, answer: int) -> float:
    """1.0 if the last \\boxed{...} in `text` holds `answer`, else 0.0."""
    found = BOXED.findall(text)
    return float(bool(found) and int(found[-1].replace(",", "")) == answer)
''')

code(r'''
import importlib

import arith_task

importlib.reload(arith_task)
from arith_task import make_problems, reward_fn, safe_eval

TRAIN = make_problems(4000, seed=1)
EVAL = make_problems(200, seed=2)
INSTRUCTION = r"Show brief working, then give the final answer as \boxed{N}."

for p in TRAIN[:3]:
    print(p)
for text, answer in [(r"... so \boxed{42}", 42), (r"\boxed{41}", 42), ("The answer is 42.", 42), (r"\boxed{1,234}", 1234)]:
    print(f"reward({text!r}, {answer}) = {reward_fn(text, answer)}")
for bad in ["__import__('os').system('id')", "2 ** 999999"]:
    try:
        safe_eval(bad)
    except ValueError as exc:
        print("rejected:", exc)
''')

md(r'''
### 1.2 The policy, and sampling from it

The policy is the LLM. We keep **fp32 master weights** and run forward passes under **bf16 autocast**, as verl's FSDP actor does, with FlashAttention-2.

Look closely at the sampling arguments in `hf_generate`. RL needs samples from the policy's *actual* distribution π_θ, because the loss later compares log π_θ of those samples. The model's `generation_config` asks for `top_k=20, top_p=0.8, repetition_penalty=1.1, temperature=0.7`, which are sensible for chat but sample from a *different*, truncated distribution. So we override all four: `temperature=1, top_p=1, top_k=0 (off), repetition_penalty=1`. Remember this; it comes back in §2.
''')

code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # hides the "FA2 with fp32 weights" notice: fp32 master weights + bf16 autocast is intended

tokenizer = AutoTokenizer.from_pretrained(MODEL)
PAD_ID = tokenizer.pad_token_id
STOP_IDS = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
print("model's own sampling defaults:", {k: v for k, v in GenerationConfig.from_pretrained(MODEL).to_dict().items() if k in ("temperature", "top_p", "top_k", "repetition_penalty")})


def chat_prompt_ids(question: str) -> list[int]:
    return tokenizer.apply_chat_template([{"role": "user", "content": f"{question} {INSTRUCTION}"}], add_generation_prompt=True)


def decode(ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_policy(path: str = MODEL):
    hf_logging.set_verbosity_error()  # also inside Ray workers
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="flash_attention_2").to(DEVICE)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model


def make_optimizer(model, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def truncate_at_stop(ids: list[int]) -> list[int]:
    for i, t in enumerate(ids):
        if t in STOP_IDS:
            return ids[: i + 1]  # keep the stop token: the model must learn when to stop
    return ids


@torch.no_grad()
def hf_generate(model, prompts: list[list[int]], n: int, max_new_tokens: int = 256) -> list[dict]:
    """n samples per prompt with plain `model.generate`. Samples are prompt-major: prompt 0's n samples come first."""
    model.eval()
    rows = [p for p in prompts for _ in range(n)]
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, p in enumerate(rows):  # left-pad, so every prompt ends where generation starts
        input_ids[i, width - len(p):] = torch.tensor(p)
        attention_mask[i, width - len(p):] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE),
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            max_new_tokens=max_new_tokens, pad_token_id=PAD_ID, use_cache=True,
        )
    model.train()
    return [{"prompt_ids": p, "response_ids": truncate_at_stop(r), "log_probs": None} for p, r in zip(rows, out[:, width:].tolist())]


policy = load_policy()
print(f"policy: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M parameters")
samples = hf_generate(policy, [chat_prompt_ids(TRAIN[0]["question"])], n=3)
for s in samples:
    print("─" * 80, f"\nreward {reward_fn(decode(s['response_ids']), TRAIN[0]['answer'])}:", decode(s["response_ids"]))
''')

md(r'''
### 1.3 From samples to tensors

A batch is a set of padded tensors. Prompts are **left**-padded and responses **right**-padded, so every row's response starts at the same column. Two masks matter:

* `attention_mask` (prompt + response): which positions are real tokens, not padding.
* `response_mask` (response only): which response tokens are **trained**. For now that's every real response token. In §7 the tool outputs inside a response get `0`.

The key names (`prompts`, `responses`, `input_ids`, `attention_mask`, `position_ids`, `response_mask`) are verl's, which will make §3 a one-line change.
''')

code(r'''
def collate(prompts: list[list[int]], responses: list[list[int]], response_masks: list[list[int]] | None = None) -> dict[str, torch.Tensor]:
    B, P, R = len(prompts), max(map(len, prompts)), max(1, max(map(len, responses)))
    prompt_t = torch.full((B, P), PAD_ID, dtype=torch.long)
    response_t = torch.full((B, R), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((B, P + R), dtype=torch.long)
    response_mask = torch.zeros((B, R), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        prompt_t[i, P - len(p):] = torch.tensor(p)
        response_t[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        attention_mask[i, P - len(p): P + len(r)] = 1
        response_mask[i, : len(r)] = torch.tensor(response_masks[i] if response_masks else [1] * len(r), dtype=torch.long)
    return {
        "prompts": prompt_t,
        "responses": response_t,
        "input_ids": torch.cat([prompt_t, response_t], dim=1),
        "attention_mask": attention_mask,
        "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
        "response_mask": response_mask,
    }


demo = collate([s["prompt_ids"] for s in samples], [s["response_ids"] for s in samples])
for k, v in demo.items():
    print(f"{k:15} {tuple(v.shape)}")
''')

md(r'''
### 1.4 Log-probs, advantages, loss

Three small functions are the whole algorithm:

* **`token_log_probs`**: one forward pass over prompt + response, returning log π(yₜ | context) for every response position. `logits_to_keep=R+1` asks the model for logits of the last R+1 positions only (the logit at position t−1 predicts token t), which saves a lot of memory with a 150k-token vocabulary.
* **`grpo_advantages`**: per group of n attempts at the same question, A = (r − mean) / (std + ε).
* **`ppo_clip_loss`**: with ρ = π_θ / π_old, minimise −min(ρA, clip(ρ, 1−ε_low, 1+ε_high)A), summed over trained tokens and divided by the number of trained tokens in the **whole mini-batch**. Dividing by the mini-batch total, not the micro-batch, keeps gradient accumulation exact: the micro-batch losses add up to a token-mean over the mini-batch.
''')

code(r'''
def rows(batch, idx):
    """Index a batch given as a dict of tensors, or as verl's TensorDict (§3)."""
    return {k: v[idx] for k, v in batch.items()} if isinstance(batch, dict) else batch[idx]


def token_log_probs(model, mb, grad: bool = False) -> torch.Tensor:
    R = mb["responses"].shape[1]
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=mb["input_ids"].to(DEVICE), attention_mask=mb["attention_mask"].to(DEVICE),
            position_ids=mb["position_ids"].to(DEVICE), logits_to_keep=R + 1,
        ).logits[:, :-1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, mb["responses"].to(DEVICE).unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def compute_log_probs(model, batch, micro_batch_size: int = 16) -> torch.Tensor:
    model.eval()
    n = batch["input_ids"].shape[0]
    out = torch.cat([token_log_probs(model, rows(batch, slice(s, s + micro_batch_size))) for s in range(0, n, micro_batch_size)])
    model.train()
    return out.cpu()


def grpo_advantages(rewards: torch.Tensor, group_ids, eps: float = 1e-6) -> torch.Tensor:
    adv = torch.zeros_like(rewards)
    group_ids = np.asarray(group_ids)
    for g in np.unique(group_ids):
        idx = torch.as_tensor(np.nonzero(group_ids == g)[0])
        r = rewards[idx]
        adv[idx] = (r - r.mean()) / (r.std() + eps) if len(idx) > 1 else 0.0
    return adv


def ppo_clip_loss(log_probs, old_log_probs, advantages, mask, n_tokens, clip_low: float = 0.2, clip_high: float = 0.28):
    ratio = torch.exp(log_probs - old_log_probs)
    loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - clip_low, 1 + clip_high))
    return (loss * mask).sum() / n_tokens


def ppo_update(model, optimizer, batch, old_log_probs, advantages, mini_batch_size: int = 64, micro_batch_size: int = 8, loss_fn=ppo_clip_loss) -> dict:
    """One pass over the batch: one optimizer step per mini-batch, gradient accumulation over micro-batches."""
    n = batch["input_ids"].shape[0]
    grad_norms = []
    for start in range(0, n, mini_batch_size):
        stop = min(start + mini_batch_size, n)
        n_tokens = batch["response_mask"][start:stop].sum().item()
        for s in range(start, stop, micro_batch_size):
            idx = slice(s, min(s + micro_batch_size, stop))
            mb = rows(batch, idx)
            log_probs = token_log_probs(model, mb, grad=True)
            mask = mb["response_mask"].to(DEVICE).float()
            loss_fn(log_probs, old_log_probs[idx].to(DEVICE), advantages[idx].to(DEVICE), mask, n_tokens).backward()
        grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"grad_norm": float(np.mean(grad_norms))}
''')

md(r'''
### 1.5 The training step

`rl_step` is the whole RL loop for one step: **sample → score → batch → old log-probs → advantages → update**. It takes the generation function as an argument, which is what lets later stages swap the inference engine without touching anything else.

If the inference engine returned its own log-probs for the sampled tokens (HF `generate` doesn't, vLLM will), `rl_step` also reports `logp_mismatch`: the mean |log π_trainer − log π_engine| over trained tokens. It should be small, and in §2 it tells us whether weight sync works.
''')

code(r'''
def rl_step(policy, optimizer, generate, problems, n: int = 8, **update_kwargs) -> tuple[dict, dict, list[dict]]:
    timing = {}
    t0 = time.time()
    samples = generate([chat_prompt_ids(p["question"]) for p in problems], n)
    timing["gen"] = time.time() - t0

    rewards = torch.tensor([reward_fn(decode(s["response_ids"]), problems[i // n]["answer"]) for i, s in enumerate(samples)])
    batch = collate([s["prompt_ids"] for s in samples], [s["response_ids"] for s in samples])

    t0 = time.time()
    old_log_probs = compute_log_probs(policy, batch)
    timing["old_log_prob"] = time.time() - t0

    advantages = grpo_advantages(rewards, [i // n for i in range(len(samples))])[:, None] * batch["response_mask"]

    t0 = time.time()
    stats = ppo_update(policy, optimizer, batch, old_log_probs, advantages, **update_kwargs)
    timing["update"] = time.time() - t0

    metrics = {"reward": rewards.mean().item(), "response_len": batch["response_mask"].sum(-1).float().mean().item(), **stats}
    if samples[0]["log_probs"] is not None:
        engine = torch.zeros_like(old_log_probs)
        for i, s in enumerate(samples):
            engine[i, : len(s["log_probs"])] = torch.tensor(s["log_probs"])
        m = batch["response_mask"].bool()
        metrics["logp_mismatch"] = (old_log_probs - engine)[m].abs().mean().item()
    metrics.update({f"time/{k}": v for k, v in timing.items()})
    return metrics, batch, samples


def show(step: int, m: dict) -> None:
    extra = f" | logp mismatch {m['logp_mismatch']:.4f}" if "logp_mismatch" in m else ""
    times = " ".join(f"{k[5:]} {v:4.1f}s" for k, v in m.items() if k.startswith("time/"))
    print(f"step {step:2d} | reward {m['reward']:.3f} | len {m['response_len']:5.1f} | grad_norm {m['grad_norm']:.3f}{extra} | {times}")


LR = 2e-6
optimizer = make_optimizer(policy, LR)
step_rng = random.Random(SEED)
hf_history = []
for step in range(3):
    m, batch, samples = rl_step(policy, optimizer, lambda prompts, n: hf_generate(policy, prompts, n), step_rng.sample(TRAIN, 8), n=8)
    hf_history.append(m)
    show(step, m)
''')

md(r'''
That's a complete RL trainer. Look at the timing: **generation dominates the step**, even for 64 short samples. `model.generate` runs one Python-level step per token with no paging of the KV cache or continuous batching. At thousands of samples per step this is the bottleneck, and it's the first thing every real framework replaces.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 2. Swap 1: a real inference engine (vLLM)

vLLM generates with PagedAttention, continuous batching and CUDA graphs. We use its offline `LLM` class in this process (it starts its engine in a child process), on the same GPU as the policy, with 30% of GPU memory for weights and KV cache.

Two things change the moment generation leaves the training model:

1. **There are now two copies of the weights.** After every update the trainer must push its weights into vLLM, otherwise the next samples come from a stale policy (off-policy data).
2. **vLLM has its own sampling defaults**, and it takes them from the same `generation_config` you saw in §1.2.
''')

code(r'''
from vllm import LLM, SamplingParams, TokensPrompt

llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=4096, seed=SEED)
print("vLLM's default sampling params for this model:", llm.get_default_sampling_params())
''')

md(r'''
Those defaults are the model's chat-friendly settings. Call `llm.generate(prompts)` without your own `SamplingParams` and you'd silently be sampling from a top-k/top-p-truncated, repetition-penalised distribution. That's not the policy you're training. So `vllm_generate` always passes explicit, neutral parameters, and asks for the log-prob of each sampled token (`logprobs=0`). verl does the same: it overrides the generation config when it launches vLLM.

Note that vLLM returns log-probs of the *raw* logits by default (before temperature scaling), so they only equal the sampling distribution at temperature 1. RL rollouts use temperature 1 anyway.
''')

code(r'''
def vllm_generate(llm, prompts: list[list[int]], n: int, max_tokens: int = 256, temperature: float = 1.0) -> list[dict]:
    params = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0, max_tokens=max_tokens, logprobs=0)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
    samples = []
    for p, out in zip(prompts, outputs):
        for c in out.outputs:
            ids = list(c.token_ids)
            samples.append({"prompt_ids": list(p), "response_ids": ids, "log_probs": [lp[t].logprob for t, lp in zip(ids, c.logprobs)]})
    return samples


prompts = [chat_prompt_ids(p["question"]) for p in TRAIN[:8]]
for name, fn in [("HF generate", lambda: hf_generate(policy, prompts, 8)), ("vLLM", lambda: vllm_generate(llm, prompts, 8))]:
    t0 = time.time()
    out = fn()
    print(f"{name:12} {len(out)} samples in {time.time() - t0:5.2f}s")
print("vLLM samples end with a stop token:", np.mean([s["response_ids"][-1] in STOP_IDS for s in out]))
''')

md(r'''
### 2.1 Weight sync

The simplest correct weight sync, in two steps:

1. The trainer writes its weights to `/dev/shm` (RAM) in the Hugging Face checkpoint format, as bf16. That's the **data plane**.
2. A tiny control message tells vLLM to reload from there, `collective_rpc("reload_weights", ...)`, which runs on every vLLM worker. That's the **control plane**.

Then we **reset vLLM's prefix cache**: it holds KV-cache blocks computed with the *old* weights, and every prompt shares the same chat-template prefix, so stale blocks would quietly be reused.

verl does the same thing faster. It sends bf16 tensors straight from the FSDP shards to vLLM over CUDA IPC (same GPU) or NCCL (different GPUs), in buckets, with no copy through host memory. The structure is identical: a control call says *"new weights are coming"*, then a data channel moves them.

The policy already took 3 steps in §1, and vLLM loaded the original checkpoint from disk, so we sync once before doing anything else.
''')

code(r'''
import safetensors.torch


def save_for_inference(model, path: Path = SHM / "policy") -> str:
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)  # same tensor as the embedding; vLLM ties them itself
    safetensors.torch.save_file(state, str(path / "model.safetensors"))
    return str(path)


def push_weights(model, llm) -> None:
    llm.collective_rpc("reload_weights", kwargs={"weights_path": save_for_inference(model)})
    llm.reset_prefix_cache()


t0 = time.time()
push_weights(policy, llm)
print(f"weights pushed in {time.time() - t0:.1f}s")


def evaluate(generate_greedy, problems=EVAL) -> float:
    samples = generate_greedy([chat_prompt_ids(p["question"]) for p in problems], 1)
    return float(np.mean([reward_fn(decode(s["response_ids"]), p["answer"]) for s, p in zip(samples, problems)]))


acc_before = evaluate(lambda p, n: vllm_generate(llm, p, n, temperature=0.0))
print(f"greedy accuracy on {len(EVAL)} held-out problems before RL: {acc_before:.3f}")
''')

md(r'''
### 2.2 Train, with vLLM generating

Same `rl_step` as before, now with `vllm_generate`, and with `push_weights` after every update. Each step samples 16 questions × 8 attempts. Watch three things:

* `reward` should trend up (noisily: 16 questions is a small batch);
* `logp_mismatch` should stay small and flat. Here vLLM and the trainer use different kernels and precisions, so a small non-zero value (below 0.01 here) is normal. If you deleted the `push_weights` line, it would creep up as the trainer drifts away from what vLLM serves;
* `gen` time is now a small part of the step.
''')

code(r'''
vllm_history = []
for step in range(20):
    m, batch, samples = rl_step(policy, optimizer, lambda prompts, n: vllm_generate(llm, prompts, n), step_rng.sample(TRAIN, 16), n=8)
    t0 = time.time()
    push_weights(policy, llm)
    m["time/sync"] = time.time() - t0
    vllm_history.append(m)
    show(step, m)

acc_after = evaluate(lambda p, n: vllm_generate(llm, p, n, temperature=0.0))
print(f"\ngreedy accuracy on held-out problems: {acc_before:.3f} before -> {acc_after:.3f} after {len(vllm_history)} steps")
''')

code(r'''
def plot_histories(histories: dict[str, list[dict]], keys=("reward", "response_len")) -> None:
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3))
    for ax, key in zip(np.atleast_1d(axes), keys):
        for name, hist in histories.items():
            ax.plot([m[key] for m in hist], marker="o", ms=3, label=name)
        ax.set_title(key)
        ax.set_xlabel("step")
    np.atleast_1d(axes)[0].legend()
    plt.tight_layout()
    plt.show()


plot_histories({"§2 vLLM": vllm_history})
''')

md(r'''
Most of the early gain is usually the model learning to *always* finish with `\boxed{N}`: the reward checks the format, so the format is the first thing to improve. Look at a few samples (`decode(samples[i]["response_ids"])`) to see it.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 3. Swap 2: the batch becomes a `DataProto`

Our batch is a dict of tensors plus some side lists (group ids, answers). That stops working once the batch has to travel between processes and be split across workers, which is exactly what §5–§6 need. verl's container for this is `DataProto`:

* `.batch`: a `TensorDict`, i.e. tensors that share the first (batch) dimension;
* `.non_tensor_batch`: numpy object arrays with the same first dimension (ids, strings, answers);
* `.meta_info`: a plain dict for batch-level values (temperature, step, …).

Because §1 used verl's key names, the swap is one call: `DataProto.from_dict(tensors=collate(...), ...)`. And because `rows()` also accepts a `TensorDict`, every function from §1 works unchanged on `batch.batch`.
''')

code(r'''
from verl import DataProto

dp = DataProto.from_dict(
    tensors=collate([s["prompt_ids"] for s in samples], [s["response_ids"] for s in samples]),
    non_tensors={"uid": np.array([i // 8 for i in range(len(samples))], dtype=object)},
    meta_info={"temperature": 1.0},
)
print(f"{len(dp)} rows; tensors: {list(dp.batch.keys())}; non-tensors: {list(dp.non_tensor_batch)}; meta: {dp.meta_info}")

print("index with a list:     ", len(dp[[0, 5, 9]]), "rows, uids", dp[[0, 5, 9]].non_tensor_batch["uid"].tolist())
halves = dp.chunk(2)
print("chunk(2):              ", [len(h) for h in halves], "-> concat:", len(DataProto.concat(halves)))
extra = DataProto.from_dict(tensors={"old_log_probs": torch.zeros_like(dp.batch["responses"], dtype=torch.float32)})
print("union adds a column:   ", "old_log_probs" in dp.union(extra).batch.keys())
print("repeat(2, interleave): ", dp[[0, 1]].repeat(2, interleave=True).non_tensor_batch["uid"].tolist())
print("works with §1 helpers: ", compute_log_probs(policy, dp.batch).shape)
''')

md(r'''
`chunk` and `concat` are the operations a worker group (§6) is built on: split the batch into one piece per worker, run, and concatenate the results. `union` is how verl attaches results to the batch as it moves through the pipeline: `old_log_probs` from the actor, `ref_log_prob` from the reference model, `values` from the critic, then `advantages`.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 4. Swap 3: verl's algorithm functions

`verl.trainer.ppo.core_algos` holds the maths: advantage estimators (GAE, GRPO, RLOO, REINFORCE++, …) and policy losses (vanilla PPO-clip, GSPO, CISPO, …), registered by name. Swapping ours for theirs needs two adaptations:

* verl expects rewards **per token**, `token_level_rewards` of shape [B, R], with the scalar reward on each row's last response token. GRPO only uses the row sum; PPO with a critic would discount it backwards.
* The vanilla policy loss reads its clip ratios from a config object, and the token count for `token-mean` comes in through `config.global_batch_info`. That is our `n_tokens`: verl solved the same micro-batch accounting problem as §1.4.

First we check that the two implementations agree on a real batch.
''')

code(r'''
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, get_policy_loss_fn
from verl.utils.torch_functional import logprobs_from_logits

# Advantages: ours vs verl's, on the last §2 batch with random 0/1 rewards (more mixed groups than real ones)
n = 8
rewards = torch.tensor(np.random.default_rng(0).integers(0, 2, len(samples)), dtype=torch.float32)
mask = batch["response_mask"]
lengths = mask.sum(-1)
token_level_rewards = torch.zeros(mask.shape)
token_level_rewards[torch.arange(len(lengths)), lengths - 1] = rewards
uid = np.array([i // n for i in range(len(samples))], dtype=object)

ours = grpo_advantages(rewards, uid)[:, None] * mask
theirs, _ = compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards, response_mask=mask, index=uid)
print("GRPO advantages match:", torch.allclose(ours, theirs, atol=1e-5))

# Loss: ours vs verl's vanilla PPO-clip, at ratios away from 1 so that clipping is active
verl_policy_loss = get_policy_loss_fn("vanilla")


def verl_ppo_loss(log_probs, old_log_probs, advantages, mask, n_tokens, clip_low=0.2, clip_high=0.28, clip_c=3.0):
    cfg = OmegaConf.create({
        "clip_ratio": clip_low, "clip_ratio_low": clip_low, "clip_ratio_high": clip_high, "clip_ratio_c": clip_c,
        "global_batch_info": {"batch_num_tokens": n_tokens, "dp_size": 1},
    })
    loss, _ = verl_policy_loss(old_log_prob=old_log_probs, log_prob=log_probs, advantages=advantages, response_mask=mask, loss_agg_mode="token-mean", config=cfg)
    return loss


old = -torch.rand(mask.shape)
new = old + 0.5 * torch.randn(mask.shape)
args = (new, old, theirs, mask.float(), mask.sum().item())
print(f"PPO-clip loss: ours {ppo_clip_loss(*args):.6f} | verl with dual-clip off {verl_ppo_loss(*args, clip_c=1e9):.6f} "
      f"| verl with dual-clip (c=3) {verl_ppo_loss(*args):.6f}")

# Log-probs: verl's fused kernel (flash-attn's cross-entropy) vs our log_softmax + gather
logits = torch.randn(4, 64, 5000, device=DEVICE)
labels = torch.randint(0, logits.shape[-1], logits.shape[:-1], device=DEVICE)
ours_lp = torch.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
print("log-probs match:", torch.allclose(ours_lp, logprobs_from_logits(logits, labels), atol=1e-4))
''')

md(r'''
With dual-clip switched off, verl's loss is ours to the last digit. With it on (verl's default, `clip_ratio_c=3`), they differ only on tokens with a negative advantage whose ratio exceeded 3: verl caps their loss at 3·|A|, so a single token whose probability exploded can't dominate the update. The deliberately noisy ratios above produce a few such tokens; in real training, with ratios near 1, they're rare. Everything else matches to rounding.

Now the training loop with verl's pieces in place: the batch is a `DataProto`, and advantages and loss come from `core_algos`. `build_batch` produces exactly the batch layout verl's trainer works with, and we'll reuse it in every remaining stage.
''')

code(r'''
def build_batch(samples: list[dict], answers: list[int], uids: list) -> DataProto:
    """Score samples and lay them out the way verl's trainer expects."""
    rewards = torch.tensor([reward_fn(decode(s["response_ids"]), a) for s, a in zip(samples, answers)])
    batch = DataProto.from_dict(
        tensors=collate([s["prompt_ids"] for s in samples], [s["response_ids"] for s in samples]),
        non_tensors={"uid": np.array(uids, dtype=object), "answer": np.array(answers, dtype=object)},
        meta_info={"temperature": 1.0},
    )
    mask = batch.batch["response_mask"]
    scores = torch.zeros(mask.shape)
    scores[torch.arange(len(batch)), mask.sum(-1).clamp(min=1) - 1] = rewards
    batch.batch["token_level_rewards"] = scores
    rollout_log_probs = torch.zeros(mask.shape)
    for i, s in enumerate(samples):
        if s["log_probs"] is not None:
            rollout_log_probs[i, : len(s["log_probs"])] = torch.tensor(s["log_probs"])
    batch.batch["rollout_log_probs"] = rollout_log_probs
    return batch


def add_advantages(batch: DataProto) -> DataProto:
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_rewards"], response_mask=batch.batch["response_mask"], index=batch.non_tensor_batch["uid"],
    )
    batch.batch["advantages"] = adv
    return batch


def batch_metrics(batch: DataProto) -> dict:
    mask = batch.batch["response_mask"].bool()
    out = {"reward": batch.batch["token_level_rewards"].sum(-1).mean().item(), "response_len": mask.sum(-1).float().mean().item()}
    if "old_log_probs" in batch.batch.keys():
        out["logp_mismatch"] = (batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"])[mask].abs().mean().item()
    return out


def rl_step_verl(problems, n: int = 8) -> tuple[dict, DataProto]:
    timing = {}
    t0 = time.time()
    samples = vllm_generate(llm, [chat_prompt_ids(p["question"]) for p in problems], n)
    timing["gen"] = time.time() - t0
    batch = build_batch(samples, [problems[i // n]["answer"] for i in range(len(samples))], [i // n for i in range(len(samples))])
    t0 = time.time()
    batch.batch["old_log_probs"] = compute_log_probs(policy, batch.batch)
    timing["old_log_prob"] = time.time() - t0
    add_advantages(batch)
    t0 = time.time()
    stats = ppo_update(policy, optimizer, batch.batch, batch.batch["old_log_probs"], batch.batch["advantages"], loss_fn=verl_ppo_loss)
    timing["update"] = time.time() - t0
    t0 = time.time()
    push_weights(policy, llm)
    timing["sync"] = time.time() - t0
    return {**batch_metrics(batch), **stats, **{f"time/{k}": v for k, v in timing.items()}}, batch


verl_fn_history = []
for step in range(5):
    m, last_batch = rl_step_verl(step_rng.sample(TRAIN, 16))
    verl_fn_history.append(m)
    show(len(vllm_history) + step, m)
''')

md(r'''
Same numbers, same behaviour; we've only changed *whose* code computes them. That's the point of this stage: verl's algorithm code is not magic, it's these formulas plus many variants and a lot of care for edge cases (all-equal groups, dual-clip, loss aggregation modes, importance-sampling corrections between vLLM and trainer log-probs).

Everything still runs in **one process**. That's fine on one GPU for a 0.5B model, and impossible for a 70B model on 64 GPUs.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 5. Swap 4: separate processes with Ray

Right now one Python process owns the policy, the optimizer, the vLLM engine and the training loop. At scale these become different *roles* with different resources: the rollout engine wants many GPUs with lots of KV-cache memory, the trainer wants GPUs sharded with FSDP, and a reward model might want its own. [Ray](https://docs.ray.io) lets one driver script start Python objects in other processes, on any machine, and call their methods as if they were local:

* a **task** is a remote function call: `f.remote(x)` returns a future, `ray.get(future)` waits for the value;
* an **actor** is a remote object: a dedicated process that keeps state (a model, an optimizer) between calls;
* the **object store** passes arguments and results between processes without copying them through the driver more than needed.

First, free the GPU: the in-process vLLM engine and the policy go away, and Ray actors take over.
''')

code(r'''
llm.llm_engine.engine_core.shutdown()
del llm, policy, optimizer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU free: {gpu_free_gib():.0f} GiB")

import ray

ray.init(num_cpus=32, num_gpus=1, include_dashboard=False, runtime_env={"env_vars": {k: os.environ[k] for k in ("PYTHONPATH", "TOKENIZERS_PARALLELISM", "VLLM_LOGGING_LEVEL")}})


@ray.remote
def square(x):
    return x * x, os.getpid()


print("driver pid:", os.getpid(), "| tasks ran in:", ray.get([square.remote(i) for i in range(4)]))
print("cluster resources:", ray.cluster_resources())
''')

md(r'''
Now the two roles become actors. The code inside them is the code you already wrote: `vllm_generate`, `compute_log_probs`, `ppo_update`, `save_for_inference`. Ray ships these functions to the worker processes.

`num_gpus=0.3` and `num_gpus=0.6` are **scheduling labels**, not memory limits: they let Ray place both actors on one GPU (0.3 + 0.6 ≤ 1). Memory is still governed by vLLM's `gpu_memory_utilization` and by what the trainer allocates. Putting rollout and training on the same GPUs is verl's **colocated** (hybrid engine) layout. The alternative, separate GPU pools for each role, is what verl calls *disaggregated*, and is what you need for asynchronous training.
''')

code(r'''
@ray.remote(num_gpus=0.3)
class RolloutWorker:
    def __init__(self, model: str, gpu_memory_utilization: float = 0.3):
        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=4096, seed=SEED)

    def generate(self, prompts, n, max_tokens=256, temperature=1.0):
        return vllm_generate(self.llm, prompts, n, max_tokens, temperature)

    def load_weights(self, path: str) -> None:
        self.llm.collective_rpc("reload_weights", kwargs={"weights_path": path})
        self.llm.reset_prefix_cache()

    def shutdown(self) -> None:
        self.llm.llm_engine.engine_core.shutdown()

    def info(self) -> dict:
        return {"role": "rollout", "pid": os.getpid(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


@ray.remote(num_gpus=0.6)
class TrainerWorker:
    def __init__(self, model: str, lr: float):
        self.policy = load_policy(model)
        self.optimizer = make_optimizer(self.policy, lr)

    def compute_log_probs(self, batch: DataProto) -> torch.Tensor:
        return compute_log_probs(self.policy, batch.batch)

    def update(self, batch: DataProto) -> dict:
        return ppo_update(self.policy, self.optimizer, batch.batch, batch.batch["old_log_probs"], batch.batch["advantages"], loss_fn=verl_ppo_loss)

    def save_weights(self) -> str:
        return save_for_inference(self.policy)

    def info(self) -> dict:
        return {"role": "trainer", "pid": os.getpid(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


rollout = RolloutWorker.remote(MODEL)
trainer = TrainerWorker.remote(MODEL, LR)
for info in ray.get([rollout.info.remote(), trainer.info.remote()]):
    print(info)
print("\nprocesses on the GPU:\n" + subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"], capture_output=True, text=True).stdout)
''')

md(r'''
The driver loop is the §4 loop with `.remote(...)` and `ray.get(...)` added. The driver itself never touches the GPU: it builds batches and computes advantages (cheap tensor maths), exactly like verl's `RayPPOTrainer.fit()`.

Weight sync now crosses a process boundary, and the control/data split from §2.1 becomes visible: the **path** travels through Ray, the **weights** travel through `/dev/shm`.

These actors start from the base model (the §2 policy is gone), so the reward starts low again.
''')

code(r'''
def ray_rl_step(problems, n: int = 8) -> tuple[dict, DataProto, list[dict]]:
    timing = {}
    t0 = time.time()
    samples = ray.get(rollout.generate.remote([chat_prompt_ids(p["question"]) for p in problems], n))
    timing["gen"] = time.time() - t0
    batch = build_batch(samples, [problems[i // n]["answer"] for i in range(len(samples))], [i // n for i in range(len(samples))])
    t0 = time.time()
    batch.batch["old_log_probs"] = ray.get(trainer.compute_log_probs.remote(batch))
    timing["old_log_prob"] = time.time() - t0
    add_advantages(batch)
    t0 = time.time()
    stats = ray.get(trainer.update.remote(batch))
    timing["update"] = time.time() - t0
    t0 = time.time()
    ray.get(rollout.load_weights.remote(ray.get(trainer.save_weights.remote())))
    timing["sync"] = time.time() - t0
    return {**batch_metrics(batch), **stats, **{f"time/{k}": v for k, v in timing.items()}}, batch, samples


ray_history = []
for step in range(10):
    m, last_batch, last_samples = ray_rl_step(step_rng.sample(TRAIN, 16))
    ray_history.append(m)
    show(step, m)
plot_histories({"§2 vLLM, one process": vllm_history, "§5 Ray actors": ray_history})
''')

md(r'''
What Ray bought us, even on one GPU: the roles are independent processes with their own lifetimes and resources, and the driver is a plain Python loop. The same code would place `RolloutWorker` and `TrainerWorker` on different machines if the cluster had them.

What it **didn't** buy yet: each role is a single process. A 70B policy doesn't fit on one GPU, and one vLLM engine can't serve thousands of samples quickly enough. Each role needs to become *N* processes working together.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 6. Swap 5: many workers per role

### 6.1 Your own worker group

A worker group is *N* actors of the same class plus a rule for each method that says how to split the input and combine the output. Two rules cover most of verl:

* **one-to-all**: send the same arguments to every worker and return all results. Used for `init_model`, `save_checkpoint`, …
* **data-parallel**: `chunk` the `DataProto` into *N* pieces, give one piece to each worker, and `concat` the results in order. Used for `compute_log_prob`, `update_actor`, …

On one GPU we can't run *N* GPU trainers meaningfully, so the workers here are CPU actors that **score** responses. It's a real job (verl runs reward computation in parallel workers too) and cheap enough to demonstrate the pattern.

The workers also join a `torch.distributed` process group (`gloo`, the CPU backend) and run an `all_reduce`. That's the other half of multi-worker training: **Ray** creates the processes and calls their methods (control), while **`torch.distributed`** moves tensors directly between them (data). FSDP's all-gathers and gradient reductions are exactly such collectives, over NCCL between GPUs.
''')

code(r'''
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@ray.remote(num_cpus=1)
class ScoreWorker:
    def __init__(self, rank: int, world_size: int, master_port: int):
        self.rank, self.world_size = rank, world_size
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def init_process_group(self) -> int:
        torch.distributed.init_process_group("gloo", rank=self.rank, world_size=self.world_size)
        return self.rank

    def all_reduce(self, value: float) -> float:
        t = torch.tensor([value])
        torch.distributed.all_reduce(t)  # sum across all workers, directly between the processes
        return t.item()

    def score(self, data: DataProto) -> DataProto:
        texts = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        rewards = torch.tensor([reward_fn(t, int(a)) for t, a in zip(texts, data.non_tensor_batch["answer"])])
        return DataProto.from_dict(tensors={"reward": rewards}, non_tensors={"scored_by": np.full(len(data), self.rank, dtype=object)})


class MyWorkerGroup:
    def __init__(self, workers):
        self.workers = workers

    def one_to_all(self, method: str, *args):
        return ray.get([getattr(w, method).remote(*args) for w in self.workers])

    def data_parallel(self, method: str, data: DataProto) -> DataProto:
        chunks = data.chunk(len(self.workers))
        return DataProto.concat(ray.get([getattr(w, method).remote(c) for w, c in zip(self.workers, chunks)]))


port = free_port()
my_wg = MyWorkerGroup([ScoreWorker.remote(rank, 2, port) for rank in range(2)])
print("ranks joined:", my_wg.one_to_all("init_process_group"))
print("each worker contributes rank + 1; all_reduce returns the sum on every worker:",
      ray.get([w.all_reduce.remote(float(r + 1)) for r, w in enumerate(my_wg.workers)]))

scored = my_wg.data_parallel("score", last_batch)
reference = last_batch.batch["token_level_rewards"].sum(-1)
print(f"scored {len(scored)} rows; matches the driver's rewards: {torch.equal(scored.batch['reward'], reference)}")
print("rows per worker:", pd.Series(scored.non_tensor_batch["scored_by"]).value_counts().sort_index().to_dict())
''')

md(r'''
### 6.2 verl's `RayWorkerGroup`

verl packages exactly this pattern. You subclass `Worker`, and **decorate each method with its dispatch rule**. `RayWorkerGroup` then creates one actor per slot in a `RayResourcePool`, sets `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` in each actor's environment, and gives the group a method of the same name that splits, calls and gathers for you:

| Your §6.1 code | verl |
|---|---|
| `my_wg.one_to_all("init_process_group")` | `@register(dispatch_mode=Dispatch.ONE_TO_ALL)` → `wg.init_process_group()` |
| `my_wg.data_parallel("score", batch)` | `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` → `wg.score(batch)` |
| `rank`, `world_size`, `master_port` passed by hand | read from the environment verl sets: `self.rank`, `self.world_size` |
| `[ScoreWorker.remote(...) for rank in range(2)]` | `RayResourcePool(process_on_nodes=[2])` + `RayClassWithInitArgs` |

A resource pool reserves one Ray *placement-group bundle* per worker, and with `use_gpu=True` each bundle is **one whole GPU**. That's why the demonstration uses CPU workers (`use_gpu=False`): on one GPU, a GPU pool could hold only a single worker.
''')

code(r'''
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


class VerlScoreWorker(Worker):
    def __init__(self, model: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_process_group(self) -> int:
        torch.distributed.init_process_group("gloo", rank=self.rank, world_size=self.world_size)  # MASTER_ADDR/PORT set by verl
        return self.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def all_reduce_rank(self) -> float:
        t = torch.tensor([float(self.rank + 1)])
        torch.distributed.all_reduce(t)
        return t.item()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def score(self, data: DataProto) -> DataProto:
        texts = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        rewards = torch.tensor([reward_fn(t, int(a)) for t, a in zip(texts, data.non_tensor_batch["answer"])])
        return DataProto.from_dict(tensors={"reward": rewards}, non_tensors={"scored_by": np.full(len(data), self.rank, dtype=object)})


pool = RayResourcePool(process_on_nodes=[2], use_gpu=False, max_colocate_count=4)
verl_wg = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(VerlScoreWorker), model=MODEL))
print("world size:", verl_wg.world_size, "| ranks:", verl_wg.init_process_group(), "| all_reduce:", verl_wg.all_reduce_rank())

scored_verl = verl_wg.score(last_batch)
print(f"verl worker group: same rewards as ours: {torch.equal(scored_verl.batch['reward'], scored.batch['reward'])}")
print("rows per worker:", pd.Series(scored_verl.non_tensor_batch["scored_by"]).value_counts().sort_index().to_dict())
''')

md(r'''
That is verl's whole **single-controller** architecture. Its training workers are this pattern with heavier insides:

* `ActorRolloutRefWorker` (`verl/workers/engine_workers.py`) holds an FSDP- or Megatron-sharded policy, an optimizer, and (in colocated mode) a handle to a vLLM/SGLang replica on the same GPU. Its methods `compute_log_prob` and `update_actor` are data-parallel, and `init_model` is one-to-all.
* The driver, `RayPPOTrainer.fit()`, is your §5 loop: call the worker groups in order, attach results to the `DataProto` with `union`, compute advantages on the driver.

Scaling from 1 to 64 GPUs changes the resource pool, not the loop. Clean up the CPU workers before moving on.
''')

code(r'''
for w in my_wg.workers + list(verl_wg.workers):
    ray.kill(w)
print("score workers stopped; rollout and trainer actors still running")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 7. Multi-turn: agents that call tools

So far one prompt in, one response out. Agents don't work like that: the model writes a tool call, something executes it, the result goes back into the context, and the model continues. The training data for one attempt becomes a single sequence where some tokens came from the model and some from the environment:

```
[ prompt ][ model: <tool_call>... ][ env: <tool_response>... ][ model: ... \boxed{N} ]
            response_mask = 1          response_mask = 0            response_mask = 1
```

Only the model's tokens are trained. Tool outputs stay in the sequence as context (the model's later tokens were conditioned on them), with mask 0.

### 7.1 Your own token-level tool loop

A harder task makes the tool worth calling: `abc × def − ghij`, which a 0.5B model can't multiply in its head. The calculator is `safe_eval` from §1, and its schema goes into the prompt through the chat template's `tools=` argument. Qwen's template renders it into the system prompt along with instructions to emit `<tool_call>{json}</tool_call>`.
''')

code(r'''
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression with numbers, + - * / and parentheses.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "The arithmetic expression, e.g. '123 * 456 - 789'."}},
            "required": ["expression"],
        },
    },
}
TOOL_INSTRUCTION = r"Use the calculate tool for the arithmetic, then give the final answer as \boxed{N}."
HARD_TRAIN, HARD_EVAL = make_problems(2000, seed=3, hard=True), make_problems(100, seed=4, hard=True)


def tool_prompt_ids(question: str) -> list[int]:
    return tokenizer.apply_chat_template([{"role": "user", "content": f"{question} {TOOL_INSTRUCTION}"}], tools=[CALC_TOOL], add_generation_prompt=True)


print(tokenizer.decode(tool_prompt_ids(HARD_TRAIN[0]["question"])))
''')

md(r'''
The environment's tokens must be exactly what the chat template would produce for a `tool` message, so that training sees the same format as inference. We build them by hand and **check against the template**: render a conversation that ends in a tool result and a new generation prompt, and verify it ends with our observation text.
''')

code(r'''
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def run_tool_call(raw: str) -> str:
    """Parse one model-written tool call and execute it. Errors go back to the model as text."""
    try:
        call = json.loads(raw)
        args = call["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        if call.get("name") != "calculate":
            return f"Error: unknown tool {call.get('name')!r}"
        value = safe_eval(str(args["expression"]))
        return str(int(value)) if float(value).is_integer() else f"{value:.6g}"
    except Exception as exc:
        return f"Error: {exc}"


def observation_text(results: list[str]) -> str:
    """What Qwen's chat template appends after the model's <|im_end|>: the tool results, then a new assistant turn."""
    return "\n<|im_start|>user" + "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results) + "<|im_end|>\n<|im_start|>assistant\n"


check = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is 2*3?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "calculate", "arguments": {"expression": "2*3"}}}]},
        {"role": "tool", "content": "6"},
    ],
    tools=[CALC_TOOL], add_generation_prompt=True, tokenize=False,
)
print("our observation tokens match the chat template:", check.endswith("<|im_end|>" + observation_text(["6"])))
print(repr(observation_text(["6"])))
''')

code(r'''
def run_episodes(problems, n: int, max_turns: int = 3, max_tokens: int = 256) -> list[dict]:
    """Multi-turn rollouts: generate, run tool calls, append observations, repeat. All in token space."""
    episodes = [
        {"question": p["question"], "answer": p["answer"], "group": g, "prompt_ids": tool_prompt_ids(p["question"]),
         "response_ids": [], "mask": [], "log_probs": [], "tool_calls": 0, "tool_results": [], "done": False}
        for g, p in enumerate(problems) for _ in range(n)
    ]
    for turn in range(max_turns):
        active = [e for e in episodes if not e["done"]]
        if not active:
            break
        outs = ray.get(rollout.generate.remote([e["prompt_ids"] + e["response_ids"] for e in active], 1, max_tokens))
        for e, out in zip(active, outs):
            e["response_ids"] += out["response_ids"]
            e["mask"] += [1] * len(out["response_ids"])
            e["log_probs"] += out["log_probs"]
            e["last_text"] = tokenizer.decode(out["response_ids"])
            calls = TOOL_CALL.findall(e["last_text"])
            finished = bool(out["response_ids"]) and out["response_ids"][-1] in STOP_IDS
            if not calls or not finished or turn == max_turns - 1:
                e["done"] = True
                continue
            results = [run_tool_call(c) for c in calls]
            obs = tokenizer.encode(observation_text(results), add_special_tokens=False)
            e["response_ids"] += obs
            e["mask"] += [0] * len(obs)
            e["log_probs"] += [0.0] * len(obs)
            e["tool_calls"] += len(calls)
            e["tool_results"].append(results)
    for e in episodes:
        e["reward"] = reward_fn(e["last_text"], e["answer"])
    return episodes


def episodes_to_batch(episodes: list[dict]) -> DataProto:
    batch = DataProto.from_dict(
        tensors=collate([e["prompt_ids"] for e in episodes], [e["response_ids"] for e in episodes], [e["mask"] for e in episodes]),
        non_tensors={"uid": np.array([e["group"] for e in episodes], dtype=object)},
    )
    R = batch.batch["responses"].shape[1]
    scores, rollout_lp = torch.zeros(len(episodes), R), torch.zeros(len(episodes), R)
    for i, e in enumerate(episodes):
        scores[i, max(len(e["response_ids"]), 1) - 1] = e["reward"]
        rollout_lp[i, : len(e["log_probs"])] = torch.tensor(e["log_probs"])
    batch.batch["token_level_rewards"] = scores
    batch.batch["rollout_log_probs"] = rollout_lp
    return batch


episodes = run_episodes(HARD_TRAIN[:4], n=4)
for e in episodes[:2]:
    print("═" * 100, f"\nreward {e['reward']} | tool calls {e['tool_calls']} | answer {e['answer']}")
    print(tokenizer.decode(e["response_ids"]))
''')

code(r'''
import html as _html

from IPython.display import HTML, display


def show_episode(e: dict) -> None:
    """Green: trained (model) tokens. Orange: environment tokens, context only."""
    spans, cur, kind = [], [], None
    for t, m in zip(e["response_ids"], e["mask"]):
        if m != kind and cur:
            spans.append((kind, cur))
            cur = []
        kind = m
        cur.append(t)
    spans.append((kind, cur))
    style = {1: "background:#b9f0b4", 0: "background:#ffd59e"}
    body = "".join(f'<span style="{style[k]}">{_html.escape(tokenizer.decode(t)).replace(chr(10), "↵<br>")}</span>' for k, t in spans)
    display(HTML(f"<b>reward {e['reward']}, {sum(e['mask'])} trained / {len(e['mask'])} response tokens</b>"
                 f'<div style="font-family:monospace;font-size:12px;border:1px solid #ccc;padding:6px">{body}</div>'))


show_episode(max(episodes, key=lambda e: (e["tool_calls"] > 0, e["reward"])))
''')

md(r'''
Training on these episodes needs **no change to the trainer**: the batch has the same keys, and `response_mask` already zeroes the tool tokens, so the loss ignores them. That's why verl's actor doesn't know or care whether a row came from one turn or ten.
''')

code(r'''
def tool_rl_step(problems, n: int = 8) -> tuple[dict, DataProto, list[dict]]:
    timing = {}
    t0 = time.time()
    episodes = run_episodes(problems, n)
    timing["gen"] = time.time() - t0
    batch = episodes_to_batch(episodes)
    batch.batch["old_log_probs"] = ray.get(trainer.compute_log_probs.remote(batch))
    add_advantages(batch)
    t0 = time.time()
    stats = ray.get(trainer.update.remote(batch))
    timing["update"] = time.time() - t0
    t0 = time.time()
    ray.get(rollout.load_weights.remote(ray.get(trainer.save_weights.remote())))
    timing["sync"] = time.time() - t0
    m = {**batch_metrics(batch), **stats, "tool_use": float(np.mean([e["tool_calls"] > 0 for e in episodes])), **{f"time/{k}": v for k, v in timing.items()}}
    return m, batch, episodes


tool_history, all_episodes = [], []
for step in range(6):
    m, _, eps = tool_rl_step(step_rng.sample(HARD_TRAIN, 16))
    tool_history.append(m)
    all_episodes += eps
    show(step, m)
    print(f"        tool use {m['tool_use']:.2f}")
''')

md(r'''
### 7.2 The same thing in verl: `ToolAgentLoop`

verl ships this loop as `ToolAgentLoop` (`verl/experimental/agent_loop/tool_agent_loop.py`): a state machine `PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → …` that works on token IDs, parses tool calls with a `ToolParser` (`hermes` for Qwen), runs tools, and appends their tokens with `response_mask = 0`. Here is its generation step; compare it with `run_episodes`:
''')

code(r'''
import inspect

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

src = inspect.getsource(ToolAgentLoop._handle_generating_state).splitlines()
print("\n".join(line for line in src if line.strip() and not line.strip().startswith("#"))[:4000])
''')

md(r'''
| `run_episodes` (§7.1) | verl `ToolAgentLoop` |
|---|---|
| `tool_prompt_ids` (chat template with `tools=`) | `_handle_pending_state` → `apply_chat_template(..., tools=schemas)` |
| `rollout.generate.remote(prompt_ids + response_ids)` | `server_manager.generate(prompt_ids=agent_data.prompt_ids)` |
| `mask += [1] * len(response)` | `agent_data.response_mask += [1] * len(response_ids)` |
| `TOOL_CALL.findall` + `run_tool_call` | `tool_parser.extract_tool_calls` + `_call_tool` |
| `observation_text` → tokens, mask 0 | `_handle_processing_tools_state`: tool messages → template → tokens, mask 0 |
| `max_turns` | `multi_turn.max_assistant_turns` |
| `safe_eval` as the tool | a function decorated with verl's `@function_tool`, or a `BaseTool` subclass |

To use it, you express the agent **in verl's terms**: tools as `@function_tool` functions or `BaseTool` classes, the conversation as a verl dataset row, limits as verl config. For an agent you design for RL from the start, that's a fine deal, and §9.1 runs exactly this with plain verl.

### 7.3 A real agent

Now take an agent that already exists. Calc-X's agent (`examples/calc_x/calc_agent.py`) is ordinary application code:
''')

code(r'''
print((REPO / "examples/calc_x/calc_agent.py").read_text())
''')

md(r'''
Try to put this inside `run_episodes`, and every assumption breaks:

* **It speaks HTTP, not tokens.** It calls an OpenAI-compatible `/chat/completions` endpoint and gets text and parsed `tool_calls` back. There is no `generate(prompt_ids)` to call.
* **It owns the control flow.** AutoGen decides when to call the model, how to format tool results, and when to stop. With `reflect_on_tool_use=True` its last call even uses a different system prompt (no tools).
* **Its tools are separate processes.** The calculator is an MCP server it starts over stdio. Other agents run tests in Docker, browse the web, or call internal services.
* **It computes its own reward** and reports it at the end.

You could rewrite it as a verl agent loop, but then you are training *the rewrite*, and every future change to the real agent has to be ported. The alternative is to leave it alone and record what it does. But from where? Its logs hold text, and training needs **token IDs and log-probs**. Turning logged text back into tokens *usually* works, and that's the trap: when it doesn't, nothing tells you. Here are two checks on the episodes we just generated, where we *do* know the true tokens:

1. **Text round-trip**: decode each model turn and re-encode it. At temperature 1 the model sometimes samples a non-canonical tokenization (for example `" 1" + "2"` instead of `" 12"`), which re-encoding "fixes" into tokens the model never produced.
2. **Re-rendering from an OpenAI-style log**: rebuild the conversation as messages with parsed `tool_calls` (what an HTTP log contains), apply the chat template, and compare with the real token sequence. The template re-serialises the tool-call JSON its own way.
''')

code(r'''
def model_turns(e: dict) -> list[list[int]]:
    turns, cur = [], []
    for t, m in zip(e["response_ids"], e["mask"]):
        if m:
            cur.append(t)
        elif cur:
            turns.append(cur)
            cur = []
    return turns + ([cur] if cur else [])


def rerender(e: dict) -> list[int] | None:
    """Rebuild the episode the way an HTTP log would describe it, and re-apply the chat template."""
    msgs = [{"role": "user", "content": f"{e['question']} {TOOL_INSTRUCTION}"}]
    turns = model_turns(e)
    for i, turn in enumerate(turns):
        text = tokenizer.decode(turn).removesuffix("<|im_end|>")
        calls = TOOL_CALL.findall(text)
        if i < len(turns) - 1 and calls:
            try:
                tool_calls = [{"type": "function", "function": {"name": json.loads(c)["name"], "arguments": json.loads(c)["arguments"]}} for c in calls]
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
            msgs.append({"role": "assistant", "content": text.split("<tool_call>")[0].strip(), "tool_calls": tool_calls})
            msgs += [{"role": "tool", "content": r} for r in e["tool_results"][i]]
        else:
            msgs.append({"role": "assistant", "content": text})
    return tokenizer.apply_chat_template(msgs, tools=[CALC_TOOL])


turns = [t for e in all_episodes for t in model_turns(e)]
roundtrip_bad = [t for t in turns if tokenizer.encode(tokenizer.decode(t), add_special_tokens=False) != t]
print(f"1. text round-trip: {len(roundtrip_bad)} of {len(turns)} model turns ({len(roundtrip_bad) / max(len(turns), 1):.1%}) re-encode to different tokens")

with_tools = [e for e in all_episodes if e["tool_calls"] > 0]
results = []
for e in with_tools:
    actual = e["prompt_ids"] + e["response_ids"]
    rebuilt = rerender(e)
    results.append(None if rebuilt is None else rebuilt[: len(actual)] == actual)
n_ok = sum(r is True for r in results)
print(f"2. re-rendered tool-using episodes: {n_ok} of {len(results)} reproduce the exact tokens "
      f"({sum(r is None for r in results)} had tool calls that don't even parse as JSON)")
example = next((e for e, r in zip(with_tools, results) if r is False), None)
if example is not None:
    actual, rebuilt = example["prompt_ids"] + example["response_ids"], rerender(example)
    i = next(i for i, (a, b) in enumerate(zip(actual, rebuilt)) if a != b)
    print(f"\nfirst difference at token {i}:\n  model produced: {tokenizer.decode(actual[i - 8: i + 12])!r}\n  log re-renders: {tokenizer.decode(rebuilt[i - 8: i + 12])!r}")
''')

md(r'''
With this simple agent and a model that follows its template closely, mismatches are rare: around 1% of episodes. But every one is a training example where the loss pushes on tokens the model never chose, with log-probs that don't belong to them, and nothing flags it. With messier agents (frameworks that reformat messages, models that don't follow the template exactly, long multi-turn traces) the rate goes up, and a real agent's logs often don't contain the full prompt the model saw at all.

So there's exactly one place where a real agent's training data can be captured correctly: **at the HTTP boundary between the agent and the inference engine**, where the engine still knows the exact token IDs and log-probs. Let's build that.
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 8. Build it yourself: a token-recording gateway

The idea in one sentence: give each rollout its **own base URL**, and put a proxy there that forwards OpenAI requests to vLLM, asks vLLM to return token IDs and log-probs, and records them under that rollout's ID. The agent needs no changes except its base URL.

We need vLLM as an HTTP server now. Two flags matter for RL:

* `--generation-config vllm` ignores the model's chat sampling defaults from §2 (top-k 20, top-p 0.8, repetition penalty 1.1). Without it, every request that doesn't set them explicitly samples from the wrong distribution.
* `VLLM_SERVER_DEV_MODE=1` exposes `/collective_rpc` and `/reset_prefix_cache`, the HTTP versions of the weight-sync calls from §2.1. They're development endpoints: the server listens on 127.0.0.1 only.

Calc-X questions are harder than our arithmetic, and the 0.5B model almost never solves them, which would leave every GRPO group with zero advantage. From here on we use the 1.5B model the Calc-X recipe trains, so some attempts succeed.

First, stop Ray: its actors hold the GPU.
''')

code(r'''
ray.get(rollout.shutdown.remote())  # stop vLLM's engine process before its parent actor goes away
ray.shutdown()
time.sleep(5)
print(f"GPU free: {gpu_free_gib():.0f} GiB")

CALC_X_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # same tokenizer as the 0.5B model
VLLM_PORT = 8110
VLLM_ROOT = f"http://127.0.0.1:{VLLM_PORT}"
start_service("vllm", [
    str(VENV_BIN / "vllm"), "serve", CALC_X_MODEL, "--host", "127.0.0.1", "--port", str(VLLM_PORT),
    "--gpu-memory-utilization", "0.3", "--max-model-len", "8192", "--seed", str(SEED),
    "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--generation-config", "vllm",
], env={"VLLM_SERVER_DEV_MODE": "1"})
wait_until_healthy("vllm", f"{VLLM_ROOT}/health")
''')

md(r'''
The whole gateway: one route that proxies chat completions and records tokens, one that accepts the agent's reward event. It rewrites three things in every request: the model name (the agent says `"auto"`), `temperature=1.0` (sample from the policy), and the two flags that make vLLM return `prompt_token_ids`, `token_ids` and per-token `logprobs`.
''')

code(r'''
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

RECORDS: dict[str, dict] = {}
gateway = FastAPI()


def record(rollout_id: str) -> dict:
    return RECORDS.setdefault(rollout_id, {"calls": [], "reward": None})


@gateway.get("/healthz")
async def healthz():
    return {"ok": True}


@gateway.post("/rollout/{rollout_id}/v1/chat/completions")
async def chat_completions(rollout_id: str, request: Request):
    body = await request.json()
    body.update(model=CALC_X_MODEL, temperature=1.0, top_p=1.0, logprobs=True, return_token_ids=True, stream=False)
    async with httpx.AsyncClient(timeout=300) as client:
        upstream = await client.post(f"{VLLM_ROOT}/v1/chat/completions", json=body)
    data = upstream.json()
    if upstream.status_code == 200:
        choice = data["choices"][0]
        record(rollout_id)["calls"].append({
            "prompt_ids": data.get("prompt_token_ids") or choice.get("prompt_token_ids"),
            "response_ids": choice["token_ids"],
            "log_probs": [t["logprob"] for t in choice["logprobs"]["content"]],
        })
    return JSONResponse(data, status_code=upstream.status_code)


@gateway.post("/rollout/{rollout_id}/events")
async def events(rollout_id: str, request: Request):
    event = await request.json()
    if event.get("event_type") == "reward":
        record(rollout_id)["reward"] = float(event["data"]["value"])
    return {"ok": True}


GATEWAY_PORT = 8291
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}"
gateway_server = uvicorn.Server(uvicorn.Config(gateway, host="127.0.0.1", port=GATEWAY_PORT, log_level="warning"))
threading.Thread(target=gateway_server.run, daemon=True).start()
while not gateway_server.started:
    time.sleep(0.1)
print("gateway up:", httpx.get(f"{GATEWAY_URL}/healthz").json())
''')

md(r'''
And a "controller": start one agent process per rollout with the environment variables `calc_agent.py` reads, and wait for all of them. The agent posts its reward to `AGL_EVENT_URL` and talks to the model through `AGL_OPENAI_BASE_URL`, and both point at our gateway.
''')

code(r'''
import uuid

(WORK / "agents").mkdir(exist_ok=True)


def launch_agent(rollout_id: str, question: str, result) -> subprocess.Popen:
    env = {
        **os.environ, "QUESTION": question, "RESULT": str(result), "AGL_KEY": "not-checked-by-our-gateway",
        "AGL_OPENAI_BASE_URL": f"{GATEWAY_URL}/rollout/{rollout_id}/v1",
        "AGL_EVENT_URL": f"{GATEWAY_URL}/rollout/{rollout_id}/events",
    }
    with open(WORK / "agents" / f"{rollout_id}.log", "w") as log:
        return subprocess.Popen(
            [sys.executable, "-c", "import asyncio, calc_agent; asyncio.run(calc_agent.Agent().run())"],
            cwd=REPO / "examples/calc_x", env=env, stdout=log, stderr=subprocess.STDOUT,
        )


def run_agents(jobs: list[tuple[str, str, object]], timeout: float = 300) -> dict[str, int]:
    procs = {rid: launch_agent(rid, q, r) for rid, q, r in jobs}
    deadline = time.time() + timeout
    for p in procs.values():
        try:
            p.wait(max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    return {rid: p.returncode for rid, p in procs.items()}


calc_train = pd.read_parquet(CALC_X_DATA / "train.parquet")
row = calc_train.iloc[0]
rid = f"demo-{uuid.uuid4().hex[:6]}"
print("exit code:", run_agents([(rid, row["question"], row["result"])])[rid])
rec = RECORDS[rid]
print(f"question: {row['question']}\nreward: {rec['reward']} (truth {row['result']!r})")
for i, call in enumerate(rec["calls"]):
    print(f"call {i}: {len(call['prompt_ids'])} prompt tokens, {len(call['response_ids'])} response tokens, {len(call['log_probs'])} log-probs")
    print("   ", repr(tokenizer.decode(call["response_ids"])[:200]))
''')

md(r'''
The unmodified AutoGen agent ran, and we hold **exactly** the tokens vLLM sampled for every one of its model calls, with their log-probs, and its reward. That's enough to train on.

The batch is built **per model call** ("transition" level): each call is one row, with its own prompt (whatever the agent sent) and response. That's robust to anything the agent does to its context between calls, like the system-prompt change from `reflect_on_tool_use`. Two consequences:

* All rows of a rollout share the rollout's reward.
* The GRPO advantage must be computed **per rollout**, not per row. Otherwise a rollout that made 3 calls counts 3 times in its group's mean and std. So we compute advantages over rollouts and then copy each rollout's advantage onto its rows (Agent Lightning's `compute_rollout_level_advantage` does the same).

Training then uses the same functions as before, and weight sync goes over HTTP.
''')

code(r'''
def push_weights_http(model) -> None:
    path = save_for_inference(model)
    httpx.post(f"{VLLM_ROOT}/collective_rpc", json={"method": "reload_weights", "kwargs": {"weights_path": path}}, timeout=600).raise_for_status()
    httpx.post(f"{VLLM_ROOT}/reset_prefix_cache", timeout=60).raise_for_status()


def gateway_batch(rollouts: list[tuple[str, int]]) -> tuple[DataProto, dict]:
    """rollouts: (rollout_id, question_index). One row per recorded model call; advantages per rollout."""
    rewards = torch.tensor([RECORDS.get(rid, {}).get("reward") or 0.0 for rid, _ in rollouts])
    rollout_adv = grpo_advantages(rewards, [q for _, q in rollouts])
    prompts, responses, lps, row_adv, row_reward, uids = [], [], [], [], [], []
    for (rid, q), adv, r in zip(rollouts, rollout_adv.tolist(), rewards.tolist()):
        for call in RECORDS.get(rid, {}).get("calls", []):
            prompts.append(call["prompt_ids"])
            responses.append(call["response_ids"])
            lps.append(call["log_probs"])
            row_adv.append(adv)
            row_reward.append(r)
            uids.append(q)
    batch = DataProto.from_dict(tensors=collate(prompts, responses), non_tensors={"uid": np.array(uids, dtype=object)})
    mask = batch.batch["response_mask"]
    batch.batch["advantages"] = torch.tensor(row_adv)[:, None] * mask
    scores, rollout_lp = torch.zeros(mask.shape), torch.zeros(mask.shape)
    for i, (resp, lp) in enumerate(zip(responses, lps)):
        scores[i, max(len(resp), 1) - 1] = row_reward[i]
        rollout_lp[i, : len(lp)] = torch.tensor(lp)
    batch.batch["token_level_rewards"] = scores
    batch.batch["rollout_log_probs"] = rollout_lp
    return batch, {"rollout_reward": rewards.mean().item(), "rows": len(batch), "rollouts_without_calls": sum(not RECORDS.get(rid, {}).get("calls") for rid, _ in rollouts)}


policy = load_policy(CALC_X_MODEL)
optimizer = make_optimizer(policy, LR)
calc_rng = random.Random(SEED)
gateway_history = []
for step in range(3):
    questions = calc_train.iloc[calc_rng.sample(range(len(calc_train)), 8)]
    rollouts = [(f"step{step}-q{qi}-{uuid.uuid4().hex[:6]}", qi) for qi in range(len(questions)) for _ in range(4)]
    t0 = time.time()
    run_agents([(rid, questions.iloc[qi]["question"], questions.iloc[qi]["result"]) for rid, qi in rollouts])
    t_gen = time.time() - t0
    batch, info = gateway_batch(rollouts)
    batch.batch["old_log_probs"] = compute_log_probs(policy, batch.batch)
    stats = ppo_update(policy, optimizer, batch.batch, batch.batch["old_log_probs"], batch.batch["advantages"], mini_batch_size=len(batch), loss_fn=verl_ppo_loss)
    push_weights_http(policy)
    m = {**batch_metrics(batch), **stats, "time/gen": t_gen}
    gateway_history.append({**m, **info})
    print(f"step {step} | rollout reward {info['rollout_reward']:.3f} | {info['rows']} rows from {len(rollouts)} rollouts "
          f"({info['rollouts_without_calls']} without model calls) | logp mismatch {m['logp_mismatch']:.4f} | grad_norm {m['grad_norm']:.3f} | agents {t_gen:.0f}s")
''')

md(r'''
**A real, unmodified agent is now training**, through about 60 lines of gateway and 20 lines of process launching. That's the core idea of Agent Lightning, and you now know it from the inside.

Here's what this toy version doesn't handle. Each item is something you hit as soon as you run it for real, and each is what Agent Lightning's Gateway, Controller and trainer integration actually consist of:

| Problem | Our toy | Agent Lightning |
|---|---|---|
| Where rollouts live | a Python dict in the notebook | a rollout store with a state machine (`queuing → running → succeeded/failed`), queried over HTTP |
| Starting agents | `Popen` in a loop, all at once | Controller: reconcile loop, concurrency limit, per-rollout timeouts, local processes **or one Kubernetes Job per rollout** |
| Failures | a crashed agent is just "reward 0" | explicit failed state; failed rollouts are excluded rather than trained as zero-reward data |
| Auth and isolation | none, localhost only | API keys; agents only reach their own rollout's URLs |
| Which vLLM to call | one hard-coded URL | model registry, so the trainer can register *its* replicas each step |
| Weight-sync timing | we just don't launch agents during an update | pause and drain the proxy, abort in-flight requests, resume after sync |
| Batch building | one row per call | `transition` **or** `trajectory`: merge consecutive calls into one row when the token prefix is continuous (fewer, longer rows, like §7), with the prefix check from the walkthrough's §4.7 |
| Loss weighting | rows weighted by token count | rollout-level advantage, plus a `per_rollout_mean` loss so long rollouts don't dominate |
| Async training | none | rollouts keep running across steps; stale ones carried over |
| The trainer | our loop | `AgentLightningRayPPOTrainer`: verl's trainer with `generate_sequences` replaced by this rollout path |

Tear down the toy and run the real things.
''')

code(r'''
gateway_server.should_exit = True
stop_service("vllm")
del policy, optimizer
gc.collect()
torch.cuda.empty_cache()
time.sleep(5)
print(f"GPU free: {gpu_free_gib():.0f} GiB")
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 9. The real thing

### 9.1 Plain verl: the §7 tool task, with verl's `ToolAgentLoop`

Everything from §1–§7.2 is now verl's job: Ray worker groups, FSDP actor, colocated vLLM with weight sync, `DataProto`, GRPO, PPO-clip, and the multi-turn tool loop. What *you* provide is the part that is specific to your task, in verl's format:

* **the tool**, as a function decorated with `@function_tool` (its docstring and type hints become the JSON schema in the prompt);
* **the reward**, as `compute_score(data_source, solution_str, ground_truth, extra_info)`;
* **the data**, as parquet rows with a chat `prompt`, a `data_source` and `reward_model.ground_truth`.

All three reuse `arith_task`.
''')

code(r'''
%%writefile /tmp/verl-from-scratch/verl_calc_tool.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arith_task import safe_eval  # noqa: E402
from verl.tools.function_tool import function_tool  # noqa: E402


@function_tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression with numbers, + - * / and parentheses.

    Args:
        expression: The arithmetic expression, e.g. '123 * 456 - 789'.
    """
    try:
        value = safe_eval(expression)
    except Exception as exc:
        return f"Error: {exc}"
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"
''')

code(r'''
%%writefile /tmp/verl-from-scratch/verl_reward.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arith_task import reward_fn  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return reward_fn(solution_str, int(ground_truth))
''')

code(r'''
import datasets


def verl_rows(problems):
    return [
        {"data_source": "arith", "prompt": [{"role": "user", "content": f"{p['question']} {TOOL_INSTRUCTION}"}],
         "reward_model": {"style": "rule", "ground_truth": str(p["answer"])}, "extra_info": {"index": i}}
        for i, p in enumerate(problems)
    ]


datasets.Dataset.from_list(verl_rows(HARD_TRAIN[:512])).to_parquet(str(WORK / "arith_train.parquet"))
datasets.Dataset.from_list(verl_rows(HARD_EVAL[:64])).to_parquet(str(WORK / "arith_val.parquet"))

PLAIN_VERL_METRICS = WORK / "plain_verl_metrics.jsonl"
PLAIN_VERL_METRICS.unlink(missing_ok=True)
verl_cmd = [
    sys.executable, "-m", "verl.trainer.main_ppo",
    "algorithm.adv_estimator=grpo",
    f"data.train_files={WORK / 'arith_train.parquet'}", f"data.val_files={WORK / 'arith_val.parquet'}",
    "data.train_batch_size=16", "data.max_prompt_length=1024", "data.max_response_length=1024", "data.dataloader_num_workers=0",
    f"actor_rollout_ref.model.path={MODEL}",
    "actor_rollout_ref.model.use_remove_padding=True", "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    f"actor_rollout_ref.actor.optim.lr={LR}",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
    "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.actor.clip_ratio_low=0.2", "actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.4", "actor_rollout_ref.rollout.n=8",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
    "actor_rollout_ref.rollout.multi_turn.enable=True", "actor_rollout_ref.rollout.multi_turn.format=hermes",
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3",
    f"actor_rollout_ref.rollout.multi_turn.function_tool_path={WORK / 'verl_calc_tool.py'}",
    "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent", "actor_rollout_ref.rollout.agent.num_workers=2",
    f"reward.custom_reward_function.path={WORK / 'verl_reward.py'}", "reward.custom_reward_function.name=compute_score",
    "reward.num_workers=2",
    "trainer.n_gpus_per_node=1", "trainer.nnodes=1", "trainer.total_training_steps=3",
    "trainer.val_before_train=True", "trainer.test_freq=3", "trainer.save_freq=-1",
    "trainer.logger=[console,file]", "trainer.project_name=verl_from_scratch", "trainer.experiment_name=plain_verl",
    f"trainer.rollout_data_dir={WORK / 'verl_rollouts'}", f"trainer.validation_data_dir={WORK / 'verl_val'}",
]
print(" \\\n  ".join(verl_cmd[2:]))
''')

md(r'''
Map the config back onto what you built:

| Config | Your code |
|---|---|
| `algorithm.adv_estimator=grpo` | `grpo_advantages` / `compute_grpo_outcome_advantage` (§1.4, §4) |
| `actor.clip_ratio_low/high`, `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu` | `ppo_clip_loss` and `ppo_update`'s mini/micro-batch loops (§1.4) |
| `rollout.name=vllm`, `gpu_memory_utilization`, `n` | `RolloutWorker` and `vllm_generate(n=...)` (§2, §5) |
| `rollout.multi_turn.*`, `agent.default_agent_loop=tool_agent` | `run_episodes` (§7.1) |
| `trainer.n_gpus_per_node`, `nnodes` | the resource pool (§6.2) |
| `reward.custom_reward_function` | `reward_fn` |

This takes a few minutes, mostly start-up (Ray, the FSDP actor, vLLM with CUDA graphs). Selected log lines are printed; the full log is in `/tmp/verl-from-scratch/logs/plain_verl.log`.
''')

code(r'''
start_service("plain_verl", verl_cmd, cwd=WORK, env={"VERL_FILE_LOGGER_PATH": str(PLAIN_VERL_METRICS), "WANDB_MODE": "disabled"})
rc = follow_log("plain_verl", r"step:\d+ |Initial validation|Training Progress")
print(f"\nverl exited with code {rc}")
if rc != 0:
    raise RuntimeError(log_tail("plain_verl", 60))
''')

code(r'''
def read_metrics(path: Path, patterns: list[str]) -> pd.DataFrame:
    """verl's file logger writes one JSON line per step; keep the metrics whose names match any pattern, in pattern order."""
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    names = sorted({k for r in recs for k in r["data"]})
    keep = [k for p in patterns for k in names if re.fullmatch(p, k)]
    keep = list(dict.fromkeys(keep))
    return pd.DataFrame([{"step": r["step"], **{k: r["data"].get(k) for k in keep}} for r in recs]).groupby("step").last()


read_metrics(PLAIN_VERL_METRICS, [
    r"val-core/.*mean@1", r"critic/score/mean", r"actor/pg_loss", r"actor/grad_norm", r"actor/ppo_kl", r"actor/pg_clipfrac",
    r"response_length/mean", r"num_turns/mean", r"timing_s/(gen|old_log_prob|update_actor|update_weights|step)",
]).T
''')

md(r'''
`trainer.rollout_data_dir` and `trainer.validation_data_dir` make verl dump every episode it generated as JSONL: the decoded prompt and response (tool calls and tool outputs included), the ground truth and the score. Here's one validation episode from before training, exactly as `ToolAgentLoop` produced it:
''')

code(r'''
val0 = [json.loads(line) for line in (WORK / "verl_val" / "0.jsonl").read_text().splitlines()]
print(f"{len(val0)} validation episodes before training, mean score {np.mean([v['score'] for v in val0]):.2f}\n")
ex = val0[0]
print("…" + ex["input"][-200:], "\n" + "─" * 100 + "\n" + ex["output"], f"\n{'─' * 100}\nground truth {ex['gts']} | score {ex['score']}")
for path in sorted((WORK / "verl_rollouts").glob("*.jsonl"), key=lambda p: int(p.stem)):
    scores = [json.loads(line)["score"] for line in path.read_text().splitlines()]
    print(f"training step {path.stem}: {len(scores)} episodes, mean score {np.mean(scores):.2f}")
''')

md(r'''
`critic/score/mean` is the training reward (verl logs rewards under `critic/` even without a critic), `num_turns` counts messages per episode, and the `timing_s/*` rows are the same phases you timed by hand in every stage.

### 9.2 verl + Agent Lightning: the real agent

Now the agent from §7.3 and §8, with the real Gateway and Controller instead of our toy, and `AgentLightningRayPPOTrainer` instead of our loop. This is the same command the walkthrough notebook (§10) and `examples/calc_x/run_local.sh` use, with the 0.5B model and a few short-run overrides.
''')

code(r'''
AGL_PORT, AGL_KEY = 8283, "verl-from-scratch"
AGL_URL = f"http://127.0.0.1:{AGL_PORT}"
start_service("agl_gateway", [
    sys.executable, "-m", "agentlightning.server", "host=127.0.0.1", f"port={AGL_PORT}", f"key={AGL_KEY}",
    f"default_proxy.model_name={CALC_X_MODEL}", f"hydra.run.dir={WORK}/hydra/gateway", "hydra.output_subdir=null",
])
wait_until_healthy("agl_gateway", f"{AGL_URL}/healthz")
start_service("agl_controller", [
    sys.executable, "-m", "agentlightning.controller", "runner_type=local", f"agl_server.url={AGL_URL}", f"agl_server.key={AGL_KEY}",
    "local_runner.poll_interval=1", "local_runner.maximum_size=128", f"hydra.run.dir={WORK}/hydra/controller", "hydra.output_subdir=null",
], cwd=REPO / "examples/calc_x")
time.sleep(3)
assert SERVICES["agl_controller"].poll() is None, log_tail("agl_controller")

AGL_METRICS = WORK / "agl_metrics.jsonl"
AGL_METRICS.unlink(missing_ok=True)
start_service("agl_trainer", [
    sys.executable, "train_calc_agent.py", "--agl-base-url", AGL_URL, "--agl-key", AGL_KEY, "--model", CALC_X_MODEL, "--run-name", "from_scratch",
    "--train-file", str(CALC_X_DATA / "train.parquet"), "--val-file", str(CALC_X_DATA / "test_mini.parquet"),
    "trainer.total_training_steps=3", "data.train_batch_size=8", "actor_rollout_ref.actor.ppo_mini_batch_size=8",
    "trainer.val_before_train=True", "trainer.save_freq=-1", "trainer.logger=[console,file]",
], cwd=REPO / "examples/calc_x", env={"VERL_FILE_LOGGER_PATH": str(AGL_METRICS), "WANDB_MODE": "disabled"})
rc = follow_log("agl_trainer", r"Initial validation|Final validation|Training Progress|AglRolloutManager: completed=(\d+)/\1 |step:\d+ ")
print(f"\ntrainer exited with code {rc}")
if rc != 0:
    raise RuntimeError(log_tail("agl_trainer", 60))
''')

code(r'''
read_metrics(AGL_METRICS, [
    r"val/reward", r"training/reward", r"training/n_groups", r"training/n_zero_adv_groups", r"training/n_sample_trained",
    r"actor/pg_loss", r"actor/grad_norm", r"actor/ppo_kl", r"response_length/mean",
    r"timing_s/(gen|old_log_prob|update_actor|update_weights|step)",
]).T
''')

md(r'''
Compare the two runs' configs. The verl side is almost the same. What changed is *where rollouts come from*: in §9.1 verl ran a token-level agent loop inside its Ray workers; in §9.2 it registered its vLLM replicas with the Gateway, queued rollouts, and the Controller ran the unmodified AutoGen agent as ordinary processes (or Kubernetes pods) that never knew they were being trained.
''')

code(r'''
for name in ["agl_trainer", "agl_controller", "agl_gateway"]:
    stop_service(name)
import shutil

shutil.rmtree(SHM, ignore_errors=True)
print("remaining services:", [n for n, p in SERVICES.items() if p.poll() is None])
''')

# ─────────────────────────────────────────────────────────────────────────────
md(r'''
## 10. What you built, and what each library is for

| You wrote | It became | Why the library version |
|---|---|---|
| `hf_generate` | vLLM | generation speed: paged KV cache, continuous batching, CUDA graphs |
| `save_for_inference` + `reload_weights` | verl's checkpoint engine (CUDA IPC / NCCL, bucketed) | no round trip through host memory; works across GPUs and nodes |
| dict of padded tensors | `DataProto` | splits, concatenates and travels between processes |
| `grpo_advantages`, `ppo_clip_loss` | `core_algos` | the same maths, plus dual-clip, many estimators and losses, aggregation modes |
| one process | Ray actors | separate roles, resources and lifetimes |
| `MyWorkerGroup` | `RayWorkerGroup` + `@register(Dispatch...)` | one worker per GPU, N GPUs, ranks and rendezvous handled |
| `run_episodes` | `ToolAgentLoop` | multi-turn rollouts inside verl, for agents written in verl's terms |
| the 60-line gateway + `Popen` loop | Agent Lightning Gateway + Controller + `AgentLightningRayPPOTrainer` | real agents, unmodified: state, failures, scale, async, trajectory merging |

**When to use what:**

* **Single-turn**, or an agent you're happy to (re)write as a verl agent loop with verl tools → **verl alone** (§9.1).
* An **existing agent**: its own framework (AutoGen, LangChain, OpenAI Agents SDK…), its own tools and sandboxes, its own processes or containers, or code you don't control → **verl + Agent Lightning** (§9.2). The agent only changes its base URL.

The hard parts of agent RL that §7.3 and §8 made concrete — exact tokens from HTTP traffic, per-rollout bookkeeping, rollouts as processes — are exactly the parts Agent Lightning owns. Everything below that line is still verl.

### Exercises

1. **Forget to sync.** In §2.2, comment out `push_weights` and watch `logp_mismatch` over 10 steps. Then put it back but remove `reset_prefix_cache()`. Can you see a difference? Why might it be small here and large elsewhere?
2. **The sampling-defaults trap.** In §2, sample with `llm.generate(prompts)` *without* `SamplingParams`, compute the trainer's log-probs of those samples, and compare the mean log-prob with properly sampled ones. Which is higher, and why is that a problem for RL?
3. **Dr. GRPO.** Pass `norm_adv_by_std_in_grpo=False` to `compute_grpo_outcome_advantage` in `add_advantages`. How do the advantages of a 1-of-8 group compare to a 7-of-8 group?
4. **Trajectory merging.** For the §7 episodes, all model turns already form one row. For §8's Calc-X calls, check how often call *k+1*'s `prompt_ids` start with call *k*'s `prompt_ids + response_ids`. When they do, the calls could be merged into one row. When don't they, and why (hint: `reflect_on_tool_use`)?
5. **Scale out.** On a machine with 2+ GPUs, change §6.2's pool to `RayResourcePool(process_on_nodes=[2], use_gpu=True)` and give `VerlScoreWorker` a `compute_log_prob` method holding a copy of the policy. Initialise with the `nccl` backend instead of `gloo`.
''')


def build(path: str) -> None:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (agl-rl-tutorials .venv)", "language": "python", "name": "agl-rl-tutorials"},
            "language_info": {"name": "python", "version": "3.12.15"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path}: {len(cells)} cells")


if __name__ == "__main__":
    build(sys.argv[1])
