"""CPU-only checks of the notebook's pure-Python parts (no GPU, no vLLM)."""

import json
from pathlib import Path
import os
import re

NB = str(Path(__file__).resolve().parents[1] / "tutorial/verl_from_scratch.ipynb")
cells = [("".join(c["source"]), c["cell_type"]) for c in json.load(open(NB))["cells"]]
code = [s for s, t in cells if t == "code"]


def cell(prefix: str) -> str:
    matches = [s for s in code if s.startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]


ns: dict = {"__name__": "__main__"}


def run(src: str, label: str) -> None:
    print(f"\n===== {label} =====")
    exec(compile(src, label, "exec"), ns)


# setup, with DEVICE forced to CPU and no GPU memory query
setup = cell("import gc").replace('DEVICE = "cuda"', 'DEVICE = "cpu"')
setup = re.sub(r"free, total = torch.cuda.mem_get_info\(\)\n.*", "", setup, flags=re.S)
run(setup, "setup")
run(cell("import httpx"), "helpers")
ns.setdefault("CALC_X_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # defined in a GPU cell (§8) the harness skips

writefile = cell("%%writefile /tmp/verl-from-scratch/arith_task.py")
path, body = writefile.split("\n", 1)
open(path.split()[1], "w").write(body)
run(cell("import importlib"), "task")

tok_cell = cell("from transformers import AutoModelForCausalLM")
tok_cell = tok_cell.split("\n\npolicy = load_policy()")[0]
run(tok_cell, "tokenizer + helpers (no model)")
run(cell("def collate("), "collate (with sample data)") if False else None
collate_src = cell("def collate(").split("\n\ndemo = ")[0]
run(collate_src, "collate")
run(cell("def rows("), "algorithm functions")

# fake samples, prompt-major, n=8
import random

import torch

tok = ns["tokenizer"]
TRAIN = ns["TRAIN"]
rng = random.Random(0)
samples = []
for p in TRAIN[:4]:
    prompt = ns["chat_prompt_ids"](p["question"])
    for k in range(8):
        text = f"work... \\boxed{{{p['answer'] if rng.random() < 0.5 else p['answer'] + 1}}}"
        ids = tok.encode(text, add_special_tokens=False)[: rng.randint(3, 20)] + [tok.convert_tokens_to_ids("<|im_end|>")]
        samples.append({"prompt_ids": prompt, "response_ids": ids, "log_probs": [-0.1] * len(ids)})
ns["samples"] = samples
ns["batch"] = ns["collate"]([s["prompt_ids"] for s in samples], [s["response_ids"] for s in samples])

# §3 DataProto demo (skip the compute_log_probs line)
dp_src = cell("from verl import DataProto").replace('print("works with §1 helpers: ", compute_log_probs(policy, dp.batch).shape)', "")
run(dp_src, "DataProto")

# §4 comparisons (skip the flash-attn logprobs part, which needs CUDA)
cmp_src = cell("from omegaconf import OmegaConf").split("# Log-probs: verl's fused kernel")[0]
run(cmp_src, "advantage + loss comparison")

# build_batch / add_advantages
bb = cell("def build_batch(").split("\n\nverl_fn_history = []")[0]
run(bb, "build_batch")
problems = TRAIN[:4]
last_batch = ns["build_batch"](samples, [problems[i // 8]["answer"] for i in range(len(samples))], [i // 8 for i in range(len(samples))])
ns["add_advantages"](last_batch)
last_batch.batch["old_log_probs"] = torch.full(last_batch.batch["responses"].shape, -0.12)
print("batch_metrics:", ns["batch_metrics"](last_batch))
ns["last_batch"] = last_batch

# §6 worker groups on CPU-only Ray
import ray

ray.init(num_cpus=16, num_gpus=0, include_dashboard=False, runtime_env={"env_vars": {k: os.environ[k] for k in ("PYTHONPATH", "TOKENIZERS_PARALLELISM")}})
ns["ray"] = ray
run(cell("def free_port()"), "MyWorkerGroup")
run(cell("from verl.single_controller.base import Worker"), "verl RayWorkerGroup")
run(cell("for w in my_wg.workers"), "kill workers")
ray.shutdown()

# §7 tool prompt + observation check + tool call parsing
run(cell("CALC_TOOL = {"), "tool prompt")
run(cell('TOOL_CALL = re.compile'), "observation check")
for raw in ['{"name": "calculate", "arguments": {"expression": "123 * 456 - 789"}}', '{"name": "calculate", "arguments": "{\\"expression\\": \\"1/0\\"}"}',
            '{"name": "rm", "arguments": {}}', "not json", '{"name": "calculate", "arguments": {"expression": "__import__(\'os\')"}}']:
    print(f"run_tool_call({raw[:50]!r}) -> {ns['run_tool_call'](raw)!r}")

# §7.3 rerender on a synthetic tool-using episode built with the exact template text
run(cell("def run_episodes(").split("\n\nepisodes = run_episodes(")[0], "episode helpers")
q = ns["HARD_TRAIN"][0]
prompt = ns["tool_prompt_ids"](q["question"])
turn1 = tok.encode('<tool_call>\n{"name": "calculate", "arguments": {"expression": "1 * 2"}}\n</tool_call>', add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
obs = tok.encode(ns["observation_text"](["2"]), add_special_tokens=False)
turn2 = tok.encode("So \\boxed{2}", add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
ep = {"question": q["question"], "answer": 2, "group": 0, "prompt_ids": prompt, "response_ids": turn1 + obs + turn2,
      "mask": [1] * len(turn1) + [0] * len(obs) + [1] * len(turn2), "log_probs": [0.0] * (len(turn1) + len(obs) + len(turn2)),
      "tool_calls": 1, "tool_results": [["2"]], "done": True, "last_text": tok.decode(turn2)}
ep["reward"] = ns["reward_fn"](ep["last_text"], 2)
ns["all_episodes"] = [ep, dict(ep, group=1)]
b = ns["episodes_to_batch"](ns["all_episodes"])
print("episode batch:", {k: tuple(v.shape) for k, v in b.batch.items()}, "trained tokens:", b.batch["response_mask"].sum().item())
rerender_src = cell("def model_turns(")
run(rerender_src, "rerender check (synthetic, should match)")

# §8 gateway_batch on synthetic records
gb = cell("def push_weights_http(").split("\n\npolicy = load_policy(")[0]
run(gb, "gateway_batch defs")
ns["RECORDS"] = {
    "r0": {"calls": [{"prompt_ids": prompt, "response_ids": turn1, "log_probs": [-0.1] * len(turn1)}, {"prompt_ids": prompt + turn1 + obs, "response_ids": turn2, "log_probs": [-0.2] * len(turn2)}], "reward": 1.0},
    "r1": {"calls": [{"prompt_ids": prompt, "response_ids": turn2, "log_probs": [-0.3] * len(turn2)}], "reward": 0.0},
    "r2": {"calls": [], "reward": None},
}
gbatch, info = ns["gateway_batch"]([("r0", 0), ("r1", 0), ("r2", 0)])
print("gateway batch rows:", len(gbatch), info, "row advantages:", gbatch.batch["advantages"].sum(-1) / gbatch.batch["response_mask"].sum(-1))
print("\nALL CPU CHECKS RAN")
