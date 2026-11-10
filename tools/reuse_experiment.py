"""Notebook 1, §1.7: does taking several optimizer steps per batch help? Several seeds per setting, 60 steps each.

Each setting trains from the base model with notebook 1's final `rl_step` and vLLM generation (16 questions x 8 samples
per step), and scores greedy accuracy on 1,000 held-out questions every 10 steps. One JSON line per (setting, seed) goes
to tutorial/series/results/reuse_experiment.jsonl; finished runs are skipped, so an interrupted run can be resumed.
About 5 minutes per run on an H100, so about an hour for everything.

Usage: .venv/bin/python tools/reuse_experiment.py [SETTING ...]   (default: every setting)
"""

import gc
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tutorial/lib"))
OUT = REPO / "tutorial/series/results/reuse_experiment.jsonl"
NUM_STEPS = 60
EVAL_EVERY = 10
SEEDS = [0, 1, 2]


def main():
    from common import MODEL, SEED, setup

    setup()

    import torch
    from arith_task import make_problems
    from rl_core import LR, TRAIN, evaluate, load_policy, make_optimizer, push_weights, rl_step, vllm_generate
    from vllm import LLM

    # name -> (optimizer steps per batch, mini_batch_size, learning rate); every setting trains on 128 rows per step
    all_settings = {"1": (1, 128, LR), "2": (2, 64, LR), "4": (4, 32, LR), "4_lr/4": (4, 32, LR / 4)}
    settings = {name: all_settings[name] for name in (sys.argv[1:] or all_settings)}
    eval_problems = make_problems(1000, seed=3)

    llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=4096, seed=SEED)
    generate = lambda prompts, samples_per_prompt: vllm_generate(llm, prompts, samples_per_prompt)
    generate_greedy = lambda prompts, samples_per_prompt: vllm_generate(llm, prompts, samples_per_prompt, temperature=0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finished = {(run["setting"], run["seed"]) for run in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()

    for seed in SEEDS:
        for setting, (optimizer_steps, mini_batch_size, lr) in settings.items():
            if (setting, seed) in finished:
                continue
            torch.manual_seed(seed)
            policy = load_policy()
            optimizer = make_optimizer(policy, lr)
            push_weights(policy, llm)  # vLLM back to the base model's weights
            question_rng = random.Random(seed)  # with the same seed, every setting sees the same questions
            accuracy = {0: evaluate(generate_greedy, eval_problems)}
            rewards = []
            start = time.time()
            for step in range(1, NUM_STEPS + 1):
                metrics, _, _ = rl_step(policy, optimizer, generate, question_rng.sample(TRAIN, 16), samples_per_prompt=8,
                                        mini_batch_size=mini_batch_size)
                push_weights(policy, llm)
                rewards.append(metrics["reward"])
                if step % EVAL_EVERY == 0:
                    accuracy[step] = evaluate(generate_greedy, eval_problems)
                    print(f"setting {setting} seed {seed} step {step:2d} | reward (last 10) {sum(rewards[-10:]) / 10:.3f} | "
                          f"accuracy {accuracy[step]:.3f} | {time.time() - start:.0f}s", flush=True)
            with OUT.open("a") as f:
                f.write(json.dumps({"setting": setting, "optimizer_steps": optimizer_steps, "lr": lr, "seed": seed,
                                    "accuracy": accuracy, "rewards": rewards}) + "\n")
            del policy, optimizer
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":  # vLLM starts its engine with `spawn`, which re-imports this file
    main()
