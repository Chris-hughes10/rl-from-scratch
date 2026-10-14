"""Trainer stand-in: run ONE Agent Lightning rollout step in a single debuggable process.

Mirrors what `AgentLightningRayPPOTrainer._rollout()` and the advantage part of `_train_step()`
(agentlightning/verl/trainer.py) do, but without Ray or verl workers, so every line can be stepped
through in the VS Code debugger:

    register model endpoint -> enqueue rollouts -> wait -> build CompletedRollouts
    -> RolloutAdapter (traces -> verl DataProto) -> rollout-level GRPO advantages

The policy update (old/ref log-probs, PPO loss, optimizer step) is intentionally not run here.

Requires the vLLM task, the Gateway and the Controller to be running (see .vscode/launch.json).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.calc_x.train_calc_agent import build_config  # noqa: E402
from verl.trainer.ppo.ray_trainer import compute_response_mask  # noqa: E402

from agentlightning.verl.agl_rollout_manager import AglRolloutManager  # noqa: E402
from agentlightning.verl.rollout_adapter import RolloutAdapter  # noqa: E402
from agentlightning.verl.rollout_level_advantage import compute_rollout_level_advantage  # noqa: E402


def wait_for(url: str, name: str, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        print(f"waiting for {name} at {url} ...")
        time.sleep(3)
    raise TimeoutError(f"{name} not reachable at {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agl-url", default="http://127.0.0.1:8181")
    parser.add_argument("--agl-key", default="dummy")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default=str(REPO_ROOT / "examples/calc_x/data/train.parquet"))
    parser.add_argument("--num-questions", type=int, default=2)
    parser.add_argument("--rollout-n", type=int, default=4, help="rollouts per question (GRPO group size)")
    parser.add_argument(
        "--level",
        choices=["trajectory", "transition"],
        default=None,
        help="trace aggregation level (default: from config, i.e. trajectory)",
    )
    args = parser.parse_args()

    # Same merged config the real Calc-X run uses (AGL base config + verl ppo_trainer + Calc-X overrides).
    config = build_config(model=args.model, agl_base_url=args.agl_url, agl_key=args.agl_key)
    agl = config.agentlightning
    level = args.level or agl.trace_aggregator.level

    wait_for(f"{args.agl_url}/healthz", "Gateway")
    wait_for(f"{args.vllm_url.removesuffix('/v1')}/health", "vLLM")

    # The trainer gets this dict from verl's dataloader (gen_batch.non_tensor_batch): one object array per column.
    rows = pd.read_parquet(args.data).sample(n=args.num_questions, random_state=0)
    data_dict = {col: np.array(rows[col].tolist(), dtype=object) for col in rows.columns}

    # --- 1. Rollout: what trainer._rollout() does ---------------------------------------------------
    manager = AglRolloutManager(
        agl_base_url=args.agl_url,
        agl_key=args.agl_key,
        model=args.model,
        step=0,
        train_rollout_n=args.rollout_n,
        rollout_timeout_seconds=agl.rollout_timeout_seconds,
        local_agent_class=agl.local.agent_class,
        local_env_map=dict(agl.local.env_map),
    )
    manager.delete_model()
    manager.register_model([args.vllm_url])
    completed = manager.enqueue_and_wait_until_completed(data_dict, is_train=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("\n=== Completed rollouts ===")
    for r in sorted(completed, key=lambda r: (r.sample_idx_in_step, r.rollout_id)):
        last = r.triplets[-1].response["token_ids"] if r.triplets else []
        print(
            f"[q{r.sample_idx_in_step}] rollout={r.rollout_id[:8]} state={r.rollout_state} "
            f"reward={r.final_reward} model_calls={len(r.triplets or [])}"
        )
        print(f"      last response: {tokenizer.decode(last, skip_special_tokens=True)[:200]!r}")

    # --- 2. Traces -> training rows: RolloutAdapter --------------------------------------------------
    trace_cfg = agl.trace_aggregator
    is_trajectory = level == "trajectory"
    adapter = RolloutAdapter(
        max_prompt_length=trace_cfg.trajectory_max_prompt_length if is_trajectory else config.data.max_prompt_length,
        max_response_length=(
            trace_cfg.trajectory_max_response_length if is_trajectory else config.data.max_response_length
        ),
        max_total_length=trace_cfg.get("trajectory_max_total_length") if is_trajectory else None,
        device=torch.device("cpu"),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        reward_fillna_value=agl.reward_fillna_value,
        trace_aggregator_level=level,
        tokenizer=tokenizer,
    )
    batch, adapter_metrics = adapter.get_train_data_batch(completed, global_steps=0)

    # --- 3. Advantages: the "adv" block of trainer._train_step() -------------------------------------
    batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]
    if "response_mask" not in batch.batch:
        batch.batch["response_mask"] = compute_response_mask(batch)
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    batch, adv_metrics = compute_rollout_level_advantage(
        batch,
        adv_estimator=config.algorithm.adv_estimator,
        gamma=config.algorithm.gamma,
        lam=config.algorithm.lam,
        num_repeat=args.rollout_n,
        norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
        config=config.algorithm,
    )

    print(f"\n=== Training rows (level={level}) ===")
    mask = batch.batch["response_mask"].bool()
    for i in range(len(batch)):
        trained = int(mask[i].sum())
        advantage = batch.batch["advantages"][i][mask[i]]
        print(
            f"row {i}: q={str(batch.non_tensor_batch['data_id_list'][i])[:8]} "
            f"rollout={str(batch.non_tensor_batch['rollout_id_list'][i])[:8]} "
            f"reward={batch.batch['token_level_scores'][i].sum().item():.2f} "
            f"advantage={advantage[0].item() if trained else float('nan'):+.3f} "
            f"total_tokens={int(batch.batch['attention_mask'][i].sum())} trained_tokens={trained}"
        )
    print("\nadapter metrics:", {k: v for k, v in adapter_metrics.items() if not k.startswith("timing/")})
    print("advantage metrics:", adv_metrics)
    print("\nbatch tensors:", {k: tuple(v.shape) for k, v in batch.batch.items()})


if __name__ == "__main__":
    main()
