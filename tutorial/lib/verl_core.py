"""Notebook 2's DataProto helpers (§3), so notebook 3 can import them. Copied verbatim from notebook 2.

`verl_ppo_loss` is taught in notebook 1 and lives in rl_core; it is re-exported here for notebook 3."""

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from arith_task import reward_fn
from rl_core import decode_tokens, make_padded_batch, verl_ppo_loss  # noqa: F401  (verl_ppo_loss: re-exported for notebook 3)


def build_batch(samples: list[dict], answers: list[int], uids: list) -> DataProto:
    """Score samples and lay them out the way verl's trainer expects."""
    rewards = torch.tensor([reward_fn(decode_tokens(sample["response_ids"]), answer) for sample, answer in zip(samples, answers)])
    batch = DataProto.from_dict(
        tensors=make_padded_batch([sample["prompt_ids"] for sample in samples], [sample["response_ids"] for sample in samples]),
        non_tensors={"uid": np.array(uids, dtype=object), "answer": np.array(answers, dtype=object)},
        meta_info={"temperature": 1.0},
    )
    mask = batch.batch["response_mask"]

    # Each response's reward goes on its last token (verl's token_level_rewards layout)
    last_token = mask.sum(-1).clamp(min=1) - 1
    token_level_rewards = torch.zeros(mask.shape)
    token_level_rewards[torch.arange(len(batch)), last_token] = rewards
    batch.batch["token_level_rewards"] = token_level_rewards

    # The inference engine's log-probs, padded like the responses (for the logp_mismatch metric)
    rollout_log_probs = torch.zeros(mask.shape)
    for row, sample in enumerate(samples):
        if sample["log_probs"] is not None:
            rollout_log_probs[row, : len(sample["log_probs"])] = torch.tensor(sample["log_probs"])
    batch.batch["rollout_log_probs"] = rollout_log_probs
    return batch


def add_advantages(batch: DataProto) -> DataProto:
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_rewards"], response_mask=batch.batch["response_mask"], index=batch.non_tensor_batch["uid"],
    )
    batch.batch["advantages"] = advantages
    return batch


def compute_batch_metrics(batch: DataProto) -> dict:
    trained = batch.batch["response_mask"].bool()
    metrics = {"reward": batch.batch["token_level_rewards"].sum(-1).mean().item(), "response_len": trained.sum(-1).float().mean().item()}
    if "old_log_probs" in batch.batch.keys():
        metrics["logp_mismatch"] = (batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"])[trained].abs().mean().item()
    return metrics
