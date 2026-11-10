
"""Notebook 1's trainer as a module, so notebooks 2-4 can import it without re-running notebook 1.
The vLLM pieces (vllm_generate, save_for_inference, push_weights, plot_histories) are taught in notebook 2, §2.3-§2.5.

Every function here is copied verbatim from those notebooks (tools/check_series.py checks that they match);
read them there, where each one is explained step by step.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, get_policy_loss_fn
from vllm import SamplingParams, TokensPrompt

from arith_task import BOXED, make_problems, reward_fn
from common import DEVICE, MODEL, SHM

TRAIN = make_problems(4000, seed=1)
EVAL = make_problems(200, seed=2)
INSTRUCTION = r"Show brief working, then give the final answer as \boxed{N}."
LR = 2e-6

hf_logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(MODEL)


PAD_ID = tokenizer.pad_token_id


STOP_IDS = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}


def chat_prompt_ids(question: str) -> list[int]:
    """The question plus our instruction, wrapped in the model's chat template, as token IDs."""
    messages = [{"role": "user", "content": f"{question} {INSTRUCTION}"}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def decode_tokens(ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_policy(path: str = MODEL):
    hf_logging.set_verbosity_error()  # also inside Ray workers
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="flash_attention_2")  # fp32 master weights
    model = model.to(DEVICE)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # recompute activations in backward: less memory
    model.config.use_cache = False  # the KV cache only speeds up generation, not training
    return model


def make_optimizer(model, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def left_pad(prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack prompts of different lengths into one tensor, padded on the left so they all end in the last column."""
    prompt_width = max(len(prompt) for prompt in prompts)  # the longest prompt
    input_ids = torch.full((len(prompts), prompt_width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)

    for row, prompt in enumerate(prompts):
        num_padding = prompt_width - len(prompt)
        input_ids[row, num_padding:] = torch.tensor(prompt)
        attention_mask[row, num_padding:] = 1

    return input_ids, attention_mask


def hf_generate(model, prompts: list[list[int]], samples_per_prompt: int, max_new_tokens: int = 256, greedy: bool = False) -> list[dict]:
    """samples_per_prompt responses to each prompt, all generated together. Prompt-major: prompt 0's samples come first."""
    row_prompts = [prompt for prompt in prompts for _ in range(samples_per_prompt)]  # one row per sample
    input_ids, attention_mask = left_pad(row_prompts)
    prompt_width = input_ids.shape[1]

    if greedy:  # always the most likely token: for evaluation
        sampling_settings = {"do_sample": False}
    else:       # a true sample from π: for training
        sampling_settings = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}

    model.eval()  # inference mode while generating; back to training mode afterwards
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE),
            max_new_tokens=max_new_tokens, pad_token_id=PAD_ID, use_cache=True, **sampling_settings,
        )
    model.train()

    samples = []
    for prompt, output_row in zip(row_prompts, output_ids.tolist()):
        response_ids = truncate_at_stop(output_row[prompt_width:])  # the response starts after the (padded) prompt
        samples.append({"prompt_ids": prompt, "response_ids": response_ids, "log_probs": None})  # HF doesn't give us log-probs
    return samples


def truncate_at_stop(ids: list[int]) -> list[int]:
    """Cut a response just after its first stop token (generate pads rows that finished early)."""
    for position, token_id in enumerate(ids):
        if token_id in STOP_IDS:
            return ids[: position + 1]  # keep the stop token: the model must learn when to stop
    return ids


def make_padded_batch(prompts: list[list[int]], responses: list[list[int]], response_masks: list[list[int]] | None = None) -> dict[str, torch.Tensor]:
    batch_size = len(prompts)
    prompt_width = max(len(prompt) for prompt in prompts)                       # the longest prompt
    response_width = max(1, max(len(response) for response in responses))     # the longest response (at least 1 column)

    prompt_ids = torch.full((batch_size, prompt_width), PAD_ID, dtype=torch.long)
    response_ids = torch.full((batch_size, response_width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, prompt_width + response_width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, response_width), dtype=torch.long)

    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        prompt_start = prompt_width - len(prompt)  # prompts are left-padded...
        response_end = len(response)               # ...and responses right-padded
        prompt_ids[row, prompt_start:] = torch.tensor(prompt)
        response_ids[row, :response_end] = torch.tensor(response, dtype=torch.long)
        attention_mask[row, prompt_start : prompt_width + response_end] = 1
        trained = response_masks[row] if response_masks else [1] * len(response)  # default: train on every response token
        response_mask[row, :response_end] = torch.tensor(trained, dtype=torch.long)

    return {
        "prompts": prompt_ids,
        "responses": response_ids,
        "input_ids": torch.cat([prompt_ids, response_ids], dim=1),
        "attention_mask": attention_mask,
        # each token's position among the row's real tokens (0, 1, 2, ...), so left-padding doesn't shift positions
        "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
        "response_mask": response_mask,
    }


def select_rows(batch, index):
    """Index a batch given as a dict of tensors, or as verl's TensorDict (§3)."""
    if isinstance(batch, dict):
        return {key: tensor[index] for key, tensor in batch.items()}
    return batch[index]


def micro_batch_log_probs(model, micro_batch, with_grad: bool = False) -> torch.Tensor:
    """log π(token) for every response token; shape (rows, response_width)."""
    response_ids = micro_batch["responses"].to(DEVICE)
    response_width = response_ids.shape[1]
   
    with torch.set_grad_enabled(with_grad), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(  # 1. one forward pass over prompt + response
            input_ids=micro_batch["input_ids"].to(DEVICE),
            attention_mask=micro_batch["attention_mask"].to(DEVICE),
            position_ids=micro_batch["position_ids"].to(DEVICE),
            logits_to_keep=response_width + 1,
        ).logits
    
    logits = logits[:, :-1].float()                # 2. position t-1 predicts token t, so drop the last position
    log_probs = torch.log_softmax(logits, dim=-1)  # 3. log-probs over the whole vocabulary
    token_log_prob = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1) # 4. the log-prob of the token actually there
    return token_log_prob  


def compute_log_probs(model, batch, micro_batch_size: int = 16) -> torch.Tensor:
    with torch.no_grad():
        model.eval()
        num_rows = batch["input_ids"].shape[0]
        chunks = []
        for start in range(0, num_rows, micro_batch_size):
            micro_batch = select_rows(batch, slice(start, start + micro_batch_size))
            chunks.append(micro_batch_log_probs(model, micro_batch))
    
    model.train()  # back to training mode, so ppo_update gets gradient checkpointing again
    return torch.cat(chunks).cpu()


def grpo_advantages(rewards: torch.Tensor, group_ids, eps: float = 1e-6) -> torch.Tensor:
    """Each reward minus its group's mean, divided by its group's std. A group is all the attempts at one question."""
    
    advantages = torch.zeros_like(rewards)
    group_ids = np.asarray(group_ids)
    
    for group in np.unique(group_ids):
        members = torch.as_tensor(np.nonzero(group_ids == group)[0])  # row indices of this group's attempts
        group_rewards = rewards[members]
        
        if len(members) > 1:
            advantages[members] = (group_rewards - group_rewards.mean()) / (group_rewards.std() + eps)
    
    return advantages


def ppo_clip_loss(log_probs, old_log_probs, advantages, mask, num_trained_tokens, clip_low: float = 0.2, clip_high: float = 0.28):
    ratio = torch.exp(log_probs - old_log_probs)  # ρ = π_θ / π_old, per token
    
    unclipped = -advantages * ratio
    
    clipped = -advantages * ratio.clamp(1 - clip_low, 1 + clip_high)
    
    loss = torch.maximum(unclipped, clipped)  # the larger loss = the smaller (more pessimistic) objective
    
    return (loss * mask).sum() / num_trained_tokens     # mean over trained tokens; num_trained_tokens counts the whole mini-batch (see ppo_update)


def ppo_update(model, optimizer, batch, old_log_probs, advantages, mini_batch_size: int = 64, micro_batch_size: int = 8, loss_fn=ppo_clip_loss) -> dict:
    """One pass over the batch: one optimizer step per mini-batch, gradient accumulation over micro-batches."""
    num_rows = batch["input_ids"].shape[0]
    grad_norms = []
    for mini_start in range(0, num_rows, mini_batch_size):
        mini_end = min(mini_start + mini_batch_size, num_rows)
        num_trained_tokens = batch["response_mask"][mini_start:mini_end].sum().item()  # trained tokens in the whole mini-batch

        # Accumulate gradients over micro-batches small enough to fit in memory
        for micro_start in range(mini_start, mini_end, micro_batch_size):
            row_slice = slice(micro_start, min(micro_start + micro_batch_size, mini_end))
            micro_batch = select_rows(batch, row_slice)
            log_probs = micro_batch_log_probs(model, micro_batch, with_grad=True)
            mask = micro_batch["response_mask"].to(DEVICE).float()
            loss = loss_fn(log_probs, old_log_probs[row_slice].to(DEVICE), advantages[row_slice].to(DEVICE), mask, num_trained_tokens)
            loss.backward()

        # Then one optimizer step for the mini-batch
        grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"grad_norm": float(np.mean(grad_norms))}


verl_policy_loss = get_policy_loss_fn("vanilla")


def verl_ppo_loss(log_probs, old_log_probs, advantages, mask, num_trained_tokens, clip_low=0.2, clip_high=0.28, clip_c=3.0):
    config = OmegaConf.create({
        "clip_ratio": clip_low, "clip_ratio_low": clip_low, "clip_ratio_high": clip_high, "clip_ratio_c": clip_c,
        "global_batch_info": {"batch_num_tokens": num_trained_tokens, "dp_size": 1},
    })
    loss, _ = verl_policy_loss(old_log_prob=old_log_probs, log_prob=log_probs, advantages=advantages, response_mask=mask, loss_agg_mode="token-mean", config=config)
    return loss


def rl_step(policy, optimizer, generate, problems, samples_per_prompt: int = 8, **update_kwargs) -> tuple[dict, dict, list[dict]]:
    timing = {}

    # 1. Sample several responses per problem (a GRPO group). Samples are prompt-major: sample i answers problem i // samples_per_prompt
    start = time.time()
    samples = generate([chat_prompt_ids(problem["question"]) for problem in problems], samples_per_prompt)
    timing["gen"] = time.time() - start
    group_ids = [i // samples_per_prompt for i in range(len(samples))]

    # 2. Score each response, and 3. pad everything into one batch
    texts = [decode_tokens(sample["response_ids"]) for sample in samples]
    rewards = torch.tensor([reward_fn(text, problems[group]["answer"]) for text, group in zip(texts, group_ids)])
    batch = make_padded_batch([sample["prompt_ids"] for sample in samples], [sample["response_ids"] for sample in samples])

    # 4. Log-probs under the weights that generated the samples (π_old)
    start = time.time()
    old_log_probs = compute_log_probs(policy, batch)
    timing["old_log_prob"] = time.time() - start

    # 5. Advantages with verl's GRPO estimator, which takes each response's reward on its last token
    last_token = batch["response_mask"].sum(-1).clamp(min=1) - 1
    token_level_rewards = torch.zeros(batch["response_mask"].shape)
    token_level_rewards[torch.arange(len(samples)), last_token] = rewards
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards, response_mask=batch["response_mask"], index=np.array(group_ids, dtype=object),
    )
    batch.update(old_log_probs=old_log_probs, advantages=advantages)  # kept for inspection after the update

    # 6. PPO update, with verl's loss
    start = time.time()
    stats = ppo_update(policy, optimizer, batch, old_log_probs, advantages, loss_fn=verl_ppo_loss, **update_kwargs)
    timing["update"] = time.time() - start

    metrics = {
        "reward": rewards.mean().item(),
        "format": float(np.mean([bool(BOXED.search(text)) for text in texts])),  # has a \boxed{integer} at all, right or wrong
        "response_len": batch["response_mask"].sum(-1).float().mean().item(),
        **stats,
    }
    if samples[0]["log_probs"] is not None:  # the inference engine returned its own log-probs: compare them with ours
        engine_log_probs = torch.zeros_like(old_log_probs)
        for row, sample in enumerate(samples):
            engine_log_probs[row, : len(sample["log_probs"])] = torch.tensor(sample["log_probs"])
        trained = batch["response_mask"].bool()
        metrics["logp_mismatch"] = (old_log_probs - engine_log_probs)[trained].abs().mean().item()
    metrics.update({f"time/{name}": seconds for name, seconds in timing.items()})
    return metrics, batch, samples


def print_step_metrics(step: int, metrics: dict) -> None:
    fmt = f" | format {metrics['format']:.2f}" if "format" in metrics else ""
    mismatch = f" | logp mismatch {metrics['logp_mismatch']:.4f}" if "logp_mismatch" in metrics else ""
    times = " ".join(f"{name.removeprefix('time/')} {seconds:4.1f}s" for name, seconds in metrics.items() if name.startswith("time/"))
    print(f"step {step:2d} | reward {metrics['reward']:.3f}{fmt} | len {metrics['response_len']:5.1f} | grad_norm {metrics['grad_norm']:.3f}{mismatch} | {times}")


def vllm_generate(llm, prompts: list[list[int]], samples_per_prompt: int, max_tokens: int = 256, temperature: float = 1.0) -> list[dict]:
    # logprobs=0: return the log-prob of each sampled token (and no alternatives)
    params = SamplingParams(n=samples_per_prompt, temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0, max_tokens=max_tokens, logprobs=0)
    token_prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]
    request_outputs = llm.generate(token_prompts, params, use_tqdm=False)
    samples = []
    for prompt, request_output in zip(prompts, request_outputs):
        for completion in request_output.outputs:  # samples_per_prompt completions per prompt
            response_ids = list(completion.token_ids)
            # completion.logprobs holds one {token_id: Logprob} dict per position; look up the token that was sampled
            log_probs = []
            for token_id, logprobs_at_position in zip(response_ids, completion.logprobs):
                log_probs.append(logprobs_at_position[token_id].logprob)
            samples.append({"prompt_ids": list(prompt), "response_ids": response_ids, "log_probs": log_probs})
    return samples


def save_for_inference(model, path: Path = SHM / "policy") -> str:
    path.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, tensor in model.state_dict().items():
        state[name] = tensor.detach().to(torch.bfloat16).cpu().contiguous()  # a bf16 copy on the CPU, ready to write
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)  # same tensor as the embedding; vLLM ties them itself
    safetensors.torch.save_file(state, str(path / "model.safetensors"))
    return str(path)


def push_weights(model, llm) -> None:
    weights_path = save_for_inference(model)                                     # 1. data plane: write the weights to RAM
    llm.collective_rpc("reload_weights", kwargs={"weights_path": weights_path})  # 2. control plane: tell vLLM to load them
    llm.reset_prefix_cache()  # cached KV values were computed with the old weights


def evaluate(generate_greedy, problems=EVAL) -> float:
    """Accuracy with one greedy response per problem."""
    samples = generate_greedy([chat_prompt_ids(problem["question"]) for problem in problems], 1)
    rewards = [reward_fn(decode_tokens(sample["response_ids"]), problem["answer"]) for sample, problem in zip(samples, problems)]
    return float(np.mean(rewards))


def plot_histories(histories: dict[str, list[dict]], keys=("reward", "response_len")) -> None:
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3))
    for ax, key in zip(np.atleast_1d(axes), keys):
        for name, history in histories.items():
            ax.plot([metrics[key] for metrics in history], marker="o", ms=3, label=name)
        ax.set_title(key)
        ax.set_xlabel("step")
    np.atleast_1d(axes)[0].legend()
    plt.tight_layout()
    plt.show()
