"""Article 3, §6: the followed Calc-X question's rollouts, before and after the long run.

Serves a model with vLLM, records ROLLOUTS_PER_MODEL rollouts of the question that article 3 follows from §3
(Calc-X train question FOLLOWED_INDEX) through the article's recording gateway, and saves the records, under the
model's label, to tutorial/series/results/calc_x_followed.json, which the article reads.

Usage: .venv/bin/python tools/calc_x_followed.py LABEL [MODEL_PATH]
  e.g. ... before                     (the original Qwen2.5-1.5B-Instruct)
       ... after /tmp/rl-from-scratch/checkpoints/long_run/global_step_100/actor/huggingface
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tutorial/lib"))
OUT = REPO / "tutorial/series/results/calc_x_followed.json"
FOLLOWED_INDEX = 1  # "How many degrees of 75° can form a right angle?"
ROLLOUTS_PER_MODEL = 16


def main() -> None:
    import pandas as pd

    from common import CALC_X_DATA, CALC_X_MODEL, SEED, VENV_BIN, setup, start_service, stop_all, wait_until_healthy
    from recording_gateway import RECORDS, VLLM_PORT, VLLM_ROOT, run_agents, start_gateway

    label = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else CALC_X_MODEL
    setup()
    row = pd.read_parquet(CALC_X_DATA / "train.parquet").iloc[FOLLOWED_INDEX]

    try:
        # The same server flags as article 3 §3; the gateway asks for CALC_X_MODEL by name, so serve the checkpoint under it
        start_service("followed_vllm", [
            str(VENV_BIN / "vllm"), "serve", model_path, "--served-model-name", CALC_X_MODEL,
            "--host", "127.0.0.1", "--port", str(VLLM_PORT),
            "--gpu-memory-utilization", "0.3", "--max-model-len", "8192", "--seed", str(SEED),
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--generation-config", "vllm",
        ])
        wait_until_healthy("followed_vllm", f"{VLLM_ROOT}/health")
        server = start_gateway()
        rollout_ids = [f"{label}-{attempt}" for attempt in range(ROLLOUTS_PER_MODEL)]
        run_agents([(rollout_id, row["question"], row["result"]) for rollout_id in rollout_ids])
        server.should_exit = True
    finally:
        stop_all()

    saved = json.loads(OUT.read_text()) if OUT.exists() else {}
    saved[label] = {"model": model_path, "question": row["question"], "result": str(row["result"]),
                    "rollouts": {rollout_id: RECORDS.get(rollout_id, {"calls": [], "reward": None}) for rollout_id in rollout_ids}}
    OUT.write_text(json.dumps(saved))
    rewards = [saved[label]["rollouts"][rollout_id]["reward"] for rollout_id in rollout_ids]
    print(f"{label}: {row['question']!r} rewards {rewards} -> {OUT}")


if __name__ == "__main__":
    main()
