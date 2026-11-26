"""Notebook 3, §6: a longer verl + Agent Lightning training run of the Calc-X agent, for the article to plot.

Uses the Calc-X recipe unchanged (examples/calc_x/train_calc_agent.py: 32 questions x 4 rollouts per step,
Qwen2.5-1.5B-Instruct), except for the number of steps, and validates on the 500-question test set before training
and every EVAL_EVERY steps. verl's per-step metrics (one JSON line per step) are written to
tutorial/series/results/calc_x_long_run.jsonl, which the article reads. The final weights are saved in Hugging Face
format to CHECKPOINT_DIR/global_step_<N>/actor/huggingface, for tools/calc_x_followed.py.

With --async, the same recipe runs with Agent Lightning's asynchronous rollouts (train_calc_agent.py --async: twice
as many questions are queued as each step trains on, and unfinished rollouts carry over to the next step), without
validation, and writes to tutorial/series/results/calc_x_async_run.jsonl, to compare with the synchronous run's steps.

Usage: .venv/bin/python tools/calc_x_long_run.py [NUM_STEPS] [--async]   (default 80 steps)
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tutorial/lib"))
OUT = REPO / "tutorial/series/results/calc_x_long_run.jsonl"
ASYNC_OUT = REPO / "tutorial/series/results/calc_x_async_run.jsonl"
EVAL_EVERY = 10
CHECKPOINT_DIR = Path("/tmp/rl-from-scratch/checkpoints/long_run")
AGL_PORT, AGL_KEY = 8284, "calc-x-long-run"  # a different port from the notebook's, so the two can't clash


def main() -> None:
    from common import CALC_X_DATA, CALC_X_MODEL, WORK, follow_log, log_tail, setup, start_service, stop_all, wait_until_healthy

    is_async = "--async" in sys.argv
    numbers = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    num_steps = int(numbers[0]) if numbers else 80
    out = ASYNC_OUT if is_async else OUT
    if is_async:
        script_flags = ["--async", "--run-name", "async_run"]
        validation = ["trainer.val_before_train=False", f"trainer.test_freq={num_steps + 1}"]  # no validation
    else:
        script_flags = ["--run-name", "long_run"]
        validation = ["trainer.val_before_train=True", f"trainer.test_freq={EVAL_EVERY}"]
    # Save only the final weights, in Hugging Face format (the synchronous run only)
    if is_async:
        saving = ["trainer.save_freq=-1"]
    else:
        saving = [f"trainer.save_freq={num_steps}", f"trainer.default_local_dir={CHECKPOINT_DIR}",
                  "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]"]
    setup()
    agl_url = f"http://127.0.0.1:{AGL_PORT}"

    start_service("long_agl_gateway", [
        sys.executable, "-m", "agentlightning.server", "host=127.0.0.1", f"port={AGL_PORT}", f"key={AGL_KEY}",
        f"default_proxy.model_name={CALC_X_MODEL}", f"hydra.run.dir={WORK}/hydra/long_gateway", "hydra.output_subdir=null",
    ])
    wait_until_healthy("long_agl_gateway", f"{agl_url}/healthz")
    start_service("long_agl_controller", [
        sys.executable, "-m", "agentlightning.controller", "runner_type=local", f"agl_server.url={agl_url}", f"agl_server.key={AGL_KEY}",
        "local_runner.poll_interval=1", "local_runner.maximum_size=128",
        f"hydra.run.dir={WORK}/hydra/long_controller", "hydra.output_subdir=null",
    ], cwd=REPO / "examples/calc_x")
    time.sleep(3)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    start = time.time()
    try:
        start_service("long_agl_trainer", [
            sys.executable, "train_calc_agent.py", "--agl-base-url", agl_url, "--agl-key", AGL_KEY, "--model", CALC_X_MODEL,
            *script_flags,
            "--train-file", str(CALC_X_DATA / "train.parquet"), "--val-file", str(CALC_X_DATA / "test.parquet"),
            f"trainer.total_training_steps={num_steps}", *validation,
            *saving, "trainer.logger=[console,file]",
        ], cwd=REPO / "examples/calc_x", env={"VERL_FILE_LOGGER_PATH": str(out), "WANDB_MODE": "disabled"})
        exit_code = follow_log("long_agl_trainer", r"Initial validation|Final validation|Training Progress|step:\d+ ")
        print(f"\ntrainer exited with code {exit_code} after {(time.time() - start) / 60:.0f} min; metrics in {out}")
        if exit_code != 0:
            print(log_tail("long_agl_trainer", 60))
            sys.exit(exit_code)
    finally:
        stop_all()


if __name__ == "__main__":
    main()
