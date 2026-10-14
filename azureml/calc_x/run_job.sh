#!/usr/bin/env bash
# Run Calc-X GRPO training in a single container: Ray, the Agent Lightning Gateway and Controller, then the trainer.
# Same flow as examples/calc_x/run_local.sh, adapted to a job. Used by the AzureML job and by `make local`.
#
# Usage: bash run_job.sh <data_dir> <checkpoint_dir> [train_calc_agent.py args / verl overrides ...]
#   <data_dir>        folder with train.parquet and test.parquet (the `calc-x` data asset)
#   <checkpoint_dir>  where verl writes checkpoints (only the latest one is kept)
# Run it from the folder `make job` / `make local` creates, which holds examples/calc_x/. agentlightning is in the image.
set -euo pipefail

DATA_DIR="$(realpath "$1")"
CKPT_DIR="$(realpath -m "$2")"
shift 2

ROOT="$PWD"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
AGL_PORT=8181
AGL_KEY=calc-x
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Log metrics to MLflow when a tracking server is configured: AzureML sets MLFLOW_TRACKING_URI and MLFLOW_RUN_ID in
# the job, and verl's mlflow logger then writes to the job's own run. The worker setup hook in aml_hooks.py keeps
# verl's param logging within Azure ML's 200-param limit. verl only warns (from inside a Ray worker, so invisibly) if
# MLflow fails, so check up front that the job's run is writable. Metrics also always go to outputs/metrics.jsonl.
LOGGERS="[console,file${MLFLOW_TRACKING_URI:+,mlflow}]"
export VERL_FILE_LOGGER_PATH="$ROOT/outputs/metrics.jsonl"
if [[ -n "${MLFLOW_RUN_ID:-}" ]]; then
    python -c "import mlflow, os; mlflow.MlflowClient().set_tag(os.environ['MLFLOW_RUN_ID'], 'mlflow_check', 'ok')" \
        || { echo "ERROR: cannot log to MLflow run $MLFLOW_RUN_ID at $MLFLOW_TRACKING_URI"; exit 1; }
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Multi-GPU fix: Agent Lightning's Ray worker_process_setup_hook (agentlightning/verl/per_rollout_loss.py) imports
# verl as each Ray worker starts, which initialises CUDA before Ray sets the worker's CUDA_VISIBLE_DEVICES. Every
# training worker then lands on the first GPU and NCCL fails with "Duplicate GPU detected". With this flag Ray leaves
# CUDA_VISIBLE_DEVICES alone and verl selects each worker's GPU explicitly (verl's supported path for this flag).
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Ray must start before the trainer so its workers inherit this environment (MLFLOW_*, AzureML auth).
ray start --head --disable-usage-stats --include-dashboard=false

python -m agentlightning.server \
    host=127.0.0.1 port="$AGL_PORT" key="$AGL_KEY" default_proxy.model_name="$MODEL" \
    hydra.run.dir="$LOG_DIR/hydra/server" hydra.output_subdir=null \
    >"$LOG_DIR/agl-server.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$AGL_PORT/healthz" >/dev/null && break
    sleep 1
done
curl -sf "http://127.0.0.1:$AGL_PORT/healthz" >/dev/null || { echo "Gateway did not start:"; cat "$LOG_DIR/agl-server.log"; exit 1; }

# The Controller runs from examples/calc_x so the spawned agents can `import eval_utils`.
# Its defaults (check every 10 s, at most 50 agents at once) made rollouts ~60 s of each ~70 s step: every rollout
# waited for 10 s checks and a step's 128 rollouts (32 questions x 4) ran in 3 waves. Check every second and run a
# whole step's rollouts at once instead.
(cd examples/calc_x && exec python -m agentlightning.controller \
    runner_type=local agl_server.url="http://127.0.0.1:$AGL_PORT" agl_server.key="$AGL_KEY" \
    local_runner.poll_interval=1 local_runner.maximum_size=128 \
    hydra.run.dir="$LOG_DIR/hydra/controller" hydra.output_subdir=null) \
    >"$LOG_DIR/agl-controller.log" 2>&1 &
PIDS+=($!)

cd examples/calc_x
python train_calc_agent.py \
    --agl-base-url "http://127.0.0.1:$AGL_PORT" --agl-key "$AGL_KEY" --model "$MODEL" --run-name aml \
    --train-file "$DATA_DIR/train.parquet" --val-file "$DATA_DIR/test.parquet" \
    "trainer.logger=$LOGGERS" \
    "trainer.default_local_dir=$CKPT_DIR" \
    trainer.max_actor_ckpt_to_keep=1 \
    ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=aml_hooks.setup_worker \
    "$@"
