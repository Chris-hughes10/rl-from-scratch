#!/usr/bin/env bash
# Run the Calc-X quick start fully locally (no W&B).
# Metrics go to the console, examples/calc_x/metrics_log/ (JSONL) and examples/calc_x/tensorboard_log/.
# Extra args are passed through as verl/OmegaConf overrides, e.g. trainer.total_training_steps=2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
source "$ROOT/.venv/bin/activate"

export WANDB_MODE=disabled
export VERL_FILE_LOGGER_ROOT="$ROOT/examples/calc_x/metrics_log"

cd "$ROOT/examples/calc_x"
exec bash run_local.sh 'trainer.logger=[console,file,tensorboard]' "$@"
