#!/usr/bin/env bash
# Executes tutorial/verl_from_scratch.ipynb top to bottom on one GPU and saves the executed copy to outputs/.
set -uo pipefail

export CALC_X_DATA="$(realpath "$1")"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs
nvidia-smi --query-gpu=name,memory.total --format=csv
uv pip install --quiet nbclient nbformat

python - <<'EOF'
import time

import nbformat
from nbclient import NotebookClient

nb = nbformat.read("tutorial/verl_from_scratch.ipynb", as_version=4)
t0 = time.time()


def progress(cell, cell_index, execute_reply):
    errors = [o.get("ename") for o in cell.get("outputs", []) if o.get("output_type") == "error"]
    print(f"[{time.time() - t0:6.0f}s] cell {cell_index:3d} {'ERROR ' + str(errors) if errors else 'ok'} | {cell.source.splitlines()[0][:80]}", flush=True)
    nbformat.write(nb, "outputs/verl_from_scratch.executed.ipynb")


client = NotebookClient(nb, timeout=5400, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": "tutorial"}}, on_cell_executed=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, "outputs/verl_from_scratch.executed.ipynb")
    print(f"executed in {time.time() - t0:.0f}s")
errors = [(i, o.get("ename"), o.get("evalue", "")[:300]) for i, c in enumerate(nb.cells) if c.cell_type == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
print("ERRORS:", errors if errors else "none")
EOF

tar czf outputs/work_logs.tgz -C /tmp verl-from-scratch/logs verl-from-scratch/agents verl-from-scratch/verl_rollouts verl-from-scratch/verl_val 2>/dev/null || true
