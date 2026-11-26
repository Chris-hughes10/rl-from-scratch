# AzureML jobs

Runs the Calc-X GRPO training (`examples/calc_x`) as an Azure ML command job. Everything is driven from the repo-root
`Makefile`, configured by `config.env` (placeholders; put your own workspace, resource group and cluster in a
git-ignored `config.local.env`, which overrides it):

```
config.env                       # workspace, resource group, compute cluster, paths (placeholders)
config.local.env                 # your real values (git-ignored)
Makefile                         # `make help` lists the commands
azureml/calc_x/
├── azure-ml-job.yaml            # job spec: command, inputs/outputs, environment, compute
├── run_job.sh                   # starts Ray, the Gateway and the Controller, then the trainer (same flow as run_local.sh)
├── aml_hooks.py                 # Ray worker setup hook: keeps verl's MLflow params under Azure ML's 200-param limit
└── environment/
    ├── Dockerfile               # Python 3.12 + pinned GPU stack + MLflow
    ├── requirements.txt         # exact versions of the locally verified .venv (incl. agentlightning), --no-deps
    └── wheels/                  # prebuilt flash-attn wheel (gitignored; `make flash-attn-wheel exp=calc_x`)
```

The job was developed on a Standard_NC80adis_H100_v5 cluster (2× H100). If your workspace's storage uses
identity-based access, give the cluster a managed identity with access to it.

## One-time setup

```bash
# Azure CLI with only the v2 `ml` extension (the system extension folder also has the conflicting v1 `azure-cli-ml`)
AZURE_EXTENSION_DIR=~/.azure/cliextensions az extension add -n ml
az login

make flash-attn-wheel exp=calc_x   # ~10 min on a 40-core machine; needs the repo .venv set up with setup_verl.sh
make data exp=calc_x               # uploads examples/calc_x/data/*.parquet as the `calc-x` data asset
```

## Running

```bash
make build-exp exp=calc_x          # build the Docker image locally
make local exp=calc_x script-xargs="trainer.total_training_steps=2 trainer.save_freq=-1 --val-file /mnt/data/test_mini.parquet"

# AzureML: smoke test first (2 steps, 20 validation questions), then the full recipe (512 steps)
make job exp=calc_x job-xargs="--set inputs.overrides='trainer.total_training_steps=2' inputs.val_file=test_mini.parquet"
make job exp=calc_x
```

Both `make local` and `make job` first copy the code a run needs into `azureml/isolated_runs/<run_name>/` and run from
there, so a local run executes exactly what would be submitted.

## Logging

`run_job.sh` enables verl's `mlflow` logger when `MLFLOW_TRACKING_URI` is set. Inside an AzureML job that's always
the case, and verl attaches to the job's own run (`MLFLOW_RUN_ID`), so all metrics appear on the job in Studio. The
script checks up front that the run is writable and stops otherwise. Metrics are also written to
`outputs/metrics.jsonl` as a backup.

verl logs its whole flattened config (~830 values) as MLflow params, but Azure ML allows only 200 per run. verl then
silently turns MLflow off for the whole job, with a warning hidden inside a Ray worker. `aml_hooks.py` therefore
logs only the config values that differ from verl's defaults (about 60, including every command-line override).

Checkpoints (about 19 GB each) go to the job's `checkpoints` output; only the latest is kept.

## Multi-GPU note

Agent Lightning's Ray worker setup hook imports verl as each Ray worker starts, and that initialises CUDA before
Ray assigns the worker its GPU. On multi-GPU nodes every training worker then ends up on GPU 0, and NCCL fails with
`Duplicate GPU detected`. `run_job.sh` sets `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, so verl selects each
worker's GPU explicitly instead.
