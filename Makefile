# All variables referenced in UPPER_CAPS come from configuration (config.env).
# All variables referenced in lower_caps are arguments to be passed (some with defaults).
include config.env
export AZURE_EXTENSION_DIR

.PHONY: help check-exp isolate-run flash-attn-wheel build-exp local data job

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-17s %s\n", $$1, $$2}'

check-exp:
	@test -n "$(exp)" || { echo "You must pass the 'exp' argument, e.g. exp=calc_x"; exit 1; }
	@test -d "$(CODE_PATH)/$(exp)" || { echo "Experiment '$(exp)' not found in $(CODE_PATH)"; exit 1; }

run_name := $(exp)_$(shell date +%Y%m%d_%H%M%S)
run_dir = $(ISOLATED_RUNS_PATH)/$(run_name)

# Copies exactly what a run needs (the Calc-X code, the tutorials, the experiment's job spec and scripts) into a fresh
# folder. Local runs and AzureML jobs both execute from it, and the AzureML code snapshot stays small.
# agentlightning itself comes from the image (environment/requirements.txt).
isolate-run: check-exp
	@mkdir -p $(run_dir)/examples/calc_x $(run_dir)/tutorial
	@rsync -a --exclude __pycache__ --include '*.py' --include '*.yaml' --exclude '*' examples/calc_x/ $(run_dir)/examples/calc_x/
	@cp tutorial/*.ipynb $(run_dir)/tutorial/
	@find $(CODE_PATH)/$(exp) -maxdepth 1 -type f -exec cp {} $(run_dir)/ \;
	@echo "Run isolated in: $(run_dir)"

# Needs the repo .venv set up with `bash scripts/setup_verl.sh 0.8.0 cu130` (its torch + CUDA 13.0 toolchain).
flash-attn-wheel: check-exp ## Compile the flash-attn wheel the Docker image installs (once, ~15 min): make flash-attn-wheel exp=calc_x
	@cuda_home=$$(.venv/bin/python -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))') && \
	CUDA_HOME=$$cuda_home PATH=$$cuda_home/bin:$(PWD)/.venv/bin:$$PATH \
	LIBRARY_PATH=$$cuda_home/lib LD_LIBRARY_PATH=$$cuda_home/lib CPATH=$$cuda_home/include \
	FLASH_ATTN_CUDA_ARCHS="80;90" TORCH_CUDA_ARCH_LIST="8.0;9.0" FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=32 NVCC_THREADS=2 \
	.venv/bin/pip wheel "flash-attn==2.8.3" --no-build-isolation --no-deps --no-binary flash-attn --no-cache-dir \
		-w $(CODE_PATH)/calc_x/environment/wheels

build-exp: check-exp ## Build the experiment's Docker image: make build-exp exp=calc_x
	docker build --tag $(exp):latest $(build-xargs) $(CODE_PATH)/calc_x/environment

# Lines as `<command>: var=val` define defaults for optional arguments.
local: n_gpus=1
local: build-exp isolate-run ## Run an experiment in its Docker image on this machine: make local exp=calc_x [n_gpus=1] [script-xargs='...']
	docker run --rm --gpus all --ipc host --user $$(id -u):$$(id -g) -e USER=$$(id -un) \
		-e HOME=/tmp -e HF_HOME=/hf -v $(HOME)/.cache/huggingface:/hf \
		-v $(abspath $(run_dir)):/mnt/run -v $(abspath examples/calc_x/data):/mnt/data:ro \
		--workdir /mnt/run $(run-xargs) $(exp):latest \
		bash run_job.sh /mnt/data /mnt/run/outputs/checkpoints trainer.n_gpus_per_node=$(n_gpus) $(script-xargs)

data: check-exp ## Upload examples/<exp>/data/*.parquet as a data asset (once): make data exp=calc_x
	@tmp=$$(mktemp -d) && cp examples/$(exp)/data/*.parquet $$tmp/ && \
	az ml data create --name $(subst _,-,$(exp)) --type uri_folder --path $$tmp \
		--description "Parquet files from examples/$(exp)/data" \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE); \
	status=$$?; rm -rf $$tmp; exit $$status

job: isolate-run ## Submit an experiment to Azure ML: make job exp=calc_x [job-xargs="--set inputs.overrides='trainer.total_training_steps=2'"]
	az ml job create -f $(run_dir)/azure-ml-job.yaml --name $(run_name) \
		--resource-group $(RESOURCE_GROUP) --workspace-name $(WORKSPACE) $(job-xargs)
