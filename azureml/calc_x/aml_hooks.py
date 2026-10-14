"""Ray worker setup hook for Azure ML jobs (configured in run_job.sh).

verl's mlflow logger logs the whole flattened config (~830 values) as MLflow params when it starts. Azure ML allows at
most 200 params per run, so verl's setup fails, verl gives up on MLflow for the whole job, and the warning only
appears inside a Ray worker. This hook keeps the params to the config values that differ from verl's defaults
(about 60 for Calc-X, including every command-line override). The defaults themselves are fixed by the verl
version, which is tagged on the job.
"""

from __future__ import annotations

import importlib.resources
from typing import Any

AZUREML_MAX_PARAMS = 190  # Azure ML's limit is 200 params per run


def setup_worker() -> None:
    # Agent Lightning's own hook (registers its policy losses in every Ray worker). Setting a hook replaces it.
    from agentlightning.verl.per_rollout_loss import register_in_worker

    register_in_worker()
    _log_only_non_default_params()


def _log_only_non_default_params() -> None:
    import verl.utils.tracking as tracking

    flatten = tracking._compute_mlflow_params_from_objects
    if getattr(flatten, "_azureml_non_default", False):
        return

    def non_default_params(params: Any) -> dict[str, Any]:
        defaults = flatten(_verl_default_config())
        changed = {key: value for key, value in flatten(params).items() if str(defaults.get(key)) != str(value)}
        return dict(sorted(changed.items())[:AZUREML_MAX_PARAMS])

    non_default_params._azureml_non_default = True  # type: ignore[attr-defined]
    tracking._compute_mlflow_params_from_objects = non_default_params


def _verl_default_config() -> dict[str, Any]:
    from omegaconf import OmegaConf

    path = importlib.resources.files("verl.trainer.config") / "_generated_ppo_trainer.yaml"
    return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)  # type: ignore[return-value]
