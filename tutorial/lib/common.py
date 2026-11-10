"""Shared setup for the tutorial series: paths, environment, seeds, and helpers for background services."""

import atexit
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "tutorial/lib"                 # shared code: this module, arith_task, rl_core, ...
GENERATED = REPO / "tutorial/generated"     # files the notebooks write with %%writefile (git-ignored)
WORK = Path("/tmp/rl-from-scratch")         # datasets, logs, metrics
SHM = Path("/dev/shm/rl-from-scratch")      # weight hand-off from trainer to inference engine (RAM, not disk)
VENV_BIN = Path(sys.executable).parent
CALC_X_DATA = Path(os.environ.get("CALC_X_DATA", REPO / "examples/calc_x/data"))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"        # the policy for the arithmetic task
CALC_X_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" # the policy for Calc-X (same tokenizer)
DEVICE = "cuda"
SEED = 0


def setup() -> None:
    """Create the working folders, set the environment child processes inherit, seed, and check the GPU."""
    for directory in (WORK, SHM, WORK / "logs", GENERATED):
        directory.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)  # so %%writefile with relative paths resolves correctly

    # Child processes (Ray workers, vLLM, agents, verl) inherit these
    search_path = [str(LIB), str(GENERATED), str(WORK), str(REPO)]
    for path in reversed(search_path):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["PATH"] = f"{VENV_BIN}:{os.environ['PATH']}"
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [*search_path, os.environ.get("PYTHONPATH", "")] if p)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"repo: {REPO}\npython: {sys.executable}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, {free_bytes / 2**30:.0f} of {total_bytes / 2**30:.0f} GiB free")
    if free_bytes < 0.85 * total_bytes:
        print("WARNING: something else is using this GPU. Stop it before continuing, or later stages may run out of memory.")


# ── background services (vLLM servers, gateways, trainers) ───────────────────
SERVICES: dict[str, subprocess.Popen] = {}


def start_service(name: str, cmd: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None) -> subprocess.Popen:
    stop_service(name)
    with open(WORK / "logs" / f"{name}.log", "w") as log:
        full_env = {**os.environ, **(env or {})}  # our environment, plus any extra variables
        # start_new_session: the service gets its own process group, so stop_service can kill it and all its children
        proc = subprocess.Popen(cmd, cwd=cwd, env=full_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    SERVICES[name] = proc
    print(f"started {name} (pid {proc.pid}), log: {WORK / 'logs' / f'{name}.log'}")
    return proc


def stop_service(name: str, timeout: float = 60) -> None:
    proc = SERVICES.pop(name, None)
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    print(f"stopped {name}")


def log_tail(name: str, num_lines: int = 30) -> str:
    path = WORK / "logs" / f"{name}.log"
    if not path.exists():
        return "(no log)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-num_lines:])


def wait_until_healthy(name: str, url: str, timeout: float = 900) -> None:
    start = time.time()
    while time.time() - start < timeout:
        proc = SERVICES.get(name)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{name} exited with code {proc.returncode}:\n{log_tail(name)}")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                print(f"{name} is healthy after {time.time() - start:.0f}s")
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{name} not healthy after {timeout}s:\n{log_tail(name)}")


def follow_log(name: str, pattern: str) -> int:
    """Print matching lines of a service's log until it exits; return its exit code."""
    proc = SERVICES[name]
    line_filter = re.compile(pattern)
    start = time.time()
    with open(WORK / "logs" / f"{name}.log", errors="replace") as log:
        while True:
            line = log.readline()
            if not line:
                if proc.poll() is not None:
                    return proc.returncode
                time.sleep(1)
                continue
            if line_filter.search(line):
                line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()  # drop terminal colour codes
                print(f"[{time.time() - start:5.0f}s] {line[:220]}{' …' if len(line) > 220 else ''}")


def gpu_free_gib() -> float:
    return torch.cuda.mem_get_info()[0] / 2**30


def stop_all() -> None:
    for name in reversed(list(SERVICES)):
        stop_service(name)


atexit.register(stop_all)  # don't leave vLLM or a gateway running if the kernel exits mid-notebook
