
"""Notebook 3's recording gateway (§8.1) and agent launcher, so notebook 4 can import them.

Copied verbatim from notebook 3, which explains every line. The gateway forwards an agent's chat requests to the
policy's vLLM server and records the exact token IDs and log-probs under the rollout's ID.
A judge model must never be called through it: everything on these URLs is recorded as the policy's own output.
"""

import os
import subprocess
import sys
import threading
import time

import httpx

from common import CALC_X_MODEL, REPO, WORK

VLLM_PORT = 8110
VLLM_ROOT = f"http://127.0.0.1:{VLLM_PORT}"

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

RECORDS: dict[str, dict] = {}
gateway = FastAPI()


def get_record(rollout_id: str) -> dict:
    return RECORDS.setdefault(rollout_id, {"calls": [], "reward": None})


@gateway.get("/healthz")
async def healthz():
    return {"ok": True}


@gateway.post("/rollout/{rollout_id}/v1/chat/completions")
async def chat_completions(rollout_id: str, request: Request):
    body = await request.json()
    body.update(model=CALC_X_MODEL, temperature=1.0, top_p=1.0, logprobs=True, return_token_ids=True, stream=False)
    async with httpx.AsyncClient(timeout=300) as client:
        upstream = await client.post(f"{VLLM_ROOT}/v1/chat/completions", json=body)
    data = upstream.json()
    if upstream.status_code == 200:
        choice = data["choices"][0]
        get_record(rollout_id)["calls"].append({
            "prompt_ids": data.get("prompt_token_ids") or choice.get("prompt_token_ids"),
            "response_ids": choice["token_ids"],
            "log_probs": [t["logprob"] for t in choice["logprobs"]["content"]],
        })
    return JSONResponse(data, status_code=upstream.status_code)


@gateway.post("/rollout/{rollout_id}/events")
async def events(rollout_id: str, request: Request):
    event = await request.json()
    if event.get("event_type") == "reward":
        get_record(rollout_id)["reward"] = float(event["data"]["value"])
    return {"ok": True}


GATEWAY_PORT = 8291
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}"


def start_gateway() -> "uvicorn.Server":
    """Serve the gateway from a background thread of this process; returns the server (set .should_exit to stop it)."""
    server = uvicorn.Server(uvicorn.Config(gateway, host="127.0.0.1", port=GATEWAY_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server


(WORK / "agents").mkdir(parents=True, exist_ok=True)


def launch_agent(rollout_id: str, question: str, result) -> subprocess.Popen:
    env = {
        **os.environ, "QUESTION": question, "RESULT": str(result), "AGL_KEY": "not-checked-by-our-gateway",
        "AGL_OPENAI_BASE_URL": f"{GATEWAY_URL}/rollout/{rollout_id}/v1",
        "AGL_EVENT_URL": f"{GATEWAY_URL}/rollout/{rollout_id}/events",
    }
    with open(WORK / "agents" / f"{rollout_id}.log", "w") as log:
        return subprocess.Popen(
            [sys.executable, "-c", "import asyncio, calc_agent; asyncio.run(calc_agent.Agent().run())"],
            cwd=REPO / "examples/calc_x", env=env, stdout=log, stderr=subprocess.STDOUT,
        )


def run_agents(jobs: list[tuple[str, str, object]], timeout: float = 300) -> dict[str, int]:
    """jobs: (rollout_id, question, result). Runs one agent process per job, all at once; returns their exit codes."""
    procs = {rollout_id: launch_agent(rollout_id, question, result) for rollout_id, question, result in jobs}
    deadline = time.time() + timeout
    for proc in procs.values():
        try:
            proc.wait(max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {rollout_id: proc.returncode for rollout_id, proc in procs.items()}
