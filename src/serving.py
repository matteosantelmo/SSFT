"""Helpers for waiting on an sml serving job and the model it exposes.

The model is reached through the Swiss AI serving-api gateway: the vllm replicas
register on the OpenTela mesh under one model id, and the gateway load-balances
requests across them. These helpers wait for the Slurm job to start and for the
model to actually answer, and resolve the CSCS API key used to authenticate.
"""

from __future__ import annotations

import os
import subprocess
import time

GATEWAY_URL = "https://api.swissai.svc.cscs.ch/v1"

# Slurm states a job never recovers from — seeing one means it will not serve.
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE",
})


def load_api_key() -> str:
    """Return the CSCS Serving API key."""
    key = os.environ.get("CSCS_API_KEY")
    if not key:
        from swiss_ai_model_launch.cli.configuration import InitConfig

        key = InitConfig.load().get_non_none_value("cscs_api_key")
    if not key:
        raise RuntimeError("No CSCS API key: set $CSCS_API_KEY or run `sml init`.")
    return key


def job_state(job_id: int) -> str:
    """Return the Slurm state of ``job_id`` (squeue while queued, else sacct)."""
    queued = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True, text=True,
    ).stdout.strip()
    if queued:
        return queued
    finished = subprocess.run(
        ["sacct", "-j", str(job_id), "-n", "-o", "State", "--parsable2"],
        capture_output=True, text=True,
    ).stdout.strip()
    return finished.splitlines()[0].split()[0] if finished else "UNKNOWN"


def wait_for_running(job_id: int, timeout: int, poll: int = 10) -> None:
    """Block until ``job_id`` is RUNNING.

    Raises ``RuntimeError`` if the job reaches a terminal state first, or
    ``TimeoutError`` if it is still not running after ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    seen = None
    while time.monotonic() < deadline:
        state = job_state(job_id)
        if state != seen:
            print(f"[wait] job {job_id} state: {state}", flush=True)
            seen = state
        if state == "RUNNING":
            return
        if state in _TERMINAL_STATES:
            raise RuntimeError(f"job {job_id} reached terminal state {state} before serving.")
        time.sleep(poll)
    raise TimeoutError(f"job {job_id} did not start running within {timeout}s.")


def wait_for_model(client, model: str, timeout: int, job_id: int | None = None, poll: int = 15) -> None:
    """Block until ``model`` answers a minimal completion through the gateway.

    If ``job_id`` is given, abort early (``RuntimeError``) should the serving job
    die while loading. Raises ``TimeoutError`` if the model never answers in time.
    """
    print(f"[wait] probing model '{model}' on {client.base_url} ...", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_id is not None and job_state(job_id) in _TERMINAL_STATES:
            raise RuntimeError(f"serving job {job_id} ended before model '{model}' was ready.")
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
        except Exception as exc:
            # The model is still loading or the gateway hasn't seen it yet; both
            # surface as assorted openai/httpx errors, so retry until the deadline.
            remaining = int(deadline - time.monotonic())
            print(f"[wait] not ready ({type(exc).__name__}); ~{remaining}s left", flush=True)
            time.sleep(poll)
        else:
            print(f"[ready] model '{model}' is answering.", flush=True)
            return
    raise TimeoutError(f"model '{model}' did not become ready within {timeout}s.")
