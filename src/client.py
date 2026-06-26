#!/usr/bin/env python3
"""Query the SSFT serving PoC through the Swiss AI serving-api gateway.

Waits for the Slurm serving job and the gateway-served model to be ready, then
sends a few example chat completions (temperature 0.7, top-p 0.95) and prints the
outputs. The launching half of the pipeline lives in ``scripts/run_poc.sh``.
"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI

from serving import GATEWAY_URL, load_api_key, wait_for_model, wait_for_running

EXAMPLE_PROMPTS = [
    "In one sentence, what is the Swiss National Supercomputing Centre (CSCS)?",
    "Write a haiku about tensor parallelism.",
    "What is 17 * 24? Show your reasoning briefly, then give the final answer.",
]


def run_queries(client: OpenAI, model: str, temperature: float, top_p: float, max_tokens: int) -> None:
    """Send each example prompt to ``model`` and print the response."""
    print("\n" + "=" * 72)
    print(f"Serving endpoint : {client.base_url}")
    print(f"Model id         : {model}")
    print(f"Sampling         : temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    print("=" * 72)

    for i, prompt in enumerate(EXAMPLE_PROMPTS, 1):
        print(f"\n----- query {i}/{len(EXAMPLE_PROMPTS)} -----")
        print(f"prompt: {prompt}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        print(f"output: {response.choices[0].message.content}")
        if response.usage:
            print(f"tokens: prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens}")

    print("\n" + "=" * 72)
    print("Done. (The serving job keeps running until its time limit or scancel.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for the SSFT serving job, then query it via the gateway.")
    parser.add_argument("--served-model-name", required=True, help="Model id to request (matches the launched name).")
    parser.add_argument("--job-id", type=int, default=None, help="Slurm job id to watch (fail fast if it dies).")
    parser.add_argument("--base-url", default=GATEWAY_URL, help="OpenAI-compatible endpoint (default: serving-api gateway).")
    parser.add_argument("--job-timeout", type=int, default=1800, help="Seconds to wait for the job to start RUNNING.")
    parser.add_argument("--ready-timeout", type=int, default=1800, help="Seconds to wait for the model to answer.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        client = OpenAI(base_url=args.base_url, api_key=load_api_key())
        if args.job_id is not None:
            wait_for_running(args.job_id, args.job_timeout)
        wait_for_model(client, args.served_model_name, args.ready_timeout, job_id=args.job_id)
    except (RuntimeError, TimeoutError) as exc:
        sys.exit(f"[fatal] {exc}")
    run_queries(client, args.served_model_name, args.temperature, args.top_p, args.max_tokens)


if __name__ == "__main__":
    main()
