#!/usr/bin/env python3
"""Scalable generation + verification pipeline for SSFT.

Given a verl-format parquet of prompts, this:
  1. waits for the sml serving job + model to be ready (gateway load-balances
     across all replicas under one served-model-name);
  2. generates responses with high async concurrency through the OpenAI API;
  3. verifies each response immediately with verl's reward-score verifiers
     (overlapping verification with in-flight generations);
  4. streams every {id, score, response, ...} record to results.jsonl as it
     completes, so multi-hour runs persist incrementally and can be resumed.

The launching half lives in scripts/run_pipeline.sh. Verification scope and the
verifier wrapper live in src/verifier.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from serving import GATEWAY_URL, load_api_key, wait_for_model, wait_for_running
from verifier import verify

RESULTS_FILENAME = "results.jsonl"


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _json_default(o):
    """JSON fallback for the verl-format payloads (extra_info has numpy values)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _to_messages(prompt) -> list[dict]:
    """Normalize a parquet prompt cell into a plain list of chat messages."""
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


def _row_index(extra_info, fallback) -> str:
    if isinstance(extra_info, dict) and extra_info.get("index") is not None:
        return str(extra_info["index"])
    return str(fallback)


_REASON_OPEN = ("<|inner_prefix|>", "<think>")
_REASON_CLOSE = ("<|inner_suffix|>", "</think>")

def split_reasoning(content: str | None):
    """Split content into (reasoning, answer). reasoning is None if no block."""
    text = content or ""
    lo, lo_tok = -1, ""
    for m in _REASON_OPEN:
        i = text.find(m)
        if i != -1 and (lo == -1 or i < lo):
            lo, lo_tok = i, m
    if lo == -1:
        return None, text  # no reasoning block -> all answer
    rest = text[lo + len(lo_tok):]
    hi, hi_len = -1, 0
    for m in _REASON_CLOSE:
        j = rest.find(m)
        if j != -1 and (hi == -1 or j < hi):
            hi, hi_len = j, len(m)
    if hi == -1:
        return rest, ""  # opened but never closed -> no usable answer
    return rest[:hi], rest[hi + hi_len:]


def build_work_items(df: pd.DataFrame, repeats: int, seed: int) -> list[dict]:
    """Expand rows into (row x repeat) work items with stable ids.

    Each repeat gets a distinct request seed (``seed + repeat_idx``) so repeated
    generations of the same prompt actually diverge — the vLLM replicas run with a
    fixed engine seed, so without a per-request seed identical prompts sample
    identically (you'd see one output per replica).
    """
    items = []
    for pos, row in enumerate(df.itertuples(index=False)):
        rd = row._asdict()
        data_source = rd["data_source"]
        extra_info = rd.get("extra_info")
        ground_truth = rd["reward_model"]["ground_truth"]
        messages = _to_messages(rd["prompt"])
        base_id = f"{data_source}:{_row_index(extra_info, pos)}"
        for r in range(repeats):
            items.append({
                "id": f"{base_id}#{r}",
                "repeat_idx": r,
                "seed": seed + r,
                "data_source": data_source,
                "ability": rd.get("ability"),
                "messages": messages,
                "ground_truth": ground_truth,
                "extra_info": extra_info if isinstance(extra_info, dict) else None,
            })
    return items


def load_done_ids(results_path: str) -> set[str]:
    """Read an existing results.jsonl and return the set of completed ids."""
    done: set[str] = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # half-written trailing line from a previous interruption; ignore
                continue
            if "id" in rec:
                done.add(rec["id"])
    return done


async def writer_task(queue: asyncio.Queue, results_path: str) -> None:
    """Drain result records onto disk, durably, one JSON line each."""
    with open(results_path, "a", encoding="utf-8") as fh:
        while True:
            rec = await queue.get()
            if rec is None:  # sentinel
                queue.task_done()
                return
            fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            queue.task_done()


async def process_item(
    item: dict,
    client: AsyncOpenAI,
    model: str,
    gen_sem: asyncio.Semaphore,
    verify_sem: asyncio.Semaphore,
    verify_pool: ThreadPoolExecutor,
    queue: asyncio.Queue,
    sampling: dict,
    extra_body_base: dict,
    enable_thinking: bool,
) -> float | None:
    """Generate one response, verify it, enqueue the record. Returns the score."""
    record = {
        "id": item["id"],
        "data_source": item["data_source"],
        "ability": item["ability"],
        "repeat_idx": item["repeat_idx"],
        "seed": item["seed"],
        "prompt": item["messages"],
        "ground_truth": item["ground_truth"],
        "extra_info": item["extra_info"],
        "score": None,
        "response": None,
        "reasoning": None,
        "missing_reasoning": None,
        "finish_reason": None,
        "completion_tokens": None,
        "verify_seconds": None,
    }
    try:
        async with gen_sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=item["messages"],
                seed=item["seed"],
                extra_body=extra_body_base or None,
                **sampling,
            )

        choice = resp.choices[0]
        reasoning, answer = split_reasoning(choice.message.content)
        record["response"] = answer  # the answer (post-reasoning) is what gets scored
        record["reasoning"] = reasoning
        record["missing_reasoning"] = bool(enable_thinking and not reasoning)
        record["finish_reason"] = choice.finish_reason
        if resp.usage:
            record["completion_tokens"] = resp.usage.completion_tokens

        loop = asyncio.get_running_loop()
        async with verify_sem:
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                verify_pool, verify, item["data_source"], answer,
                item["ground_truth"], item["extra_info"],
            )
            verify_seconds = time.perf_counter() - t0
        record.update(result)  # adds score (+ error/extra keys if any)
        record["verify_seconds"] = round(verify_seconds, 4)
    except Exception as exc:  # noqa: BLE001 - record failure, keep the run alive
        record["error"] = f"{type(exc).__name__}: {exc}"

    await queue.put(record)
    return record


async def run(args, items: list[dict], results_path: str) -> None:
    """Drive all work items concurrently (generate + verify) and stream results to disk."""
    client = AsyncOpenAI(base_url=args.base_url, api_key=load_api_key(), max_retries=args.max_retries)
    gen_sem = asyncio.Semaphore(args.concurrency)
    verify_sem = asyncio.Semaphore(args.verify_concurrency)
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    verify_pool = ThreadPoolExecutor(max_workers=args.verify_concurrency)

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    extra_body_base = {
        "skip_special_tokens": args.skip_special_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }

    writer = asyncio.create_task(writer_task(queue, results_path))
    tasks = [
        asyncio.create_task(process_item(
            it, client, args.served_model_name, gen_sem, verify_sem,
            verify_pool, queue, sampling, extra_body_base, args.enable_thinking,
        ))
        for it in items
    ]

    verify_times: list[float] = []
    pbar = tqdm(total=len(tasks), desc="generate+verify", unit="sample")
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        if rec.get("verify_seconds") is not None:
            verify_times.append(rec["verify_seconds"])
        pbar.update(1)
    pbar.close()

    if verify_times:
        total = sum(verify_times)
        print(f"[verify] {len(verify_times)} samples, cumulative {total:.1f}s of scoring "
              f"(mean {total / len(verify_times):.3f}s/sample, max {max(verify_times):.3f}s; "
              f"runs concurrently, so wall-clock is lower)")

    await queue.put(None)  # tell writer to finish
    await writer
    verify_pool.shutdown(wait=True)
    print(f"[done] wrote {len(tasks)} results to {results_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate + verify responses for a verl-format parquet.")
    p.add_argument("--input", required=True, help="Input parquet (verl format).")
    p.add_argument("--served-model-name", required=True, help="Model id to query on the gateway.")
    p.add_argument("--output-dir", required=True, help="Run dir; results.jsonl written here.")
    p.add_argument("--job-id", type=int, default=None, help="Slurm job id to watch (fail fast).")
    p.add_argument("--base-url", default=GATEWAY_URL)
    p.add_argument("--concurrency", type=int, default=128, help="Max in-flight generations.")
    p.add_argument("--verify-concurrency", type=int, default=32, help="Max concurrent verifications.")
    p.add_argument("--repeats", type=int, default=1, help="Generations per prompt.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base request seed; repeat i uses seed+i so repeats diverge yet stay reproducible.")
    p.add_argument("--start", type=int, default=0, help="First row (inclusive).")
    p.add_argument("--end", type=int, default=None, help="Last row (exclusive); default = end of file.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--enable-thinking", type=_str2bool, default=False,
                   help="Global thinking flag sent as chat_template_kwargs.enable_thinking for all samples.")
    p.add_argument("--skip-special-tokens", type=_str2bool, default=True,
                   help="vLLM skip_special_tokens. false keeps markers (e.g. <|inner_prefix|>) in the response.")
    p.add_argument("--max-retries", type=int, default=6, help="OpenAI client retry budget per request.")
    p.add_argument("--job-timeout", type=int, default=1800)
    p.add_argument("--ready-timeout", type=int, default=1800)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, RESULTS_FILENAME)

    df = pd.read_parquet(args.input)
    end = args.end if args.end is not None else len(df)
    df = df.iloc[args.start:end].reset_index(drop=True)
    print(f"[input] {args.input}: rows [{args.start}:{end}] -> {len(df)} prompts")

    items = build_work_items(df, args.repeats, args.seed)
    done = load_done_ids(results_path)
    if done:
        before = len(items)
        items = [it for it in items if it["id"] not in done]
        print(f"[resume] {len(done)} results already present; "
              f"skipping {before - len(items)}, {len(items)} remaining.")
    if not items:
        print("[done] nothing to do — all requested items already present.")
        return

    # Block until the job + model are ready (sync client is fine for the probe).
    try:
        probe = OpenAI(base_url=args.base_url, api_key=load_api_key())
        if args.job_id is not None:
            wait_for_running(args.job_id, args.job_timeout)
        wait_for_model(probe, args.served_model_name, args.ready_timeout, job_id=args.job_id)
    except (RuntimeError, TimeoutError) as exc:
        sys.exit(f"[fatal] {exc}")

    asyncio.run(run(args, items, results_path))


if __name__ == "__main__":
    main()
