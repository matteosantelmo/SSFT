#!/usr/bin/env python3
"""Convert verl RL evaluation parquet files to SFT train-time rollout format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


TASKS = {
    "openai/gsm8k": "gsm8k",
    "HuggingFaceH4/MATH-500": "math500",
    "aime2024": "aime2024",
    "aime2025": "aime2025",
    "gpqa_diamond": "gpqa_diamond",
    "mmlu": "mmlu",
    "google/IFEval": "ifeval",
    "allenai/IFBench_test": "ifbench",
    "humaneval": "humaneval",
}


def parse_json(value: Any, *, field: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {field}: {exc}") from exc


def convert_prompt(prompt: Any) -> list[dict[str, Any]]:
    """Change only the message container; preserve every content string exactly."""
    prompt = parse_json(prompt, field="prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty list of messages")

    messages = []
    for message in prompt:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{role!r} message content must be a string")

        if role == "system":
            messages.append({"role": "system", "content": {"text": content}})
        elif role == "user":
            messages.append(
                {
                    "role": "user",
                    "content": {"parts": [{"type": "text", "text": content}]},
                }
            )
        else:
            raise ValueError(
                f"unsupported prompt role {role!r}; evaluation prompts must contain only system/user messages"
            )
    return messages


def convert_row(row: dict[str, Any], *, max_new_tokens: int, code_samples: int) -> dict[str, Any]:
    data_source = row.get("data_source")
    if data_source not in TASKS:
        raise ValueError(f"unsupported data_source {data_source!r}")

    reward_model = parse_json(row.get("reward_model"), field="reward_model")
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise ValueError("reward_model must contain ground_truth")

    task_name = TASKS[data_source]
    extra_info = parse_json(row.get("extra_info"), field="extra_info")
    if extra_info is not None and not isinstance(extra_info, dict):
        raise ValueError("extra_info must be a JSON object or null")

    sampling_params: dict[str, Any] = {
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "skip_special_tokens": False,
    }
    if data_source == "humaneval":
        sampling_params.update({"temperature": 0.8, "top_p": 0.95, "n": code_samples})

    rollout_params = {
        "data_source": data_source,
        "task_name": task_name,
        "ground_truth": reward_model["ground_truth"],
        "extra_info": extra_info,
        "sampling_params": sampling_params,
    }
    chat_template_kwargs = (extra_info or {}).get("apply_chat_template_kwargs") or {}
    enable_thinking = bool(chat_template_kwargs.get("enable_thinking", False))
    return {
        "messages": json.dumps(
            convert_prompt(row.get("prompt")),
            ensure_ascii=False,
        ),
        "tools": "",
        "enable_thinking": enable_thinking,
        "rollout_params": json.dumps(rollout_params, ensure_ascii=False),
    }


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
        raise ValueError(f"no parquet files found in {path}")
    raise ValueError(f"input path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RL eval parquet file or directory")
    parser.add_argument("output", type=Path, help="Output rollout.parquet path")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--code-samples", type=int, default=10)
    args = parser.parse_args()

    rows = []
    counts: dict[str, int] = {}
    for path in input_files(args.input):
        table = pq.read_table(path)
        for index, row in enumerate(table.to_pylist()):
            try:
                rows.append(
                    convert_row(
                        row,
                        max_new_tokens=args.max_new_tokens,
                        code_samples=args.code_samples,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}, row {index}: {exc}") from exc
        counts[path.name] = table.num_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")

    for name, count in counts.items():
        print(f"{name}: {count}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
