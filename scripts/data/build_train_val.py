#!/usr/bin/env python3
"""Merge processed SSFT parquet files into SFT train and validation sets."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def convert_row(row):
    developer = next(message for message in row["messages"] if message["role"] == "developer")
    content = developer["content"]
    tools = content.get("tools") or ""

    if content.get("has_thinking"):
        raise ValueError("thinking-enabled row found")
    if "display_answers" in tools:
        raise ValueError("display_answers tool found")

    messages = [message for message in row["messages"] if message["role"] != "developer"]
    return {
        "messages": json.dumps(messages, ensure_ascii=False),
        "tools": tools,
        "enable_thinking": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--num-proc", type=int, default=64)
    args = parser.parse_args()

    files = sorted(str(path) for path in args.input_dir.glob("*.parquet"))
    if not files:
        parser.error(f"no parquet files found in {args.input_dir}")

    dataset = load_dataset("parquet", data_files=files, split="train")
    dataset = dataset.map(
        convert_row,
        remove_columns=[column for column in dataset.column_names if column != "conversation_id"],
        num_proc=None if args.num_proc == 1 else args.num_proc,
    )
    splits = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
    splits = splits.remove_columns("conversation_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits["train"].to_parquet(args.output_dir / "train.parquet")
    splits["test"].to_parquet(args.output_dir / "val.parquet")

    print(f"train: {len(splits['train'])}")
    print(f"val: {len(splits['test'])}")


if __name__ == "__main__":
    main()
