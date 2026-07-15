#!/usr/bin/env python3
"""Build standalone reasoning or mixed SFT-0+reasoning datasets from verified generations.

A prompt counts as "solvable" by the student when at least one of its verified
attempts scored above the threshold (i.e. the prompt is inside the model's support).
Attempts that were cut off by the generation token limit (finish_reason "length")
never count as correct, regardless of their score.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


REASON_OPEN = ("<|inner_prefix|>", "<think>")
REASON_CLOSE = ("<|inner_suffix|>", "</think>")
DEFAULT_HASHES = 64
DEFAULT_BANDS = 16
MINHASH_PRIME = (1 << 61) - 1


def log(message: str) -> None:
    """Print a timestamped phase message immediately (log-file friendly)."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {message}")


def json_default(value: Any) -> Any:
    """Convert numpy values into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def as_plain(value: Any) -> Any:
    """Recursively convert numpy containers into plain Python values."""
    if isinstance(value, np.ndarray):
        return as_plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: as_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_plain(v) for v in value]
    return value


def parse_json_cell(value: Any, *, field: str) -> Any:
    """Decode a parquet cell that may already be structured or JSON text."""
    value = as_plain(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {field}: {exc}") from exc
    return value


def iter_result_files(paths: list[Path]) -> list[Path]:
    """Expand JSONL result files or directories into concrete files."""
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            found = sorted(path.rglob("results.jsonl"))
            if found:
                files.extend(found)
            else:
                files.extend(sorted(path.glob("*.jsonl")))
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no result JSONL files found")
    return files


def iter_parquet_files(paths: list[Path]) -> list[Path]:
    """Expand SFT-0 parquet files or directories, preferring train.parquet."""
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            train_file = path / "train.parquet"
            files.extend([train_file] if train_file.exists() else sorted(path.glob("*.parquet")))
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no parquet files found")
    return files


def prompt_id(record_id: str) -> str:
    """Strip the attempt suffix from a generation record id."""
    return record_id.rsplit("#", 1)[0]


def repeat_index(record: dict[str, Any]) -> int:
    """Return the numeric attempt index from a result record."""
    repeat = record.get("repeat_idx")
    if repeat is not None:
        return int(repeat)
    record_id = str(record.get("id", ""))
    if "#" in record_id:
        suffix = record_id.rsplit("#", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def first_close_delimiter(text: str) -> tuple[int, int]:
    """Return (index, length) of the earliest closing thinking delimiter, or (-1, 0)."""
    close_index, close_len = -1, 0
    for token in REASON_CLOSE:
        index = text.find(token)
        if index != -1 and (close_index == -1 or index < close_index):
            close_index, close_len = index, len(token)
    return close_index, close_len


def split_reasoning(text: str | None) -> tuple[str | None, str]:
    """Split rendered thinking delimiters into reasoning and final response.

    Also handles malformed generations that close a thinking block without ever
    opening one (chat templates that pre-open thinking in the generation prompt
    make the model emit only the closing delimiter): everything before the lone
    closing delimiter is treated as reasoning.
    """
    content = text or ""
    open_index, open_token = -1, ""
    for token in REASON_OPEN:
        index = content.find(token)
        if index != -1 and (open_index == -1 or index < open_index):
            open_index, open_token = index, token
    if open_index == -1:
        close_index, close_len = first_close_delimiter(content)
        if close_index == -1:
            return None, content
        return content[:close_index], content[close_index + close_len :]

    rest = content[open_index + len(open_token) :]
    close_index, close_len = first_close_delimiter(rest)
    if close_index == -1:
        return rest, ""
    return rest[:close_index], rest[close_index + close_len :]


def is_truncated(record: dict[str, Any]) -> bool:
    """True when the generation hit the token limit instead of concluding."""
    return record.get("finish_reason") == "length"


def normalized_generation(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one generation result record for selection and row building."""
    record = as_plain(record)
    reasoning = record.get("reasoning")
    response = record.get("response")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    if reasoning in (None, "") and isinstance(response, str):
        parsed_reasoning, parsed_response = split_reasoning(response)
        if parsed_reasoning is not None:
            reasoning = parsed_reasoning.strip()
            response = parsed_response
    if isinstance(response, str):
        response = response.strip()
    return {
        **record,
        "reasoning": reasoning if reasoning not in ("", None) else None,
        "response": response if response is not None else "",
        "repeat_idx": repeat_index(record),
    }


def load_best_generations(
    paths: list[Path],
    *,
    max_attempts: int,
    score_threshold: float,
    source_name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, int]]:
    """Stream results, enforce attempt limits, and keep best positive rows."""
    best: dict[str, dict[str, Any]] = {}
    attempt_masks: dict[str, int] = defaultdict(int)
    pass_masks: dict[str, int] = defaultdict(int)
    truncated_positive = 0
    for path in iter_result_files(paths):
        file_size = path.stat().st_size
        log(f"[{source_name}] reading {path} ({file_size / 2**30:.2f} GiB)")
        with path.open(encoding="utf-8") as handle:
            progress = tqdm(
                total=file_size,
                desc=f"[{source_name}] {path.parent.name}/{path.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            )
            for line_number, line in enumerate(handle, start=1):
                if line_number % 1024 == 0:
                    progress.update(handle.buffer.tell() - progress.n)
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if "id" not in raw_record:
                    raise ValueError(f"{path}:{line_number}: missing id")
                record = normalized_generation(raw_record)
                pid = prompt_id(str(record["id"]))
                repeat = int(record["repeat_idx"])
                attempt_masks[pid] |= 1 << repeat
                attempt_count = attempt_masks[pid].bit_count()
                if attempt_count > max_attempts:
                    raise ValueError(
                        f"{source_name} prompt {pid!r} has {attempt_count} attempts; "
                        f"expected at most {max_attempts}"
                    )

                score = record.get("score")
                if score is None or float(score) <= score_threshold or not (record.get("response") or "").strip():
                    continue
                if is_truncated(record):
                    truncated_positive += 1
                    continue
                pass_masks[pid] |= 1 << repeat
                current = best.get(pid)
                if current is None:
                    best[pid] = record
                    continue
                current_key = (
                    float(current["score"]),
                    1 if current.get("reasoning") else 0,
                    -int(current.get("repeat_idx") or 0),
                )
                record_key = (
                    float(record["score"]),
                    1 if record.get("reasoning") else 0,
                    -repeat,
                )
                if record_key > current_key:
                    best[pid] = record
            progress.update(file_size - progress.n)
            progress.close()

    log(
        f"[{source_name}] {len(best):,} prompts with a positive answer "
        f"out of {len(attempt_masks):,} prompts seen "
        f"({truncated_positive:,} positively-scored but truncated attempts discarded)"
    )
    attempt_counts = {pid: mask.bit_count() for pid, mask in attempt_masks.items()}
    pass_counts = {pid: mask.bit_count() for pid, mask in pass_masks.items()}
    return best, attempt_counts, pass_counts


def convert_prompt(prompt: Any) -> list[dict[str, Any]]:
    """Convert a plain generation prompt into Apertus structured messages."""
    prompt = parse_json_cell(prompt, field="prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty message list")

    messages: list[dict[str, Any]] = []
    for message in prompt:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, dict):
            content_text = content_text_for_prompt(content)
        else:
            content_text = "" if content is None else str(content)

        if role == "system":
            messages.append({"role": "system", "content": {"text": content_text}})
        elif role == "user":
            messages.append(
                {
                    "role": "user",
                    "content": {"parts": [{"type": "text", "text": content_text}]},
                }
            )
        else:
            raise ValueError(f"unsupported prompt role {role!r}")
    return messages


def make_sft_row(
    record: dict[str, Any],
    *,
    include_reasoning: bool,
    enable_thinking: bool | None = None,
    source: str,
    student_attempt_count: int | None = None,
    student_pass_count: int | None = None,
) -> dict[str, Any]:
    """Create one verl_sft-ready row from a selected generation result."""
    messages = convert_prompt(record["prompt"])
    reasoning = record.get("reasoning")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
    has_reasoning = bool(include_reasoning and reasoning)
    blocks: list[dict[str, Any]] = []
    if has_reasoning:
        blocks.append({"type": "thoughts", "text": reasoning})
    blocks.append({"type": "response", "text": str(record.get("response") or "").strip()})
    messages.append({"role": "assistant", "content": {"blocks": blocks}})
    pid = prompt_id(str(record["id"]))
    data_source = record.get("data_source") or (pid.rsplit(":", 1)[0] if ":" in pid else None)
    metadata = {
        "origin": source,
        "prompt_id": pid,
        "data_source": data_source,
        "student_attempt_count": student_attempt_count,
        "student_pass_count": student_pass_count,
    }
    return {
        "messages": json.dumps(messages, ensure_ascii=False, default=json_default),
        "tools": "",
        "enable_thinking": bool(has_reasoning or enable_thinking),
        "metadata": json.dumps(metadata, ensure_ascii=False, default=json_default),
    }


def no_cot_enable_thinking(*, strategy: str, rng: random.Random) -> bool:
    """Choose enable_thinking for a row that has no thoughts block."""
    if strategy == "none":
        return False
    if strategy == "random":
        return rng.random() < 0.5
    raise ValueError(f"unsupported no-CoT enable-thinking strategy {strategy!r}")


def build_reasoning_rows(
    *,
    strategy: str,
    teacher_best: dict[str, dict[str, Any]],
    student_best: dict[str, dict[str, Any]],
    student_attempt_counts: dict[str, int],
    student_pass_counts: dict[str, int],
    score_threshold: float,
    seed: int,
    no_cot_enable_thinking_strategy: str,
) -> list[dict[str, Any]]:
    """Build reasoning rows according to the selected data strategy."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for pid in tqdm(sorted(teacher_best), desc=f"[rows] {strategy}", unit="prompt", dynamic_ncols=True):
        teacher = teacher_best[pid]
        student = student_best.get(pid)
        attempt_count = student_attempt_counts.get(pid)
        pass_count = student_pass_counts.get(pid, 0) if attempt_count is not None else None
        solvable_by_student = (
            student is not None
            and student.get("score") is not None
            and float(student["score"]) > score_threshold
            and (student.get("response") or "").strip()
            and not is_truncated(student)
        )

        if strategy == "solvable-student-else-teacher":
            if solvable_by_student:
                rows.append(
                    make_sft_row(
                        student,
                        include_reasoning=False,
                        enable_thinking=no_cot_enable_thinking(
                            strategy=no_cot_enable_thinking_strategy,
                            rng=rng,
                        ),
                        source="student",
                        student_attempt_count=attempt_count,
                        student_pass_count=pass_count,
                    )
                )
            else:
                rows.append(
                    make_sft_row(
                        teacher,
                        include_reasoning=True,
                        source="teacher",
                        student_attempt_count=attempt_count,
                        student_pass_count=pass_count,
                    )
                )
        elif strategy == "unsolvable-teacher-only":
            if not student_attempt_counts:
                raise ValueError("--student-results is required for unsolvable-teacher-only")
            if not solvable_by_student:
                rows.append(
                    make_sft_row(
                        teacher,
                        include_reasoning=True,
                        source="teacher",
                        student_attempt_count=attempt_count,
                        student_pass_count=pass_count,
                    )
                )
        elif strategy == "teacher-only-5050":
            include_reasoning = rng.random() < 0.5
            rows.append(
                make_sft_row(
                    teacher,
                    include_reasoning=include_reasoning,
                    enable_thinking=None
                    if include_reasoning
                    else no_cot_enable_thinking(
                        strategy=no_cot_enable_thinking_strategy,
                        rng=rng,
                    ),
                    source="teacher",
                    student_attempt_count=attempt_count,
                    student_pass_count=pass_count,
                )
            )
        else:
            raise ValueError(f"unsupported strategy {strategy!r}")
    return rows


def content_text_for_prompt(content: Any) -> str:
    """Extract comparable text from Apertus-style message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, dict):
        return "" if content is None else str(content)
    text = content.get("text")
    if isinstance(text, str) and text:
        return text
    if isinstance(content.get("parts"), list):
        return "".join(
            str(part.get("text", ""))
            for part in content["parts"]
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    if isinstance(content.get("blocks"), list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content["blocks"]
            if isinstance(block, dict) and "text" in block
        )
    if isinstance(text, str):
        return text
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=json_default)


def prompt_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """Render only the prompt side of a conversation for deduplication."""
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "developer":
            continue
        if role == "assistant":
            break
        parts.append(f"{role}: {content_text_for_prompt(message.get('content'))}")
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    """Normalize prompt text before shingling or exact hashing."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def shingle_text(text: str, width: int = 5) -> frozenset[int]:
    """Create hashed word shingles for approximate prompt matching."""
    tokens = re.findall(r"\w+", normalize_text(text))
    if len(tokens) < width:
        return frozenset(stable_hash(token) for token in tokens)
    return frozenset(
        stable_hash(" ".join(tokens[i : i + width]))
        for i in range(len(tokens) - width + 1)
    )


def stable_hash(text: str) -> int:
    """Return a deterministic 64-bit hash for LSH keys."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


@lru_cache(maxsize=None)
def hash_coefficients(num_hashes: int) -> tuple[tuple[int, int], ...]:
    """Return deterministic universal-hash coefficients for MinHash slots."""
    return tuple(
        (
            stable_hash(f"minhash-a-{index}") % (MINHASH_PRIME - 1) + 1,
            stable_hash(f"minhash-b-{index}") % MINHASH_PRIME,
        )
        for index in range(num_hashes)
    )


def minhash(shingles: frozenset[int], num_hashes: int) -> tuple[int, ...]:
    """Compute a simple MinHash signature for a shingle set."""
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    signature = [MINHASH_PRIME] * num_hashes
    for shingle in shingles:
        value = shingle % MINHASH_PRIME
        for index, (multiplier, offset) in enumerate(hash_coefficients(num_hashes)):
            candidate = (multiplier * value + offset) % MINHASH_PRIME
            if candidate < signature[index]:
                signature[index] = candidate
    return tuple(signature)


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    """Compute Jaccard similarity between two shingle sets."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class PromptLSH:
    """MinHash LSH index for finding SFT-0 prompts overlapping reasoning rows."""

    def __init__(self, prompts: Iterable[str], *, num_hashes: int, bands: int):
        """Build an LSH index over prompt strings."""
        if num_hashes % bands != 0:
            raise ValueError("--lsh-num-hashes must be divisible by --lsh-bands")
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        self.prompts: dict[int, frozenset[int]] = {}
        self.index: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        prompts = list(prompts)
        for prompt in tqdm(prompts, desc="[dedup] building LSH index", unit="prompt", dynamic_ncols=True):
            shingles = shingle_text(prompt)
            key = stable_hash(normalize_text(prompt))
            self.prompts[key] = shingles
            signature = minhash(shingles, num_hashes)
            for band in range(bands):
                start = band * self.rows_per_band
                band_key = (band, signature[start : start + self.rows_per_band])
                self.index[band_key].append(key)

    def has_near_duplicate(self, prompt: str, *, threshold: float) -> bool:
        """Return whether a prompt matches an indexed prompt above threshold."""
        normalized = normalize_text(prompt)
        key = stable_hash(normalized)
        shingles = shingle_text(prompt)
        if key in self.prompts:
            return True
        signature = minhash(shingles, self.num_hashes)
        candidates: set[int] = set()
        for band in range(self.bands):
            start = band * self.rows_per_band
            band_key = (band, signature[start : start + self.rows_per_band])
            candidates.update(self.index.get(band_key, ()))
        return any(jaccard(shingles, self.prompts[candidate]) >= threshold for candidate in candidates)


def normalize_sft0_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one existing SFT-0 row to the final training schema."""
    messages = parse_json_cell(row.get("messages"), field="messages")
    if not isinstance(messages, list):
        raise ValueError("SFT-0 messages must be a list or JSON list")

    tools = row.get("tools", "")
    enable_thinking = bool(row.get("enable_thinking", False))
    cleaned_messages = []
    for message in messages:
        if message.get("role") == "developer":
            content = message.get("content") or {}
            if isinstance(content, dict):
                tools = tools or content.get("tools") or ""
                enable_thinking = enable_thinking or bool(content.get("has_thinking"))
            continue
        cleaned_messages.append(as_plain(message))
    conversation_id = as_plain(row.get("conversation_id"))
    metadata = {
        "origin": "sft0",
        "prompt_id": None if conversation_id is None else str(conversation_id),
        "data_source": None,
        "student_attempt_count": None,
        "student_pass_count": None,
    }
    return {
        "messages": json.dumps(cleaned_messages, ensure_ascii=False, default=json_default),
        "tools": "" if tools is None else str(tools),
        "enable_thinking": enable_thinking,
        "metadata": json.dumps(metadata, ensure_ascii=False, default=json_default),
    }


def load_sft0_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and normalize SFT-0 parquet rows."""
    rows: list[dict[str, Any]] = []
    for path in iter_parquet_files(paths):
        parquet_file = pq.ParquetFile(path)
        total = parquet_file.metadata.num_rows
        log(f"[sft0] reading {path} ({total:,} rows)")
        progress = tqdm(total=total, desc=f"[sft0] {path.name}", unit="row", dynamic_ncols=True)
        index = 0
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                try:
                    rows.append(normalize_sft0_row(row))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"{path}, row {index}: {exc}") from exc
                index += 1
            progress.update(batch.num_rows)
        progress.close()
    return rows


def dedup_sft0_rows(
    sft0_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
    *,
    threshold: float,
    num_hashes: int,
    bands: int,
) -> tuple[list[dict[str, Any]], int]:
    """Remove SFT-0 rows whose prompts overlap reasoning rows."""
    reasoning_prompts = [
        prompt_text_from_messages(parse_json_cell(row["messages"], field="messages"))
        for row in reasoning_rows
    ]
    lsh = PromptLSH(reasoning_prompts, num_hashes=num_hashes, bands=bands)
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in tqdm(sft0_rows, desc="[dedup] scanning SFT-0 rows", unit="row", dynamic_ncols=True):
        messages = parse_json_cell(row["messages"], field="messages")
        prompt = prompt_text_from_messages(messages)
        if lsh.has_near_duplicate(prompt, threshold=threshold):
            removed += 1
        else:
            kept.append(row)
    log(f"[dedup] removed {removed:,} of {len(sft0_rows):,} SFT-0 rows overlapping reasoning prompts")
    return kept, removed


def write_train_dataset(rows: list[dict[str, Any]], output_dir: Path, *, seed: int) -> None:
    """Shuffle rows and write the final train.parquet dataset."""
    if output_dir.exists() and any(output_dir.glob("*.parquet")):
        raise FileExistsError(f"refusing to overwrite parquet files in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[write] shuffling and writing {len(rows):,} rows to {output_dir / 'train.parquet'}")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train_rows = [
        {
            "messages": row["messages"],
            "tools": row.get("tools", ""),
            "enable_thinking": bool(row.get("enable_thinking", False)),
            "metadata": row.get("metadata", ""),
        }
        for row in shuffled
    ]

    schema = pa.schema(
        [
            pa.field("messages", pa.string()),
            pa.field("tools", pa.string()),
            pa.field("enable_thinking", pa.bool_()),
            pa.field("metadata", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(train_rows, schema=schema), output_dir / "train.parquet", compression="zstd")
    print(f"train: {len(train_rows)}")


def write_manifest(output_dir: Path, stats: dict[str, Any]) -> None:
    """Write a JSON manifest with dataset construction statistics."""
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True, ensure_ascii=False, default=json_default)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-results", type=Path, action="append", required=True)
    parser.add_argument("--student-results", type=Path, action="append", default=[])
    parser.add_argument("--sft0", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("solvable-student-else-teacher", "unsolvable-teacher-only", "teacher-only-5050"),
        default="solvable-student-else-teacher",
        help=(
            "solvable-student-else-teacher: student's own answer where the student solved the "
            "prompt, teacher CoT+answer otherwise. unsolvable-teacher-only: teacher CoT rows "
            "only for prompts the student never solved. teacher-only-5050: all-teacher "
            "rows, 50/50 coin flip on keeping the CoT."
        ),
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("standalone", "mixed"),
        default="standalone",
        help="standalone: reasoning rows only. mixed: merged with deduplicated SFT-0 rows.",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument(
        "--no-cot-enable-thinking-strategy",
        choices=("none", "random"),
        default="random",
        help=(
            "How to set enable_thinking for rows without a thoughts block "
            "(the solvable/student rows under solvable-student-else-teacher, the no-CoT "
            "coin-flip rows under teacher-only-5050): 'random' flips a 50/50 coin, "
            "'none' never enables thinking. Rows with thoughts always force "
            "enable_thinking=true."
        ),
    )
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--lsh-num-hashes", type=int, default=DEFAULT_HASHES)
    parser.add_argument("--lsh-bands", type=int, default=DEFAULT_BANDS)
    args = parser.parse_args()
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if args.dataset_mode == "mixed" and not args.sft0:
        parser.error("--sft0 is required when --dataset-mode=mixed")
    if args.strategy in {"solvable-student-else-teacher", "unsolvable-teacher-only"} and not args.student_results:
        parser.error(f"--student-results is required for --strategy={args.strategy}")
    return args


def main() -> None:
    """Build the requested train dataset and manifest."""
    args = parse_args()
    teacher_best, teacher_attempt_counts, _teacher_pass_counts = load_best_generations(
        args.teacher_results,
        max_attempts=args.max_attempts,
        score_threshold=args.score_threshold,
        source_name="teacher",
    )
    student_best, student_attempt_counts, student_pass_counts = (
        load_best_generations(
            args.student_results,
            max_attempts=args.max_attempts,
            score_threshold=args.score_threshold,
            source_name="student",
        )
        if args.student_results
        else ({}, {}, {})
    )

    reasoning_rows = build_reasoning_rows(
        strategy=args.strategy,
        teacher_best=teacher_best,
        student_best=student_best,
        student_attempt_counts=student_attempt_counts,
        student_pass_counts=student_pass_counts,
        score_threshold=args.score_threshold,
        seed=args.seed,
        no_cot_enable_thinking_strategy=args.no_cot_enable_thinking_strategy,
    )
    all_rows = reasoning_rows
    dedup_removed = 0
    sft0_count = 0
    if args.dataset_mode == "mixed":
        sft0_rows = load_sft0_rows(args.sft0)
        sft0_count = len(sft0_rows)
        kept_sft0, dedup_removed = dedup_sft0_rows(
            sft0_rows,
            reasoning_rows,
            threshold=args.dedup_threshold,
            num_hashes=args.lsh_num_hashes,
            bands=args.lsh_bands,
        )
        all_rows = kept_sft0 + reasoning_rows

    log("computing dataset statistics")
    thought_block_count = sum(
        any(
            block.get("type") == "thoughts"
            for block in (
                parse_json_cell(row["messages"], field="messages")[-1]
                .get("content", {})
                .get("blocks", [])
            )
        )
        for row in reasoning_rows
    )
    stats = {
        "dataset_mode": args.dataset_mode,
        "strategy": args.strategy,
        "score_threshold": args.score_threshold,
        "no_cot_enable_thinking_strategy": args.no_cot_enable_thinking_strategy,
        "teacher_prompts_with_positive_score": len(teacher_best),
        "student_prompt_count": len(student_attempt_counts),
        "reasoning_rows": len(reasoning_rows),
        "reasoning_rows_with_thought_blocks": thought_block_count,
        "reasoning_rows_with_thinking_enabled": sum(bool(row.get("enable_thinking")) for row in reasoning_rows),
        "sft0_rows_loaded": sft0_count,
        "sft0_rows_removed_by_prompt_dedup": dedup_removed,
        "final_rows": len(all_rows),
        "teacher_attempt_count_max": max(teacher_attempt_counts.values(), default=0),
        "student_attempt_count_max": max(student_attempt_counts.values(), default=0),
    }

    write_train_dataset(all_rows, args.output_dir, seed=args.seed)
    write_manifest(args.output_dir, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
