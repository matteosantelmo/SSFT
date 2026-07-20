#!/usr/bin/env python3
"""Build standalone reasoning or mixed SFT-0+reasoning datasets from verified generations.

A prompt counts as "solvable" by the student when at least one of its verified
attempts scored above the threshold (i.e. the prompt is inside the model's support).
Attempts that were cut off by the generation token limit (finish_reason "length")
never count as correct, regardless of their score.

One invocation can write several datasets via repeatable --build NAME:STRATEGY:MODE
specs: the heavy inputs (teacher/student results, SFT-0) are loaded once and shared
across builds, with outputs identical to separate single-build runs.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


REASON_OPEN = ("<|inner_prefix|>", "<think>")
REASON_CLOSE = ("<|inner_suffix|>", "</think>")
THINKING_MARKERS = ("<|inner_prefix|>", "<|inner_suffix|>", "<|channel>thought", "<think>", "</think>")
DEFAULT_HASHES = 64
DEFAULT_BANDS = 16
MINHASH_PRIME = (1 << 61) - 1
STRATEGIES = ("solvable-student-else-teacher", "unsolvable-teacher-only", "teacher-only")
STUDENT_STRATEGIES = ("solvable-student-else-teacher", "unsolvable-teacher-only")
DATASET_MODES = ("standalone", "mixed")


@dataclass(frozen=True)
class BuildSpec:
    """One requested output dataset: name, destination, strategy, and mode."""

    name: str
    output_dir: Path
    strategy: str
    dataset_mode: str


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
        "sample_id": str(record["id"]),
        "prompt_id": pid,
        "data_source": data_source,
        "model": record.get("teacher_model") or record.get("model"),
        "repeat_idx": record.get("repeat_idx"),
        "score": record.get("score"),
        "completion_tokens": record.get("completion_tokens"),
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


def is_solvable(student: dict[str, Any] | None, score_threshold: float) -> bool:
    """True when the student's best attempt solves the prompt (complete, above threshold)."""
    return bool(
        student is not None
        and student.get("score") is not None
        and float(student["score"]) > score_threshold
        and (student.get("response") or "").strip()
        and not is_truncated(student)
    )


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
    solvable_cot_alpha: float = 0.0,
    teacher_cot_probability: float | None = None,
) -> list[dict[str, Any]]:
    """Build reasoning rows according to the selected data strategy."""
    if strategy == "unsolvable-teacher-only" and solvable_cot_alpha:
        raise ValueError("solvable_cot_alpha is incompatible with unsolvable-teacher-only")
    if strategy == "teacher-only" and teacher_cot_probability is None:
        raise ValueError("teacher-only requires a resolved teacher_cot_probability")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for pid in tqdm(sorted(teacher_best), desc=f"[rows] {strategy}", unit="prompt", dynamic_ncols=True):
        teacher = teacher_best[pid]
        student = student_best.get(pid)
        attempt_count = student_attempt_counts.get(pid)
        pass_count = student_pass_counts.get(pid, 0) if attempt_count is not None else None
        solvable_by_student = is_solvable(student, score_threshold)

        if strategy == "solvable-student-else-teacher":
            use_student = solvable_by_student and not (
                solvable_cot_alpha and rng.random() < solvable_cot_alpha
            )
            if use_student:
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
        elif strategy == "teacher-only":
            include_reasoning = rng.random() < teacher_cot_probability
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
    """MinHash LSH index mapping prompts to the (prompt_id, origin) tags using them."""

    def __init__(self, *, num_hashes: int, bands: int):
        """Create an empty LSH index."""
        if num_hashes % bands != 0:
            raise ValueError("--lsh-num-hashes must be divisible by --lsh-bands")
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        self.shingles: dict[int, frozenset[int]] = {}
        self.tags: dict[int, set[tuple[str, str]]] = defaultdict(set)
        self.index: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    def add(self, prompt: str, tag: tuple[str, str]) -> None:
        """Index a prompt under a (prompt_id, origin) tag."""
        key = stable_hash(normalize_text(prompt))
        if key not in self.shingles:
            shingles = shingle_text(prompt)
            self.shingles[key] = shingles
            signature = minhash(shingles, self.num_hashes)
            for band in range(self.bands):
                start = band * self.rows_per_band
                band_key = (band, signature[start : start + self.rows_per_band])
                self.index[band_key].append(key)
        self.tags[key].add(tag)

    def matching_tags(self, prompt: str, *, threshold: float) -> set[tuple[str, str]]:
        """Return the tags of all indexed prompts matching this prompt above threshold."""
        key = stable_hash(normalize_text(prompt))
        shingles = shingle_text(prompt)
        matched: set[int] = set()
        if key in self.shingles:
            matched.add(key)
        signature = minhash(shingles, self.num_hashes)
        candidates: set[int] = set()
        for band in range(self.bands):
            start = band * self.rows_per_band
            band_key = (band, signature[start : start + self.rows_per_band])
            candidates.update(self.index.get(band_key, ()))
        for candidate in candidates:
            if candidate not in matched and jaccard(shingles, self.shingles[candidate]) >= threshold:
                matched.add(candidate)
        tags: set[tuple[str, str]] = set()
        for matched_key in matched:
            tags.update(self.tags[matched_key])
        return tags


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
        "sample_id": None if conversation_id is None else str(conversation_id),
        "prompt_id": None if conversation_id is None else str(conversation_id),
        "data_source": None,
        "model": None,
        "repeat_idx": None,
        "score": None,
        "completion_tokens": None,
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


def used_prompt_tags(reasoning_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Collect the (prompt_id, origin) pairs whose prompts appear in built rows."""
    tags: set[tuple[str, str]] = set()
    for row in reasoning_rows:
        metadata = json.loads(row["metadata"])
        tags.add((metadata["prompt_id"], metadata["origin"]))
    return tags


def build_prompt_index(
    tags: set[tuple[str, str]],
    teacher_best: dict[str, dict[str, Any]],
    student_best: dict[str, dict[str, Any]],
    *,
    num_hashes: int,
    bands: int,
) -> PromptLSH:
    """Index the prompt text behind every (prompt_id, origin) pair used by any build."""
    lsh = PromptLSH(num_hashes=num_hashes, bands=bands)
    for pid, origin in tqdm(sorted(tags), desc="[dedup] building LSH index", unit="prompt", dynamic_ncols=True):
        record = teacher_best[pid] if origin == "teacher" else student_best[pid]
        lsh.add(prompt_text_from_messages(convert_prompt(record["prompt"])), (pid, origin))
    return lsh


def match_sft0_rows(
    sft0_rows: list[dict[str, Any]],
    lsh: PromptLSH,
    *,
    threshold: float,
) -> dict[int, set[tuple[str, str]]]:
    """Map SFT-0 row indices to the (prompt_id, origin) tags they near-duplicate.

    Runs once per invocation; each mixed build then filters rows by intersecting
    these tags with the tags its own reasoning rows actually use, which is exactly
    equivalent to deduplicating against that build's rows alone.
    """
    matches: dict[int, set[tuple[str, str]]] = {}
    for index, row in enumerate(tqdm(sft0_rows, desc="[dedup] scanning SFT-0 rows", unit="row", dynamic_ncols=True)):
        prompt = prompt_text_from_messages(parse_json_cell(row["messages"], field="messages"))
        tags = lsh.matching_tags(prompt, threshold=threshold)
        if tags:
            matches[index] = tags
    return matches


def filter_sft0_rows(
    sft0_rows: list[dict[str, Any]],
    matches: dict[int, set[tuple[str, str]]],
    used_tags: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep SFT-0 rows that don't near-duplicate any prompt used by this build."""
    kept: list[dict[str, Any]] = []
    removed = 0
    for index, row in enumerate(sft0_rows):
        tags = matches.get(index)
        if tags and tags & used_tags:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def filter_invalid_thinking_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove rows with thinking markers while thinking is disabled."""
    return [
        row
        for row in rows
        if row.get("enable_thinking")
        or not any(marker in row["messages"] for marker in THINKING_MARKERS)
    ]


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
    parser.add_argument("--output-dir", type=Path, help="output directory for a single build")
    parser.add_argument(
        "--build",
        action="append",
        default=[],
        metavar="NAME:STRATEGY:MODE",
        help=(
            "Repeatable build spec for multi-build mode. NAME is the output directory "
            "(joined to --output-base when given), STRATEGY one of "
            f"{', '.join(STRATEGIES)}, MODE one of {', '.join(DATASET_MODES)}. "
            "All builds share one load of the teacher/student/SFT-0 inputs and produce "
            "outputs identical to separate single-build runs."
        ),
    )
    parser.add_argument("--output-base", type=Path, help="base directory prepended to --build names")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip builds whose output dir already holds parquet files instead of failing",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=None,
        help=(
            "Single-build only (default solvable-student-else-teacher). "
            "solvable-student-else-teacher: student's own answer where the student solved the "
            "prompt, teacher CoT+answer otherwise. unsolvable-teacher-only: teacher CoT rows "
            "only for prompts the student never solved. teacher-only: all-teacher rows, CoT "
            "kept with a probability matched to the cap-filter CoT rate (or "
            "--teacher-cot-probability)."
        ),
    )
    parser.add_argument(
        "--dataset-mode",
        choices=DATASET_MODES,
        default=None,
        help=(
            "Single-build only (default standalone). standalone: reasoning rows only. "
            "mixed: merged with deduplicated SFT-0 rows."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument(
        "--no-cot-enable-thinking-strategy",
        choices=("none", "random"),
        default="random",
        help=(
            "How to set enable_thinking for rows without a thoughts block "
            "(the solvable/student rows under solvable-student-else-teacher, the no-CoT "
            "rows under teacher-only): 'random' flips a 50/50 coin, "
            "'none' never enables thinking. Rows with thoughts always force "
            "enable_thinking=true."
        ),
    )
    parser.add_argument(
        "--solvable-cot-alpha",
        type=float,
        default=0.0,
        help=(
            "Probability that a student-solvable prompt still gets the teacher CoT row: "
            "under solvable-student-else-teacher each solvable prompt keeps the student's "
            "own answer with probability 1-alpha; under teacher-only the auto-matched CoT "
            "probability grows accordingly. Incompatible with unsolvable-teacher-only."
        ),
    )
    parser.add_argument(
        "--teacher-cot-probability",
        type=float,
        default=None,
        help=(
            "Explicit probability that a teacher-only row keeps its CoT. Default: matched "
            "to the cap-filter CoT rate, (unsolvable + alpha*solvable) / prompts, which "
            "requires --student-results."
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
    if not 0 <= args.solvable_cot_alpha <= 1:
        parser.error("--solvable-cot-alpha must be between 0 and 1")
    if args.teacher_cot_probability is not None and not 0 <= args.teacher_cot_probability <= 1:
        parser.error("--teacher-cot-probability must be between 0 and 1")
    if args.build and args.output_dir:
        parser.error("--build and --output-dir are mutually exclusive")
    if not args.build and not args.output_dir:
        parser.error("either --output-dir or at least one --build is required")
    if args.build and (args.strategy or args.dataset_mode):
        parser.error("--strategy/--dataset-mode apply to single builds; encode them in each --build spec")
    if args.output_base and not args.build:
        parser.error("--output-base requires --build")

    if args.build:
        builds = []
        for spec in args.build:
            parts = spec.rsplit(":", 2)
            if len(parts) != 3 or not all(parts):
                parser.error(f"invalid --build spec {spec!r}; expected NAME:STRATEGY:MODE")
            name, strategy, mode = parts
            if strategy not in STRATEGIES:
                parser.error(f"--build {spec!r}: unknown strategy {strategy!r}")
            if mode not in DATASET_MODES:
                parser.error(f"--build {spec!r}: unknown dataset mode {mode!r}")
            output_dir = args.output_base / name if args.output_base else Path(name)
            builds.append(BuildSpec(name=name, output_dir=output_dir, strategy=strategy, dataset_mode=mode))
    else:
        builds = [
            BuildSpec(
                name=args.output_dir.name,
                output_dir=args.output_dir,
                strategy=args.strategy or "solvable-student-else-teacher",
                dataset_mode=args.dataset_mode or "standalone",
            )
        ]
    output_dirs = [build.output_dir for build in builds]
    if len(set(output_dirs)) != len(output_dirs):
        parser.error("--build specs must not share an output directory")
    if any(build.dataset_mode == "mixed" for build in builds) and not args.sft0:
        parser.error("--sft0 is required when any build uses dataset mode 'mixed'")
    student_builds = [build for build in builds if build.strategy in STUDENT_STRATEGIES]
    if student_builds and not args.student_results:
        parser.error(f"--student-results is required for --strategy={student_builds[0].strategy}")
    if args.solvable_cot_alpha and any(build.strategy == "unsolvable-teacher-only" for build in builds):
        parser.error("--solvable-cot-alpha is incompatible with unsolvable-teacher-only builds")
    teacher_only_builds = [build for build in builds if build.strategy == "teacher-only"]
    if teacher_only_builds and args.teacher_cot_probability is None and not args.student_results:
        parser.error(
            "teacher-only builds need --student-results (to match the cap-filter CoT rate) "
            "or an explicit --teacher-cot-probability"
        )
    args.builds = builds
    return args


def thought_block_count(reasoning_rows: list[dict[str, Any]]) -> int:
    """Count reasoning rows whose assistant message carries a thoughts block."""
    return sum(
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


def main() -> None:
    """Build the requested train datasets and manifests."""
    args = parse_args()
    builds: list[BuildSpec] = args.builds

    conflicts = [
        build
        for build in builds
        if build.output_dir.exists() and any(build.output_dir.glob("*.parquet"))
    ]
    if conflicts and not args.skip_existing:
        raise FileExistsError(
            "refusing to overwrite parquet files in: "
            + ", ".join(str(build.output_dir) for build in conflicts)
        )
    for build in conflicts:
        log(f"[skip] {build.name}: parquet files already in {build.output_dir}")
    builds = [build for build in builds if build not in conflicts]
    if not builds:
        log("all requested builds already exist; nothing to do")
        return

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

    teacher_cot_probability = args.teacher_cot_probability
    if any(build.strategy == "teacher-only" for build in builds) and teacher_cot_probability is None:
        total = len(teacher_best)
        solvable_count = sum(
            1 for pid in teacher_best if is_solvable(student_best.get(pid), args.score_threshold)
        )
        unsolvable_count = total - solvable_count
        teacher_cot_probability = (
            (unsolvable_count + args.solvable_cot_alpha * solvable_count) / total if total else 0.0
        )
        log(
            f"[teacher-only] CoT probability {teacher_cot_probability:.4f} matched to the "
            f"cap-filter CoT rate ({unsolvable_count:,} unsolvable of {total:,} prompts"
            + (f", alpha={args.solvable_cot_alpha}" if args.solvable_cot_alpha else "")
            + ")"
        )

    rows_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for build in builds:
        if build.strategy not in rows_by_strategy:
            rows_by_strategy[build.strategy] = build_reasoning_rows(
                strategy=build.strategy,
                teacher_best=teacher_best,
                student_best=student_best,
                student_attempt_counts=student_attempt_counts,
                student_pass_counts=student_pass_counts,
                score_threshold=args.score_threshold,
                seed=args.seed,
                no_cot_enable_thinking_strategy=args.no_cot_enable_thinking_strategy,
                solvable_cot_alpha=args.solvable_cot_alpha,
                teacher_cot_probability=teacher_cot_probability,
            )

    mixed_builds = [build for build in builds if build.dataset_mode == "mixed"]
    sft0_rows: list[dict[str, Any]] = []
    matches: dict[int, set[tuple[str, str]]] = {}
    used_tags_by_name: dict[str, set[tuple[str, str]]] = {}
    if mixed_builds:
        sft0_rows = load_sft0_rows(args.sft0)
        used_tags_by_name = {
            build.name: used_prompt_tags(rows_by_strategy[build.strategy]) for build in mixed_builds
        }
        all_tags = set().union(*used_tags_by_name.values())
        lsh = build_prompt_index(
            all_tags,
            teacher_best,
            student_best,
            num_hashes=args.lsh_num_hashes,
            bands=args.lsh_bands,
        )
        matches = match_sft0_rows(sft0_rows, lsh, threshold=args.dedup_threshold)

    thought_counts: dict[str, int] = {}
    for build in builds:
        log(f"[build] {build.name}: strategy={build.strategy}, mode={build.dataset_mode}")
        reasoning_rows = rows_by_strategy[build.strategy]
        all_rows = reasoning_rows
        dedup_removed = 0
        sft0_count = 0
        if build.dataset_mode == "mixed":
            sft0_count = len(sft0_rows)
            kept_sft0, dedup_removed = filter_sft0_rows(sft0_rows, matches, used_tags_by_name[build.name])
            log(
                f"[dedup] {build.name}: removed {dedup_removed:,} of {sft0_count:,} "
                "SFT-0 rows overlapping reasoning prompts"
            )
            all_rows = kept_sft0 + reasoning_rows

        all_rows = filter_invalid_thinking_rows(all_rows)

        if build.strategy not in thought_counts:
            thought_counts[build.strategy] = thought_block_count(reasoning_rows)
        stats = {
            "dataset_mode": build.dataset_mode,
            "strategy": build.strategy,
            "score_threshold": args.score_threshold,
            "no_cot_enable_thinking_strategy": args.no_cot_enable_thinking_strategy,
            "solvable_cot_alpha": args.solvable_cot_alpha,
            "teacher_prompts_with_positive_score": len(teacher_best),
            "student_prompt_count": len(student_attempt_counts),
            "reasoning_rows": len(reasoning_rows),
            "reasoning_rows_with_thought_blocks": thought_counts[build.strategy],
            "reasoning_rows_with_thinking_enabled": sum(bool(row.get("enable_thinking")) for row in reasoning_rows),
            "sft0_rows_loaded": sft0_count,
            "sft0_rows_removed_by_prompt_dedup": dedup_removed,
            "final_rows": len(all_rows),
            "teacher_attempt_count_max": max(teacher_attempt_counts.values(), default=0),
            "student_attempt_count_max": max(student_attempt_counts.values(), default=0),
        }
        if build.strategy == "teacher-only":
            stats["teacher_cot_probability"] = teacher_cot_probability

        write_train_dataset(all_rows, build.output_dir, seed=args.seed)
        write_manifest(build.output_dir, stats)
        print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
