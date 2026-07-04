#!/usr/bin/env python3
"""Clean, inspect, and optionally save selected SFT sources.

Source datasets are never modified. CLI runs keep Arrow intermediates in an
automatically deleted temporary directory. When ``--output-dir`` is provided,
each processed source is written to one new Parquet file.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import datasets
from transformers import AutoTokenizer

try:
    from termcolor import colored
except ImportError:
    _ANSI_COLORS = {
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
    }

    def colored(
        text: str,
        color: str | None = None,
        attrs: list[str] | None = None,
        **_kwargs: Any,
    ) -> str:
        """Minimal terminal fallback when termcolor is unavailable."""
        if not sys.stdout.isatty():
            return text
        codes = []
        if color in _ANSI_COLORS:
            codes.append(str(_ANSI_COLORS[color]))
        if attrs and "bold" in attrs:
            codes.append("1")
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

DEFAULT_SOURCES = (
    "smoltalk2",
    "openlifescienceai_medmcqa",
    "wikipedia",
    "glaiveai_glaive-function-calling-v2",
    "jupyter-agent_jupyter-agent-dataset",
    "medical-o1-reasoning-SFT",
    "Llama-Nemotron-Post-Training-Dataset",
    "nemotron-aug-no-reasoning",
    "africa-sft",
    "tulu-3-sft-olmo-2-mixture-0225",
    "riddle_sense",
    "lipengcs_table_gpt",
    "dongfujiang_fetaqa",
    "nvidia_OpenCodeReasoning-2",
    "DeepMath-103K",
    "HARP-NuminaMath-1.5",
    "nvidia_OpenMathReasoning",
    "dolci-if-eng_Latn-200k",
    "dolci-if-eng_Latn-70k-balanced",
    "if-sft-verified-singleturn-all_MiniMaxAI_MiniMax-M2.7",
    "if-eng_Latn-12k-v1-mix-multiturn-verified",
    "multiturnIF_LSAIE",
)

MATH_SOURCES = {"HARP", "AI-MO/NuminaMath-1.5"}
THINKING_TOKENS = ("<think>", "</think>", "<|inner_prefix|>", "<|inner_suffix|>")
DISPLAY_ANSWERS = "display_answers"
METRIC_COLUMNS = ("_num_messages", "_num_tokens", "_ttr_scores")
_CHANGED = "__stage_changed"
_CONSISTENCY_ERROR = "_consistency_error"
INTERNAL_COLUMNS = (*METRIC_COLUMNS, _CHANGED, _CONSISTENCY_ERROR)
_TRANSIENT_CACHE_DIR: Path | None = None
_CACHE_COUNTER = itertools.count()

_BOX_COMMAND_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_DISPLAY_OPEN_RE = re.compile(
    r"(?:\\\[|\$\$|\\begin\{(?:equation|equation\*|displaymath)\})?\s*\Z"
)
_DISPLAY_CLOSE_RE = re.compile(
    r"\A\s*(?:\\\]|\$\$|\\end\{(?:equation|equation\*|displaymath)\})?\s*\Z"
)
_BOX_LEAD_IN_RE = re.compile(
    r"(?:\b(?:hence|thus|therefore|consequently|accordingly|so)|"
    r"\b(?:the\s+)?(?:final\s+)?answer(?:\s+is)?|"
    r"\b(?:the\s+)?result(?:\s+is)?|"
    r"\b(?:we\s+)?(?:get|obtain|find|conclude)(?:\s+that)?|\bis)"
    r"\s*[:;, .\-–—]*\Z",
    re.IGNORECASE,
)


def _cache_file(stage: str) -> Path | None:
    """Return a unique transient Arrow cache path when CLI caching is active."""
    if _TRANSIENT_CACHE_DIR is None:
        return None
    safe_stage = re.sub(r"[^A-Za-z0-9]+", "_", stage).strip("_")
    return _TRANSIENT_CACHE_DIR / f"{next(_CACHE_COUNTER):03d}_{safe_stage}.arrow"


def _transform_storage_kwargs(stage: str) -> dict[str, Any]:
    cache_file = _cache_file(stage)
    if cache_file is None:
        # Preserve safe behavior for callers that import this module directly.
        return {"keep_in_memory": True}
    return {"keep_in_memory": False, "cache_file_name": str(cache_file)}


def map_with_report(
    dataset: datasets.Dataset,
    function: Callable[..., dict[str, Any]],
    stage: str,
    **kwargs: Any,
) -> datasets.Dataset:
    """Run an in-memory map and print how many rows the function changed."""
    if _CHANGED in dataset.column_names:
        raise ValueError(f"reserved column already present: {_CHANGED}")
    # datasets treats num_proc=1 as a multiprocessing request.  Staying in the
    # parent process is both cheaper and usable on compute nodes that disallow
    # process-manager sockets.
    if kwargs.get("num_proc") == 1:
        kwargs["num_proc"] = None
    mapped = dataset.map(
        function,
        desc=stage,
        **_transform_storage_kwargs(stage),
        **kwargs,
    )
    changed = sum(bool(value) for value in mapped[_CHANGED])
    print(
        colored(f"[{stage}]", "cyan", attrs=["bold"]),
        colored(
            f"modified {changed:,} / {len(mapped):,} samples",
            "yellow" if changed else "green",
        ),
    )
    return mapped.remove_columns(_CHANGED)


def filter_with_report(
    dataset: datasets.Dataset,
    predicate: Callable[..., bool],
    stage: str,
    **kwargs: Any,
) -> datasets.Dataset:
    """Run an in-memory filter and print exactly how many rows were removed."""
    before = len(dataset)
    if kwargs.get("num_proc") == 1:
        kwargs["num_proc"] = None
    filtered = dataset.filter(
        predicate,
        desc=stage,
        **_transform_storage_kwargs(stage),
        **kwargs,
    )
    removed = before - len(filtered)
    print(
        colored(f"[{stage}]", "cyan", attrs=["bold"]),
        colored(
            f"filtered {removed:,} / {before:,} samples",
            "yellow" if removed else "green",
        ),
    )
    return filtered


def get_train_split(dataset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    if isinstance(dataset, datasets.DatasetDict):
        if "train" not in dataset:
            raise ValueError(f"DatasetDict has no train split: {tuple(dataset)}")
        return dataset["train"]
    return dataset


def subsample(
    dataset: datasets.Dataset, sample_size: int | None, seed: int
) -> datasets.Dataset:
    if sample_size is None or sample_size >= len(dataset):
        print(
            colored("[subsample]", "cyan", attrs=["bold"]),
            colored(f"keeping all {len(dataset):,} samples", "green"),
        )
        return dataset
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    cache_file = _cache_file("subsample_indices")
    shuffle_kwargs: dict[str, Any] = {"seed": seed}
    if cache_file is None:
        shuffle_kwargs["keep_in_memory"] = True
    else:
        shuffle_kwargs["keep_in_memory"] = False
        shuffle_kwargs["indices_cache_file_name"] = str(cache_file)
    sampled = dataset.shuffle(**shuffle_kwargs).select(range(sample_size))
    print(
        colored("[subsample]", "cyan", attrs=["bold"]),
        colored(
            f"selected {len(sampled):,} / {len(dataset):,} samples", "yellow"
        ),
    )
    return sampled


def empty_content() -> dict[str, Any]:
    return {
        "text": "",
        "tools": "",
        "has_thinking": False,
        "formatted_tools": "",
        "parts": [],
        "blocks": [],
    }


def format_math_example(example: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one selected HARP/NuminaMath row to the local message schema."""
    source = example["source"]
    source_id = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")
    return {
        "conversation_id": f"{source_id}_{index}",
        "dataset_source": source,
        "original_metadata": json.dumps(
            {
                "problem": example["problem"],
                "source": source,
                "source_index": index,
            },
            ensure_ascii=False,
        ),
        "created_timestamp": "",
        "messages": [
            {"role": "system", "content": empty_content()},
            {"role": "developer", "content": empty_content()},
            {
                "role": "user",
                "content": {
                    **empty_content(),
                    "parts": [{"type": "text", "text": example["problem"]}],
                },
            },
            {
                "role": "assistant",
                "content": {
                    **empty_content(),
                    "blocks": [
                        {
                            "type": "response",
                            "text": example["solution"],
                            "calls": [{"name": "", "arguments": ""}],
                            "outputs": [{"name": "", "output": ""}],
                        }
                    ],
                },
            },
        ],
        _CHANGED: True,
    }


def load_source(
    source: str,
    input_dir: Path,
    math_dataset_path: Path,
    sample_size: int | None,
    seed: int,
    num_proc: int | None,
) -> datasets.Dataset:
    """Load one source; rebuild the HARP/NuminaMath source in memory."""
    if source != "HARP-NuminaMath-1.5":
        dataset = get_train_split(datasets.load_from_disk(str(input_dir / source)))
        return subsample(dataset, sample_size, seed)

    raw = get_train_split(datasets.load_from_disk(str(math_dataset_path)))
    raw = filter_with_report(
        raw,
        lambda value: value in MATH_SOURCES,
        "HARP/NuminaMath: select sources",
        input_columns=["source"],
        num_proc=num_proc,
    )
    raw = subsample(raw, sample_size, seed)
    target_features = get_train_split(
        datasets.load_from_disk(str(input_dir / "DeepMath-103K"))
    ).features
    # The marker is temporary and therefore must be added to the target schema.
    map_features = copy.deepcopy(target_features)
    map_features[_CHANGED] = datasets.Value("bool")
    return map_with_report(
        raw,
        format_math_example,
        "HARP/NuminaMath: format conversations",
        with_indices=True,
        remove_columns=raw.column_names,
        features=map_features,
        num_proc=num_proc,
    )


def _matching_brace(text: str, opening_brace: int) -> int | None:
    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return index
    return None


def strip_terminal_repeated_box(text: str) -> tuple[str, bool]:
    """Remove a standalone terminal box only when its answer appeared earlier."""
    trimmed_end = len(text.rstrip())
    candidate_text = text[:trimmed_end]
    for match in reversed(list(_BOX_COMMAND_RE.finditer(candidate_text))):
        closing_brace = _matching_brace(candidate_text, match.end() - 1)
        if closing_brace is None or not _DISPLAY_CLOSE_RE.fullmatch(
            candidate_text[closing_brace + 1 :]
        ):
            continue
        boundaries = list(_BLANK_LINE_RE.finditer(candidate_text, 0, match.start()))
        if not boundaries:
            continue
        boundary = boundaries[-1]
        if not _DISPLAY_OPEN_RE.fullmatch(candidate_text[boundary.end() : match.start()]):
            continue
        answer = candidate_text[match.end() : closing_brace].strip()
        earlier_text = candidate_text[: boundary.start()].rstrip()
        if _BOX_LEAD_IN_RE.search(earlier_text):
            continue
        compact_answer = re.sub(r"\s+", "", answer)
        compact_earlier = re.sub(r"\s+", "", earlier_text)
        if compact_answer and compact_answer in compact_earlier:
            return earlier_text + text[trimmed_end:], True
    return text, False


def strip_deepmath_boxes(example: dict[str, Any]) -> dict[str, Any]:
    messages = copy.deepcopy(example["messages"])
    changed = False
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in (message.get("content") or {}).get("blocks") or []:
            if block.get("type") == "response":
                block["text"], block_changed = strip_terminal_repeated_box(
                    block.get("text") or ""
                )
                changed |= block_changed
    return {"messages": messages, _CHANGED: changed}


def filter_table_gpt(example: dict[str, Any]) -> bool:
    try:
        fewshots = json.loads(example["original_metadata"])["num_fewshots"]
        return 0 <= int(fewshots) <= 2
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def fetaqa_has_long_responses(
    example: dict[str, Any], threshold: int = 50
) -> bool:
    has_response = False
    for message in example["messages"]:
        if message.get("role") != "assistant":
            continue
        for block in (message.get("content") or {}).get("blocks") or []:
            if block.get("type") != "response":
                continue
            has_response = True
            if len(block.get("text") or "") < threshold:
                return False
    return has_response


def remove_thinking_blocks(example: dict[str, Any]) -> dict[str, Any]:
    messages = copy.deepcopy(example["messages"])
    changed = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        if content.get("has_thinking"):
            content["has_thinking"] = False
            changed = True
        blocks = content.get("blocks")
        if isinstance(blocks, list):
            kept = [block for block in blocks if block.get("type") != "thoughts"]
            changed |= len(kept) != len(blocks)
            content["blocks"] = kept
    return {"messages": messages, _CHANGED: changed}


def has_structured_thinking(example: dict[str, Any]) -> bool:
    for message in example["messages"]:
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        if content.get("has_thinking"):
            return True
        if any(
            block.get("type") == "thoughts" for block in (content.get("blocks") or [])
        ):
            return True
    return False


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _strings(nested)


def has_literal_thinking_token(example: dict[str, Any]) -> bool:
    return any(token in text for text in _strings(example["messages"]) for token in THINKING_TOKENS)


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    if isinstance(tool.get("name"), str):
        return tool["name"]
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _decode_tools(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, "", []):
        return []
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(parsed, dict) and isinstance(parsed.get("tools"), list):
        parsed = parsed["tools"]
    if not isinstance(parsed, list):
        raise ValueError("tool definitions must be a JSON list")
    return [tool for tool in parsed if isinstance(tool, dict)]


def remove_display_answers(example: dict[str, Any]) -> dict[str, Any]:
    messages = copy.deepcopy(example["messages"])
    changed = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, dict):
            continue

        raw_tools = content.get("tools")
        if raw_tools:
            try:
                tools = _decode_tools(raw_tools)
            except (TypeError, ValueError, json.JSONDecodeError):
                tools = None
            if tools is not None:
                kept_tools = [tool for tool in tools if _tool_name(tool) != DISPLAY_ANSWERS]
                if len(kept_tools) != len(tools):
                    content["tools"] = json.dumps(kept_tools, ensure_ascii=False)
                    # Downstream tokenization uses the filtered JSON definitions.
                    content["formatted_tools"] = ""
                    changed = True
        if DISPLAY_ANSWERS in (content.get("formatted_tools") or ""):
            # The canonical JSON definitions are used for rendering below.  A
            # stale preformatted copy must not leak the removed tool.
            content["formatted_tools"] = ""
            changed = True

        blocks = content.get("blocks")
        if not isinstance(blocks, list):
            continue
        cleaned: list[dict[str, Any]] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]
            if block.get("type") != "tool_calls":
                if block.get("type") == "tool_outputs":
                    outputs = block.get("outputs") or []
                    kept_outputs = [
                        output
                        for output in outputs
                        if output.get("name") != DISPLAY_ANSWERS
                    ]
                    if len(kept_outputs) != len(outputs):
                        changed = True
                        if kept_outputs:
                            block = copy.deepcopy(block)
                            block["outputs"] = kept_outputs
                        else:
                            index += 1
                            continue
                cleaned.append(block)
                index += 1
                continue

            calls = block.get("calls") or []
            display_indices = {
                i for i, call in enumerate(calls) if call.get("name") == DISPLAY_ANSWERS
            }
            if not display_indices:
                cleaned.append(block)
                index += 1
                continue

            changed = True
            kept_calls = [call for i, call in enumerate(calls) if i not in display_indices]
            if kept_calls:
                kept_call_block = copy.deepcopy(block)
                kept_call_block["calls"] = kept_calls
                cleaned.append(kept_call_block)

            if index + 1 < len(blocks) and blocks[index + 1].get("type") == "tool_outputs":
                output_block = blocks[index + 1]
                outputs = output_block.get("outputs") or []
                kept_outputs = [
                    output for i, output in enumerate(outputs) if i not in display_indices
                ]
                if kept_outputs:
                    kept_output_block = copy.deepcopy(output_block)
                    kept_output_block["outputs"] = kept_outputs
                    cleaned.append(kept_output_block)
                index += 2
            else:
                index += 1
        content["blocks"] = cleaned
    return {"messages": messages, _CHANGED: changed}


def display_answers_is_absent(example: dict[str, Any]) -> bool:
    for message in example["messages"]:
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        try:
            if any(_tool_name(tool) == DISPLAY_ANSWERS for tool in _decode_tools(content.get("tools"))):
                return False
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        if DISPLAY_ANSWERS in (content.get("formatted_tools") or ""):
            return False
        for block in content.get("blocks") or []:
            if any(call.get("name") == DISPLAY_ANSWERS for call in block.get("calls") or []):
                return False
    return True


def defined_tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        for tool in _decode_tools(content.get("tools")):
            name = _tool_name(tool)
            if name:
                names.add(name)
    return names


def consistency_error(example: dict[str, Any]) -> str | None:
    messages = example.get("messages") or []
    if not messages:
        return "empty conversation"
    if messages[-1].get("role") not in {"assistant", "tool"}:
        return "last message is neither assistant nor tool"
    try:
        defined = defined_tool_names(messages)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "invalid tool definitions"

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        blocks = content.get("blocks") if isinstance(content, dict) else None
        if not isinstance(blocks, list) or not blocks:
            return "assistant message has no blocks"
        for block_index, block in enumerate(blocks):
            if block.get("type") != "tool_calls":
                continue
            calls = [call for call in (block.get("calls") or []) if call.get("name")]
            if not calls:
                return "empty tool call block"
            if any(call["name"] not in defined for call in calls):
                return "tool call refers to an undefined tool"

            is_terminal_final_answer = (
                len(calls) == 1
                and calls[0]["name"] == "final_answer"
                and message_index == len(messages) - 1
                and block_index == len(blocks) - 1
            )
            if is_terminal_final_answer:
                # final_answer completes the conversation; unlike execution
                # tools, it intentionally has no tool output.
                continue

            if block_index + 1 < len(blocks) and blocks[block_index + 1].get("type") == "tool_outputs":
                outputs = blocks[block_index + 1].get("outputs") or []
                if len(outputs) != len(calls):
                    return "tool call/output counts differ"
                continue

            next_message = message_index + 1
            output_messages = 0
            while next_message < len(messages) and messages[next_message].get("role") == "tool":
                output_messages += 1
                next_message += 1
            if output_messages < len(calls):
                return "tool calls are not followed by tool outputs"
    return None


def add_consistency_error(example: dict[str, Any]) -> dict[str, Any]:
    error = consistency_error(example)
    return {_CONSISTENCY_ERROR: error, _CHANGED: error is not None}


def print_consistency_failures(dataset: datasets.Dataset) -> None:
    """Print precomputed row-level failures for an explicitly requested debug run."""
    conversation_ids = dataset["conversation_id"]
    errors = dataset[_CONSISTENCY_ERROR]
    for index, (conversation_id, error) in enumerate(zip(conversation_ids, errors)):
        if error is None:
            continue
        print(
            colored("[consistency debug]", "red", attrs=["bold"]),
            f"row={index} conversation_id={conversation_id}: {error}",
        )


def tools_for_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, dict):
            tools.extend(_decode_tools(content.get("tools")))
    return tools


def ttr_chunked(
    chunk: list[int], n: int = 2, window: int = 512, step: int = 512
) -> list[float]:
    """Compute n-gram TTR over response windows that contain at least one n-gram."""
    ttr_scores = []
    for start in range(0, len(chunk), step):
        window_chunk = chunk[start : start + window]
        ngram_count = len(window_chunk) - n + 1
        if ngram_count <= 0:
            continue
        ngrams = {
            tuple(window_chunk[i : i + n]) for i in range(ngram_count)
        }
        ttr_scores.append(len(ngrams) / ngram_count)
    return ttr_scores


def make_add_metrics(tokenizer: Any, ttr_n: int, ttr_window: int, ttr_step: int):
    def add_metrics(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        rendered_conversations = []
        response_texts = []
        message_counts = []
        for messages in batch["messages"]:
            tools = tools_for_template(messages)
            # In the linearised dataset the developer row stores preprocessing
            # settings (has_thinking/tools). Apertus templates generate their
            # own developer section and reject an input message with that role.
            template_messages = [
                message for message in messages if message.get("role") != "developer"
            ]
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": False,
                "enable_thinking": False,
            }
            if tools:
                template_kwargs["tools"] = tools
            rendered_conversations.append(
                tokenizer.apply_chat_template(template_messages, **template_kwargs)
            )
            response_texts.append(
                "\n".join(
                    block.get("text") or ""
                    for message in messages
                    if message.get("role") == "assistant"
                    for block in (message.get("content") or {}).get("blocks") or []
                    if block.get("type") == "response"
                )
            )
            message_counts.append(len(messages))

        # Rendering is conversation-specific because tool definitions differ,
        # while the fast tokenizer backend can encode all rendered texts at once.
        tokenized = tokenizer(
            rendered_conversations,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        response_tokenized = tokenizer(
            response_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        return {
            "_num_messages": message_counts,
            "_num_tokens": [len(token_ids) for token_ids in tokenized],
            "_ttr_scores": [
                ttr_chunked(token_ids, n=ttr_n, window=ttr_window, step=ttr_step)
                for token_ids in response_tokenized
            ],
            _CHANGED: [True] * len(message_counts),
        }

    return add_metrics


def save_length_plot(dataset: datasets.Dataset, source: str, plot_dir: Path) -> None:
    """Save one histogram only when explicitly requested; never overwrite."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    output = plot_dir / f"{source}-token-lengths.png"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing plot: {output}")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(dataset["_num_tokens"], bins=100)
    axis.set(title=f"Token lengths: {source}", xlabel="tokens", ylabel="samples")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(
        colored("[plot]", "blue", attrs=["bold"]),
        colored(f"saved {output}", "green"),
    )


def print_metric_summary(dataset: datasets.Dataset) -> None:
    lengths = dataset["_num_tokens"]
    message_counts = dataset["_num_messages"]
    if not lengths:
        print(colored("[metrics]", "blue", attrs=["bold"]), "no samples")
        return
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    print(
        colored("[metrics]", "blue", attrs=["bold"]),
        f"tokens min/median/p95/max={ordered[0]:,}/{percentile(0.5):,}/"
        f"{percentile(0.95):,}/{ordered[-1]:,}; "
        f"messages min/max={min(message_counts):,}/{max(message_counts):,}"
    )


def output_file(output_dir: Path, source: str) -> Path:
    safe_source = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")
    return output_dir / f"{safe_source}.parquet"


def save_processed_dataset(
    dataset: datasets.Dataset, source: str, output_dir: Path
) -> Path:
    """Write one source to one Parquet file without overwriting existing data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_file(output_dir, source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    dataset.to_parquet(str(output), compression="zstd")
    print(
        colored("[save]", "blue", attrs=["bold"]),
        colored(f"saved {len(dataset):,} samples to {output}", "green"),
    )
    return output


def process_source(
    source: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> datasets.Dataset:
    print("\n" + colored(f"=== {source} ===", "magenta", attrs=["bold"]))
    dataset = load_source(
        source=source,
        input_dir=args.input_dir,
        math_dataset_path=args.math_dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        num_proc=args.num_proc,
    )

    if source == "lipengcs_table_gpt":
        dataset = filter_with_report(
            dataset,
            filter_table_gpt,
            "table-gpt: keep 0-2 few-shots",
            num_proc=args.num_proc,
        )
    elif source == "dongfujiang_fetaqa":
        dataset = filter_with_report(
            dataset,
            fetaqa_has_long_responses,
            "fetaqa: remove short responses",
            fn_kwargs={"threshold": args.fetaqa_min_chars},
            num_proc=args.num_proc,
        )
    elif source == "DeepMath-103K":
        dataset = map_with_report(
            dataset,
            strip_deepmath_boxes,
            "DeepMath: strip repeated terminal boxes",
            num_proc=args.num_proc,
        )

    dataset = map_with_report(
        dataset,
        remove_thinking_blocks,
        "remove structured thinking",
        num_proc=args.num_proc,
    )
    dataset = filter_with_report(
        dataset,
        lambda example: not has_structured_thinking(example),
        "verify structured thinking removed",
        num_proc=args.num_proc,
    )
    dataset = map_with_report(
        dataset,
        remove_display_answers,
        "remove display_answers",
        num_proc=args.num_proc,
    )
    dataset = filter_with_report(
        dataset,
        display_answers_is_absent,
        "verify display_answers removed",
        num_proc=args.num_proc,
    )
    dataset = filter_with_report(
        dataset,
        lambda example: not has_literal_thinking_token(example),
        "discard literal thinking tokens",
        num_proc=args.num_proc,
    )
    if args.debug:
        dataset = map_with_report(
            dataset,
            add_consistency_error,
            "compute conversation consistency",
            num_proc=args.num_proc,
        )
        print_consistency_failures(dataset)
        dataset = filter_with_report(
            dataset,
            lambda error: error is None,
            "conversation consistency",
            input_columns=[_CONSISTENCY_ERROR],
            num_proc=args.num_proc,
        ).remove_columns(_CONSISTENCY_ERROR)
    else:
        dataset = filter_with_report(
            dataset,
            lambda example: consistency_error(example) is None,
            "conversation consistency",
            num_proc=args.num_proc,
        )

    dataset = map_with_report(
        dataset,
        make_add_metrics(
            tokenizer, args.ttr_n, args.ttr_window, args.ttr_step
        ),
        "tokenize and compute metrics",
        batched=True,
        batch_size=args.tokenizer_batch_size,
        num_proc=args.num_proc,
    )
    print_metric_summary(dataset)
    if args.plot_dir is not None:
        save_length_plot(dataset, source, args.plot_dir)

    dataset = filter_with_report(
        dataset,
        lambda count: count >= args.min_messages
        and (args.max_messages is None or count <= args.max_messages),
        "message-count bounds",
        input_columns=["_num_messages"],
        num_proc=args.num_proc,
    )
    dataset = filter_with_report(
        dataset,
        lambda length: args.max_tokens is None or length <= args.max_tokens,
        "token-length bound",
        input_columns=["_num_tokens"],
        num_proc=args.num_proc,
    )
    dataset = filter_with_report(
        dataset,
        lambda scores: bool(scores) and min(scores) >= args.min_ttr,
        "chunked-TTR bound",
        input_columns=["_ttr_scores"],
        num_proc=args.num_proc,
    )
    dataset = dataset.remove_columns(
        [column for column in INTERNAL_COLUMNS if column in dataset.column_names]
    )
    saved_path = None
    if args.output_dir is not None:
        saved_path = save_processed_dataset(dataset, source, args.output_dir)
    print(
        colored("[done]", "green", attrs=["bold"]),
        colored(
            f"{source}: {len(dataset):,} samples retained"
            + (f"; saved to {saved_path}" if saved_path else "; nothing saved"),
            "green",
        ),
    )
    return dataset


def optional_positive_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or Hugging Face ID")
    parser.add_argument("--input-dir", type=Path, default=Path("lin_data_splits"))
    parser.add_argument("--math-dataset", type=Path, default=Path("math_dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Save each processed source as one new Parquet file in this directory",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=DEFAULT_SOURCES,
        help="Process only this source; repeat for several (default: all)",
    )
    parser.add_argument("--sample-size", type=optional_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=optional_positive_int, default=16)
    parser.add_argument("--tokenizer-batch-size", type=optional_positive_int, default=128)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print row IDs and reasons for conversation consistency failures",
    )
    parser.add_argument("--fetaqa-min-chars", type=int, default=50)
    parser.add_argument("--min-messages", type=int, default=1)
    parser.add_argument("--max-messages", type=optional_positive_int, default=None)
    parser.add_argument("--max-tokens", type=optional_positive_int, default=16_384)
    parser.add_argument("--min-ttr", type=float, default=0.25)
    parser.add_argument("--ttr-n", type=int, default=2)
    parser.add_argument("--ttr-window", type=int, default=512)
    parser.add_argument("--ttr-step", type=int, default=512)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Opt in to plots. Existing files are never overwritten.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_ttr <= 1:
        parser.error("--min-ttr must be between 0 and 1")
    if args.min_messages < 0:
        parser.error("--min-messages must be non-negative")
    return args


def main() -> None:
    global _TRANSIENT_CACHE_DIR

    args = parse_args()
    sources = args.source or DEFAULT_SOURCES
    if args.output_dir is not None:
        targets = [output_file(args.output_dir, source) for source in sources]
        if len(set(targets)) != len(targets):
            raise ValueError("multiple sources resolve to the same output file")
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing datasets: "
                + ", ".join(str(path) for path in existing)
            )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Keep large Arrow intermediates off RAM and away from source directories.
    # The temporary cache and all multiprocessing shards are deleted on exit.
    with tempfile.TemporaryDirectory(prefix="process_sft_data-") as cache_dir:
        _TRANSIENT_CACHE_DIR = Path(cache_dir)
        try:
            for source in sources:
                process_source(source, tokenizer, args)
        finally:
            _TRANSIENT_CACHE_DIR = None


if __name__ == "__main__":
    main()
