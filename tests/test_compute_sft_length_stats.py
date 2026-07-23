import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "compute_sft_length_stats.py"
SPEC = importlib.util.spec_from_file_location("compute_sft_length_stats", SCRIPT_PATH)
length_stats = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = length_stats
SPEC.loader.exec_module(length_stats)


class WordTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        texts = []
        for message in messages:
            content = message.get("content", {})
            texts.extend(block["text"] for block in content.get("blocks", []) if "text" in block)
        return " ".join(texts)

    def __call__(self, texts, **_kwargs):
        return {"length": [len(text.split()) for text in texts]}


class QwenWordTokenizer(WordTokenizer):
    name_or_path = "Qwen2.5-test"
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, **_kwargs):
        assert all(isinstance(message["content"], str) for message in messages)
        return " ".join(message["content"] for message in messages)


def _messages(*blocks):
    return json.dumps(
        [
            {
                "role": "assistant",
                "content": {"blocks": list(blocks)},
            }
        ]
    )


def test_compute_statistics_counts_full_samples_and_thoughts(tmp_path):
    rows = [
        {
            "messages": _messages(
                {"type": "thoughts", "text": "one two three"},
                {"type": "response", "text": "answer"},
            ),
            "tools": "",
            "enable_thinking": True,
        },
        {
            "messages": _messages({"type": "response", "text": "short"}),
            "tools": "",
            "enable_thinking": False,
        },
    ]
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    result = length_stats.compute_length_statistics(
        [parquet_path],
        WordTokenizer(),
        batch_size=2,
        quantiles=(0.5,),
    )

    assert result["sample_tokens"]["sum"] == 5
    assert result["sample_tokens"]["median"] == 2.5
    assert result["thought_tokens_all_samples"]["sum"] == 3
    assert result["thought_tokens_all_samples"]["median"] == 1.5
    assert result["thought_tokens_samples_with_thoughts"]["mean"] == 3.0
    assert result["samples_with_thoughts"] == 1
    assert result["thought_block_count"] == 1


def test_multiple_thought_blocks_are_summed_per_sample(tmp_path):
    rows = [
        {
            "messages": _messages(
                {"type": "thoughts", "text": "one two"},
                {"type": "thoughts", "text": "three four five"},
            )
        }
    ]
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    result = length_stats.compute_length_statistics(
        [parquet_path],
        WordTokenizer(),
        batch_size=1,
        quantiles=(0.5,),
    )

    assert result["thought_tokens_samples_with_thoughts"]["sum"] == 5
    assert result["thought_block_count"] == 2


def test_update_manifest_preserves_existing_fields(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"strategy": "teacher-only", "final_rows": 2}\n', encoding="utf-8")

    length_stats.update_manifest(path, {"sample_count": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "strategy": "teacher-only",
        "final_rows": 2,
        "length_statistics": {"sample_count": 2},
    }


def test_qwen_format_converts_apertus_content_before_rendering(tmp_path):
    messages = [
        {"role": "system", "content": {"text": ""}},
        {"role": "user", "content": {"parts": [{"type": "text", "text": "question"}]}},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {"type": "thoughts", "text": "one two"},
                    {"type": "response", "text": "answer"},
                ]
            },
        },
    ]
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"messages": json.dumps(messages), "tools": "", "enable_thinking": True}]
        ),
        parquet_path,
    )
    tokenizer = QwenWordTokenizer()

    result = length_stats.compute_length_statistics(
        [parquet_path],
        tokenizer,
        batch_size=1,
        quantiles=(0.5,),
        message_format="qwen2.5",
    )

    assert result["sample_tokens"]["sum"] == 6
    assert result["thought_tokens_samples_with_thoughts"]["sum"] == 2
    assert result["processing"]["message_format"] == "qwen2.5"
    assert length_stats.resolve_message_format(tokenizer, "auto") == "qwen2.5"
