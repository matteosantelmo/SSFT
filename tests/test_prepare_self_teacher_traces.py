from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPTS))

from prepare_self_teacher_traces import (  # noqa: E402
    Candidate,
    cleaned_record,
    ensure_unique_prompts,
    prompt_id,
    select_candidates,
    write_selected,
)


def _record(
    attempt: int,
    *,
    prompt: int = 7,
    score: float = 1.0,
    reasoning: str = "reasoning that is long enough",
    response: str = "answer",
    finish_reason: str = "stop",
) -> dict:
    return {
        "id": f"math:{prompt}#{attempt}",
        "data_source": "math",
        "ability": "math",
        "repeat_idx": attempt,
        "seed": attempt,
        "prompt": [{"role": "system", "content": "format me"}],
        "original_prompt": [{"role": "user", "content": "problem"}],
        "ground_truth": "answer",
        "extra_info": {"index": 7},
        "score": score,
        "reasoning": reasoning,
        "response": response,
        "finish_reason": finish_reason,
        "completion_tokens": 12,
        "verify_seconds": 0.1,
        "raw_response": "formatted output",
        "output_format_valid": True,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_selection_matches_notebook_ranking_and_filters_unusable_rows(tmp_path):
    path = tmp_path / "results.jsonl"
    records = [
        _record(0, response="x" * 201),
        _record(1, response="short"),
        _record(2, response="a somewhat longer answer"),
        _record(3, score=0.79),
        _record(4, reasoning="too short"),
        _record(5, finish_reason="length"),
        _record(6, response=""),
    ]
    _write_jsonl(path, records)

    selected, stats = select_candidates(
        path,
        score_threshold=0.8,
        min_reasoning_chars=15,
        max_response_chars=200,
    )

    assert list(selected) == ["math:7"]
    assert selected["math:7"].response_length == len("a somewhat longer answer")
    assert stats == {
        "records": 7,
        "low_score": 1,
        "short_reasoning": 1,
        "empty_response": 1,
        "truncated": 1,
        "eligible": 3,
        "prompts_selected": 1,
        "selected_within_limits": 1,
    }


def test_equal_rank_keeps_first_source_record(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, [_record(2), _record(1)])

    selected, _ = select_candidates(
        path,
        score_threshold=0.8,
        min_reasoning_chars=15,
        max_response_chars=200,
    )
    output = tmp_path / "filtered.jsonl"
    write_selected(path, output, selected, teacher_model="self-model")

    result = json.loads(output.read_text())
    assert result["repeat_idx"] == 2


def test_output_contains_one_high_scoring_record_per_prompt(tmp_path):
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            _record(0, prompt=7, score=0.1),
            _record(1, prompt=7, score=0.8),
            _record(2, prompt=7, score=1.0, response="better answer"),
            _record(0, prompt=8, score=0.79),
            _record(1, prompt=8, score=0.9),
        ],
    )
    selected, _ = select_candidates(
        path,
        score_threshold=0.8,
        min_reasoning_chars=15,
        max_response_chars=200,
    )
    output = tmp_path / "filtered.jsonl"
    write_selected(path, output, selected, teacher_model="self-model")

    records = [json.loads(line) for line in output.read_text().splitlines()]
    prompt_ids = [prompt_id(record["id"]) for record in records]
    assert len(prompt_ids) == len(set(prompt_ids)) == 2
    assert all(record["score"] >= 0.8 for record in records)


def test_duplicate_prompt_across_result_files_is_rejected(tmp_path):
    first = tmp_path / "math" / "results.jsonl"
    second = tmp_path / "code" / "results.jsonl"
    candidate = Candidate(offset=0, response_within_limits=True, response_length=6)

    with pytest.raises(ValueError, match="refusing to emit multiple teacher traces"):
        ensure_unique_prompts(
            [
                (first, {"math:7": candidate}),
                (second, {"math:7": candidate}),
            ]
        )


def test_cleaned_record_restores_prompt_and_emits_teacher_schema():
    result = cleaned_record(
        _record(0, reasoning="  reason  ", response="  answer  "),
        "self-model",
    )

    assert result["prompt"] == [{"role": "user", "content": "problem"}]
    assert result["reasoning"] == "reason"
    assert result["response"] == "answer"
    assert result["teacher_model"] == "self-model"
    assert result["missing_reasoning"] is False
    assert "original_prompt" not in result
    assert "raw_response" not in result
    assert "output_format_valid" not in result
