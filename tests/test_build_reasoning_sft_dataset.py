import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data" / "build_reasoning_sft_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_reasoning_sft_dataset", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _record(prompt_id, repeat_idx, score, response, reasoning=None):
    return {
        "id": f"source:{prompt_id}#{repeat_idx}",
        "repeat_idx": repeat_idx,
        "score": score,
        "prompt": [{"role": "user", "content": f"Question {prompt_id}"}],
        "response": response,
        "reasoning": reasoning,
    }


def _assistant_blocks(row):
    messages = json.loads(row["messages"])
    return messages[-1]["content"]["blocks"]


def test_thought_rows_force_enable_thinking():
    row = builder.make_sft_row(
        _record("hard", 0, 1.0, "Answer", reasoning="Work it out"),
        include_reasoning=True,
        enable_thinking=False,
        source="teacher",
    )

    assert row["enable_thinking"] is True
    assert _assistant_blocks(row) == [
        {"type": "thoughts", "text": "Work it out"},
        {"type": "response", "text": "Answer"},
    ]


def test_none_strategy_disables_thinking_for_no_cot_rows():
    teacher_best = {
        "source:easy": _record("easy", 0, 1.0, "Teacher answer", reasoning="Teacher CoT"),
    }
    student = _record("easy", 0, 1.0, "Student answer")

    rows = builder.build_reasoning_rows(
        strategy="solvable-student-else-teacher",
        teacher_best=teacher_best,
        student_best={"source:easy": student},
        student_attempt_counts={"source:easy": 2},
        student_pass_counts={"source:easy": 2},
        score_threshold=0.5,
        seed=0,
        no_cot_enable_thinking_strategy="none",
    )

    assert rows[0]["enable_thinking"] is False
    assert _assistant_blocks(rows[0]) == [{"type": "response", "text": "Student answer"}]


def test_random_strategy_mixes_thinking_flags_for_no_cot_rows():
    pids = [f"p{i}" for i in range(10)]
    teacher_best = {
        f"source:{pid}": _record(pid, 0, 1.0, "Teacher answer", reasoning="Teacher CoT")
        for pid in pids
    }
    student_best = {f"source:{pid}": _record(pid, 0, 1.0, "Student answer") for pid in pids}

    rows = builder.build_reasoning_rows(
        strategy="solvable-student-else-teacher",
        teacher_best=teacher_best,
        student_best=student_best,
        student_attempt_counts={f"source:{pid}": 1 for pid in pids},
        student_pass_counts={f"source:{pid}": 1 for pid in pids},
        score_threshold=0.5,
        seed=0,
        no_cot_enable_thinking_strategy="random",
    )

    flags = {row["enable_thinking"] for row in rows}
    assert flags == {True, False}


def test_prompt_text_prefers_parts_when_struct_union_text_is_empty():
    content = {
        "text": "",
        "parts": [{"type": "text", "text": "Actual prompt"}],
        "blocks": [],
    }

    assert builder.content_text_for_prompt(content) == "Actual prompt"


def test_load_best_generations_streams_best_positive_and_counts(tmp_path):
    result_file = tmp_path / "results.jsonl"
    records = [
        _record("item", 0, 0.0, "Wrong"),
        _record("item", 1, 0.75, "Good", reasoning="Work"),
        _record("item", 2, 1.0, "Better"),
    ]
    result_file.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    best, attempt_counts, pass_counts = builder.load_best_generations(
        [result_file],
        max_attempts=4,
        score_threshold=0.5,
        source_name="test",
    )

    assert best["source:item"]["response"] == "Better"
    assert attempt_counts == {"source:item": 3}
    assert pass_counts == {"source:item": 2}


def test_self_teacher_pool_wins_score_ties_and_external_fills_gaps():
    external = {
        "source:tie": _record("tie", 0, 1.0, "External tie", reasoning="External CoT"),
        "source:higher": _record("higher", 0, 1.0, "External higher", reasoning="External CoT"),
        "source:fallback": _record("fallback", 0, 1.0, "External fallback", reasoning="External CoT"),
    }
    self_teacher = {
        "source:tie": _record("tie", 0, 1.0, "Self tie", reasoning="Self CoT"),
        "source:higher": _record("higher", 0, 0.9, "Self lower", reasoning="Self CoT"),
        "source:self-only": _record("self-only", 0, 1.0, "Self only", reasoning="Self CoT"),
        "source:fallback": _record("fallback", 0, 1.0, "No reasoning"),
    }

    merged, stats = builder.merge_teacher_pools(external, self_teacher)

    assert merged["source:tie"]["response"] == "Self tie"
    assert merged["source:higher"]["response"] == "External higher"
    assert merged["source:self-only"]["response"] == "Self only"
    assert merged["source:fallback"]["response"] == "External fallback"
    assert stats == {
        "external_candidates": 3,
        "self_teacher_candidates": 4,
        "self_teacher_without_reasoning": 1,
        "selected_self_teacher": 2,
        "selected_external_teacher": 2,
    }


def test_reasoning_and_response_are_stripped_in_blocks():
    row = builder.make_sft_row(
        _record("hard", 0, 1.0, "  Answer \n", reasoning="\n Work it out  "),
        include_reasoning=True,
        source="teacher",
    )

    assert row["enable_thinking"] is True
    assert _assistant_blocks(row) == [
        {"type": "thoughts", "text": "Work it out"},
        {"type": "response", "text": "Answer"},
    ]


def test_whitespace_only_reasoning_is_a_no_thinking_sample():
    row = builder.make_sft_row(
        _record("easy", 0, 1.0, "Answer", reasoning="\n  \n"),
        include_reasoning=True,
        enable_thinking=False,
        source="teacher",
    )

    assert row["enable_thinking"] is False
    assert _assistant_blocks(row) == [{"type": "response", "text": "Answer"}]


def test_normalized_generation_nulls_whitespace_reasoning():
    record = builder.normalized_generation(_record("easy", 0, 1.0, "  Answer  ", reasoning="   "))

    assert record["reasoning"] is None
    assert record["response"] == "Answer"


def test_truncated_positives_never_count_as_correct(tmp_path):
    result_file = tmp_path / "results.jsonl"
    records = [
        {**_record("cut", 0, 1.0, "Truncated but scored"), "finish_reason": "length"},
        {**_record("cut", 1, 0.8, "Complete answer"), "finish_reason": "stop"},
        {**_record("all-cut", 0, 1.0, "Truncated only"), "finish_reason": "length"},
    ]
    result_file.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    best, attempt_counts, pass_counts = builder.load_best_generations(
        [result_file],
        max_attempts=4,
        score_threshold=0.5,
        source_name="test",
    )

    assert best["source:cut"]["response"] == "Complete answer"
    assert pass_counts == {"source:cut": 1}
    assert "source:all-cut" not in best
    assert attempt_counts == {"source:cut": 2, "source:all-cut": 1}


def test_truncated_student_best_is_not_solvable():
    teacher_best = {
        "source:hard": _record("hard", 0, 1.0, "Teacher answer", reasoning="Teacher CoT"),
    }
    truncated_student = {**_record("hard", 0, 1.0, "Cut off"), "finish_reason": "length"}

    rows = builder.build_reasoning_rows(
        strategy="solvable-student-else-teacher",
        teacher_best=teacher_best,
        student_best={"source:hard": truncated_student},
        student_attempt_counts={"source:hard": 1},
        student_pass_counts={},
        score_threshold=0.5,
        seed=0,
        no_cot_enable_thinking_strategy="none",
    )

    assert json.loads(rows[0]["metadata"])["origin"] == "teacher"
    assert _assistant_blocks(rows[0]) == [
        {"type": "thoughts", "text": "Teacher CoT"},
        {"type": "response", "text": "Teacher answer"},
    ]


def test_split_reasoning_recovers_close_without_open():
    reasoning, response = builder.split_reasoning("Let me think.\n</think>\n\nThe answer is 5.")
    assert reasoning == "Let me think.\n"
    assert response == "\n\nThe answer is 5."

    reasoning, response = builder.split_reasoning("thoughts<|inner_suffix|>answer")
    assert reasoning == "thoughts"
    assert response == "answer"

    reasoning, response = builder.split_reasoning("no delimiters at all")
    assert reasoning is None
    assert response == "no delimiters at all"


def test_normalized_generation_recovers_unopened_reasoning():
    record = builder.normalized_generation(
        _record("hard", 0, 1.0, "Deep thoughts here.\n</think>\n\nFinal answer: 42.")
    )

    assert record["reasoning"] == "Deep thoughts here."
    assert record["response"] == "Final answer: 42."

    row = builder.make_sft_row(record, include_reasoning=True, source="teacher")
    assert row["enable_thinking"] is True
    assert _assistant_blocks(row) == [
        {"type": "thoughts", "text": "Deep thoughts here."},
        {"type": "response", "text": "Final answer: 42."},
    ]


def test_rows_carry_metadata_json():
    record = {
        **_record("hard", 1, 1.0, "Answer", reasoning="CoT"),
        "teacher_model": "teacher-a",
        "completion_tokens": 123,
    }
    row = builder.make_sft_row(
        record,
        include_reasoning=True,
        source="teacher",
        student_attempt_count=4,
        student_pass_count=0,
    )

    assert json.loads(row["metadata"]) == {
        "origin": "teacher",
        "sample_id": "source:hard#1",
        "prompt_id": "source:hard",
        "data_source": "source",
        "model": "teacher-a",
        "repeat_idx": 1,
        "score": 1.0,
        "completion_tokens": 123,
        "student_attempt_count": 4,
        "student_pass_count": 0,
    }


def test_build_reasoning_rows_threads_student_counts_into_metadata():
    teacher_best = {
        "source:hard": _record("hard", 0, 1.0, "Teacher answer", reasoning="Teacher CoT"),
    }
    student = _record("hard", 0, 1.0, "Student answer")

    rows = builder.build_reasoning_rows(
        strategy="solvable-student-else-teacher",
        teacher_best=teacher_best,
        student_best={"source:hard": student},
        student_attempt_counts={"source:hard": 4},
        student_pass_counts={"source:hard": 1},
        score_threshold=0.5,
        seed=0,
        no_cot_enable_thinking_strategy="none",
    )

    metadata = json.loads(rows[0]["metadata"])
    assert metadata["origin"] == "student"
    assert metadata["student_attempt_count"] == 4
    assert metadata["student_pass_count"] == 1


def test_sft0_rows_carry_origin_metadata():
    messages = [
        {"role": "user", "content": {"parts": [{"type": "text", "text": "Hi"}]}},
        {"role": "assistant", "content": {"blocks": [{"type": "response", "text": "Hello"}]}},
    ]
    row = builder.normalize_sft0_row(
        {"messages": json.dumps(messages), "tools": "", "enable_thinking": False}
    )

    assert json.loads(row["metadata"]) == {
        "origin": "sft0",
        "sample_id": None,
        "prompt_id": None,
        "data_source": None,
        "model": None,
        "repeat_idx": None,
        "score": None,
        "completion_tokens": None,
        "student_attempt_count": None,
        "student_pass_count": None,
    }


def test_filter_invalid_thinking_rows():
    clean = {"messages": "clean", "enable_thinking": False}
    valid_thinking = {
        "messages": "<|inner_prefix|>reasoning<|inner_suffix|>answer",
        "enable_thinking": True,
    }
    invalid_prefix = {"messages": "<|inner_prefix|>reasoning", "enable_thinking": False}
    invalid_suffix = {"messages": "reasoning<|inner_suffix|>", "enable_thinking": False}
    invalid_channel = {"messages": "<|channel>thought\nreasoning", "enable_thinking": False}

    kept = builder.filter_invalid_thinking_rows(
        [clean, valid_thinking, invalid_prefix, invalid_suffix, invalid_channel]
    )

    assert kept == [clean, valid_thinking]


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _sft0_parquet_row(text, answer):
    messages = [
        {"role": "user", "content": {"parts": [{"type": "text", "text": text}]}},
        {"role": "assistant", "content": {"blocks": [{"type": "response", "text": answer}]}},
    ]
    return {"messages": json.dumps(messages), "tools": "", "enable_thinking": False}


def _run_cli(argv):
    old_argv = sys.argv
    sys.argv = ["build_reasoning_sft_dataset.py", *argv]
    try:
        builder.main()
    finally:
        sys.argv = old_argv


def test_mixed_dedup_filters_per_build():
    teacher_best = {
        "source:a": builder.normalized_generation(
            _record("a", 0, 1.0, "Teacher answer A", reasoning="Teacher CoT A")
        ),
        "source:b": builder.normalized_generation(
            _record("b", 0, 1.0, "Teacher answer B", reasoning="Teacher CoT B")
        ),
    }
    student_best = {
        "source:a": builder.normalized_generation(_record("a", 0, 1.0, "Student answer A")),
    }
    common = dict(
        teacher_best=teacher_best,
        student_best=student_best,
        student_attempt_counts={"source:a": 1, "source:b": 1},
        student_pass_counts={"source:a": 1},
        score_threshold=0.5,
        seed=0,
        no_cot_enable_thinking_strategy="none",
    )
    fill_rows = builder.build_reasoning_rows(strategy="solvable-student-else-teacher", **common)
    hard_rows = builder.build_reasoning_rows(strategy="unsolvable-teacher-only", **common)

    used_fill = builder.used_prompt_tags(fill_rows)
    used_hard = builder.used_prompt_tags(hard_rows)
    assert used_fill == {("source:a", "student"), ("source:b", "teacher")}
    assert used_hard == {("source:b", "teacher")}

    lsh = builder.build_prompt_index(
        used_fill | used_hard, teacher_best, student_best, num_hashes=64, bands=16
    )
    sft0_rows = [
        _sft0_parquet_row("Question a", "Old answer"),
        _sft0_parquet_row("Question b", "Old answer"),
        _sft0_parquet_row("Completely unrelated cooking conversation", "Sure"),
    ]
    matches = builder.match_sft0_rows(sft0_rows, lsh, threshold=0.85)
    assert matches == {
        0: {("source:a", "student")},
        1: {("source:b", "teacher")},
    }

    kept_fill, removed_fill = builder.filter_sft0_rows(sft0_rows, matches, used_fill)
    assert removed_fill == 2
    assert kept_fill == [sft0_rows[2]]

    kept_hard, removed_hard = builder.filter_sft0_rows(sft0_rows, matches, used_hard)
    assert removed_hard == 1
    assert kept_hard == [sft0_rows[0], sft0_rows[2]]


def test_multi_build_outputs_match_single_builds(tmp_path):
    teacher_file = tmp_path / "teacher.jsonl"
    student_file = tmp_path / "student.jsonl"
    _write_jsonl(
        teacher_file,
        [
            _record("a", 0, 1.0, "Teacher answer A", reasoning="Teacher CoT A"),
            _record("b", 0, 1.0, "Teacher answer B", reasoning="Teacher CoT B"),
        ],
    )
    _write_jsonl(
        student_file,
        [
            _record("a", 0, 1.0, "Student answer A"),
            _record("b", 0, 0.0, "Wrong"),
        ],
    )
    sft0_file = tmp_path / "sft0.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _sft0_parquet_row("Question a", "Old answer"),
                _sft0_parquet_row("Question b", "Old answer"),
                _sft0_parquet_row("Completely unrelated cooking conversation", "Sure"),
            ]
        ),
        sft0_file,
    )

    combos = [
        ("fill-standalone", "solvable-student-else-teacher", "standalone"),
        ("fill-mixed", "solvable-student-else-teacher", "mixed"),
        ("hard-standalone", "unsolvable-teacher-only", "standalone"),
        ("hard-mixed", "unsolvable-teacher-only", "mixed"),
    ]
    common = [
        "--teacher-results", str(teacher_file),
        "--student-results", str(student_file),
        "--max-attempts", "4",
    ]

    for name, strategy, mode in combos:
        argv = [
            *common,
            "--output-dir", str(tmp_path / "single" / name),
            "--strategy", strategy,
            "--dataset-mode", mode,
        ]
        if mode == "mixed":
            argv += ["--sft0", str(sft0_file)]
        _run_cli(argv)

    _run_cli(
        [
            *common,
            "--sft0", str(sft0_file),
            "--output-base", str(tmp_path / "multi"),
            *[
                arg
                for name, strategy, mode in combos
                for arg in ("--build", f"{name}:{strategy}:{mode}")
            ],
        ]
    )

    for name, _, _ in combos:
        single = pq.read_table(tmp_path / "single" / name / "train.parquet")
        multi = pq.read_table(tmp_path / "multi" / name / "train.parquet")
        assert single.equals(multi), name
        single_manifest = json.loads((tmp_path / "single" / name / "manifest.json").read_text())
        multi_manifest = json.loads((tmp_path / "multi" / name / "manifest.json").read_text())
        assert single_manifest == multi_manifest, name


def test_self_teacher_pool_preserves_student_teacher_strategy(tmp_path):
    teacher_file = tmp_path / "teacher.jsonl"
    self_teacher_file = tmp_path / "self-teacher.jsonl"
    student_file = tmp_path / "student.jsonl"
    _write_jsonl(
        teacher_file,
        [
            _record("solved", 0, 1.0, "External solved", reasoning="External CoT"),
            _record("self", 0, 1.0, "External answer", reasoning="External CoT"),
            _record("fallback", 0, 1.0, "Fallback answer", reasoning="Fallback CoT"),
        ],
    )
    _write_jsonl(
        self_teacher_file,
        [_record("self", 0, 1.0, "Self answer", reasoning="Self CoT")],
    )
    _write_jsonl(
        student_file,
        [
            _record("solved", 0, 1.0, "Student answer"),
            _record("self", 0, 0.0, "Wrong"),
            _record("fallback", 0, 0.0, "Wrong"),
        ],
    )

    output_dir = tmp_path / "out"
    _run_cli(
        [
            "--teacher-results",
            str(teacher_file),
            "--self-teacher-results",
            str(self_teacher_file),
            "--student-results",
            str(student_file),
            "--output-dir",
            str(output_dir),
            "--strategy",
            "solvable-student-else-teacher",
            "--no-cot-enable-thinking-strategy",
            "none",
        ]
    )

    rows = pq.read_table(output_dir / "train.parquet").to_pylist()
    blocks_by_prompt = {
        json.loads(row["metadata"])["prompt_id"]: _assistant_blocks(row)
        for row in rows
    }
    assert blocks_by_prompt["source:solved"] == [
        {"type": "response", "text": "Student answer"}
    ]
    assert blocks_by_prompt["source:self"] == [
        {"type": "thoughts", "text": "Self CoT"},
        {"type": "response", "text": "Self answer"},
    ]
    assert blocks_by_prompt["source:fallback"] == [
        {"type": "thoughts", "text": "Fallback CoT"},
        {"type": "response", "text": "Fallback answer"},
    ]

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["selected_self_teacher_traces"] == 1
    assert manifest["selected_external_teacher_traces"] == 2


def test_self_teacher_can_be_the_only_teacher_pool(tmp_path):
    self_teacher_file = tmp_path / "self-teacher.jsonl"
    _write_jsonl(
        self_teacher_file,
        [_record("self", 0, 1.0, "Self answer", reasoning="Self CoT")],
    )
    output_dir = tmp_path / "out"

    _run_cli(
        [
            "--self-teacher-results",
            str(self_teacher_file),
            "--output-dir",
            str(output_dir),
            "--strategy",
            "teacher-only",
            "--teacher-cot-probability",
            "1",
        ]
    )

    row = pq.read_table(output_dir / "train.parquet").to_pylist()[0]
    assert _assistant_blocks(row) == [
        {"type": "thoughts", "text": "Self CoT"},
        {"type": "response", "text": "Self answer"},
    ]


def test_skip_existing_builds(tmp_path):
    teacher_file = tmp_path / "teacher.jsonl"
    _write_jsonl(teacher_file, [_record("a", 0, 1.0, "Answer", reasoning="CoT")])
    argv = [
        "--teacher-results", str(teacher_file),
        "--output-base", str(tmp_path / "out"),
        "--build", "baseline:teacher-only:standalone",
        "--teacher-cot-probability", "0.5",
    ]

    _run_cli(argv)
    assert (tmp_path / "out" / "baseline" / "train.parquet").exists()

    with pytest.raises(FileExistsError):
        _run_cli(argv)

    _run_cli([*argv, "--skip-existing"])


def test_teacher_only_cot_probability_matches_capfilter_rate(tmp_path):
    teacher_file = tmp_path / "teacher.jsonl"
    student_file = tmp_path / "student.jsonl"
    _write_jsonl(
        teacher_file,
        [_record(f"p{i}", 0, 1.0, "Teacher answer", reasoning="Teacher CoT") for i in range(4)],
    )
    _write_jsonl(
        student_file,
        [_record(f"p{i}", 0, 1.0 if i < 2 else 0.0, "Student answer") for i in range(4)],
    )
    common = ["--teacher-results", str(teacher_file), "--student-results", str(student_file)]

    _run_cli(
        [
            *common,
            "--output-dir", str(tmp_path / "auto"),
            "--strategy", "teacher-only",
        ]
    )
    manifest = json.loads((tmp_path / "auto" / "manifest.json").read_text())
    assert manifest["teacher_cot_probability"] == 0.5  # 2 unsolvable of 4

    _run_cli(
        [
            *common,
            "--output-dir", str(tmp_path / "alpha"),
            "--strategy", "teacher-only",
            "--solvable-cot-alpha", "0.5",
        ]
    )
    manifest = json.loads((tmp_path / "alpha" / "manifest.json").read_text())
    assert manifest["teacher_cot_probability"] == 0.75  # (2 + 0.5*2) / 4
    assert manifest["solvable_cot_alpha"] == 0.5

    with pytest.raises(SystemExit):  # auto matching needs student results
        _run_cli(
            [
                "--teacher-results", str(teacher_file),
                "--output-dir", str(tmp_path / "no-student"),
                "--strategy", "teacher-only",
            ]
        )


def test_alpha_injects_teacher_cot_for_solvable_prompts():
    teacher_best = {
        "source:easy": _record("easy", 0, 1.0, "Teacher answer", reasoning="Teacher CoT"),
    }
    student = _record("easy", 0, 1.0, "Student answer")
    common = dict(
        teacher_best=teacher_best,
        student_best={"source:easy": student},
        student_attempt_counts={"source:easy": 1},
        student_pass_counts={"source:easy": 1},
        score_threshold=0.5,
        seed=0,
        no_cot_enable_thinking_strategy="none",
    )

    rows = builder.build_reasoning_rows(strategy="solvable-student-else-teacher", **common)
    assert json.loads(rows[0]["metadata"])["origin"] == "student"

    rows = builder.build_reasoning_rows(
        strategy="solvable-student-else-teacher", solvable_cot_alpha=1.0, **common
    )
    assert json.loads(rows[0]["metadata"])["origin"] == "teacher"
    assert _assistant_blocks(rows[0]) == [
        {"type": "thoughts", "text": "Teacher CoT"},
        {"type": "response", "text": "Teacher answer"},
    ]

    with pytest.raises(ValueError):
        builder.build_reasoning_rows(
            strategy="unsolvable-teacher-only", solvable_cot_alpha=0.1, **common
        )


def test_alpha_rejected_for_unsolvable_teacher_only_builds(tmp_path):
    teacher_file = tmp_path / "teacher.jsonl"
    student_file = tmp_path / "student.jsonl"
    _write_jsonl(teacher_file, [_record("a", 0, 1.0, "Answer", reasoning="CoT")])
    _write_jsonl(student_file, [_record("a", 0, 0.0, "Wrong")])

    with pytest.raises(SystemExit):
        _run_cli(
            [
                "--teacher-results", str(teacher_file),
                "--student-results", str(student_file),
                "--output-dir", str(tmp_path / "hard"),
                "--strategy", "unsolvable-teacher-only",
                "--solvable-cot-alpha", "0.1",
            ]
        )
