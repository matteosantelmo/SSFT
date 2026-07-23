import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts" / "data"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("build_reasoning_sft_dataset")
merger = _load("merge_teacher_traces")


def _record(prompt_id, repeat_idx, score, response, reasoning=None, *, prompt=None, finish="stop"):
    return {
        "id": f"source:{prompt_id}#{repeat_idx}",
        "repeat_idx": repeat_idx,
        "score": score,
        "prompt": [{"role": "user", "content": prompt or f"Question {prompt_id}"}],
        "response": response,
        "reasoning": reasoning,
        "finish_reason": finish,
        "data_source": "source",
    }


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _run_cli(argv):
    old_argv = sys.argv
    sys.argv = ["merge_teacher_traces.py", *argv]
    try:
        merger.main()
    finally:
        sys.argv = old_argv


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_selection_rules(tmp_path):
    _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _record("best", 0, 0.6, "A weak", reasoning="A CoT"),
            _record("low", 0, 0.4, "A low"),
            _record("trunc", 0, 1.0, "A cut", reasoning="A CoT", finish="length"),
            _record("cot", 0, 1.0, "A plain answer"),
        ],
    )
    _write_jsonl(
        tmp_path / "b.jsonl",
        [
            _record("best", 0, 1.0, "B strong", reasoning="B CoT"),
            _record("low", 0, 0.0, "B low"),
            _record("trunc", 0, 0.8, "B complete", reasoning="B CoT"),
            _record("cot", 0, 1.0, "B answer", reasoning="B CoT"),
        ],
    )
    output = tmp_path / "merged" / "results.jsonl"
    _run_cli(
        [
            "--teacher", f"a-model={tmp_path / 'a.jsonl'}",
            "--teacher", f"b-model={tmp_path / 'b.jsonl'}",
            "--output", str(output),
            "--score-threshold", "0.5",
        ]
    )

    by_pid = {builder.prompt_id(rec["id"]): rec for rec in _read_jsonl(output)}
    assert set(by_pid) == {"source:best", "source:trunc", "source:cot"}
    # highest score wins over balance
    assert by_pid["source:best"]["teacher_model"] == "b-model"
    # positively-scored but truncated attempts never win
    assert by_pid["source:trunc"]["teacher_model"] == "b-model"
    # same score: the attempt with reasoning wins
    assert by_pid["source:cot"]["teacher_model"] == "b-model"

    manifest = json.loads((output.parent / "merge_manifest.json").read_text())
    assert manifest["models"]["a-model"]["discarded_truncated"] == 1
    assert manifest["models"]["a-model"]["discarded_low_score"] == 1
    assert manifest["models"]["b-model"]["discarded_low_score"] == 1
    assert manifest["prompts_selected"] == 3

    # the merged file is directly consumable by the dataset builder
    best, attempt_counts, _ = builder.load_best_generations(
        [output], max_attempts=8, score_threshold=0.5, source_name="teacher"
    )
    assert set(best) == set(by_pid)
    assert all(count == 1 for count in attempt_counts.values())


def test_ties_are_balanced_across_models(tmp_path):
    records = [_record(f"t{i}", 0, 1.0, "Answer", reasoning="CoT") for i in range(6)]
    _write_jsonl(tmp_path / "a.jsonl", records)
    _write_jsonl(tmp_path / "b.jsonl", records)
    output = tmp_path / "merged" / "results.jsonl"
    _run_cli(
        [
            "--teacher", f"a-model={tmp_path / 'a.jsonl'}",
            "--teacher", f"b-model={tmp_path / 'b.jsonl'}",
            "--output", str(output),
        ]
    )

    counts = Counter(rec["teacher_model"] for rec in _read_jsonl(output))
    assert counts == {"a-model": 3, "b-model": 3}


def test_same_model_across_files_keeps_single_best(tmp_path):
    _write_jsonl(tmp_path / "run1.jsonl", [_record("p", 0, 0.7, "Okay", reasoning="CoT")])
    _write_jsonl(tmp_path / "run2.jsonl", [_record("p", 1, 1.0, "Better", reasoning="CoT")])
    output = tmp_path / "merged" / "results.jsonl"
    _run_cli(
        [
            "--teacher", f"a-model={tmp_path / 'run1.jsonl'},{tmp_path / 'run2.jsonl'}",
            "--output", str(output),
            "--score-threshold", "0.5",
        ]
    )

    records = _read_jsonl(output)
    assert len(records) == 1
    assert records[0]["id"] == "source:p#1"
    assert records[0]["response"] == "Better"


def test_conflicting_prompt_for_same_id_is_rekeyed(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [_record("x", 0, 1.0, "Answer A", prompt="Question one")])
    _write_jsonl(tmp_path / "b.jsonl", [_record("x", 0, 1.0, "Answer B", prompt="Different question")])
    output = tmp_path / "merged" / "results.jsonl"
    _run_cli(
        [
            "--teacher", f"a-model={tmp_path / 'a.jsonl'}",
            "--teacher", f"b-model={tmp_path / 'b.jsonl'}",
            "--output", str(output),
        ]
    )

    ids = sorted(rec["id"] for rec in _read_jsonl(output))
    assert len(ids) == 2
    assert "source:x#0" in ids
    rewritten = next(record_id for record_id in ids if record_id != "source:x#0")
    assert rewritten.startswith("source:x~") and rewritten.endswith("#0")

    manifest = json.loads((output.parent / "merge_manifest.json").read_text())
    assert manifest["models"]["b-model"]["id_prompt_conflicts"] == 1


def test_unseparated_reasoning_is_normalized_in_output(tmp_path):
    _write_jsonl(
        tmp_path / "a.jsonl",
        [_record("p", 0, 1.0, "Deep thoughts here.\n</think>\n\nFinal answer: 42.")],
    )
    output = tmp_path / "merged" / "results.jsonl"
    _run_cli(["--teacher", f"a-model={tmp_path / 'a.jsonl'}", "--output", str(output)])

    (record,) = _read_jsonl(output)
    assert record["reasoning"] == "Deep thoughts here."
    assert record["response"] == "Final answer: 42."


def test_refuses_to_overwrite_output(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [_record("p", 0, 1.0, "Answer")])
    output = tmp_path / "merged" / "results.jsonl"
    argv = ["--teacher", f"a-model={tmp_path / 'a.jsonl'}", "--output", str(output)]
    _run_cli(argv)
    with pytest.raises(SystemExit):
        _run_cli(argv)
