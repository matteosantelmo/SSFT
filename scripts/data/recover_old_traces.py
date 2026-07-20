#!/usr/bin/env python3
"""Recover old teacher traces into the current pipeline's results.jsonl format.

Old verified generations (e.g. Kimi-K2.6 / gemma-4 on DeepMath) live as a
``datasets.Dataset`` on disk with ``problem`` / ``teacher_solution`` /
``teacher_thinking_trajectory`` / ``teacher_score`` columns. This script aligns
them with a *reference* results.jsonl produced by src/generate.py and emits one
record per unique reference problem, in the same schema:

  - the old dataset has a verified solution for that problem  ->  the record
    carries it (response = teacher_solution, reasoning = thinking trajectory,
    score = teacher_score);
  - it does not  ->  empty response, score 0.0.

The output therefore contains ALL AND ONLY the problems of the reference run,
so per-problem mixing across teachers is a straight id-for-id join. Matching is
by problem text (reference ``extra_info.question`` vs old ``problem``) because
the old ``source_index`` does not align with the reference ``extra_info.index``.

Identity fields (id, prompt, ground_truth, extra_info, ...) are copied from the
first reference record of each problem; the id keeps the pipeline shape with
attempt suffix ``#0``. A ``teacher_model`` field records provenance.

Needs ``datasets`` (not in the repo venv):

  uv run --with datasets python scripts/data/recover_old_traces.py \
    --old-dataset .../1-raw-generations/deepmath/moonshotai_Kimi-K2.6 \
    --reference outputs/teacher_qwen3.6-27b/math/results_regraded.jsonl \
    --output outputs/teacher_Kimi-K2.6-recovered/math/results_regraded.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import datasets


def load_old_traces(path: str, min_score: float) -> dict[str, dict]:
    """Map problem text -> verified trace record from the old dataset.

    Keeps the first occurrence of each problem (the dataset has ~1k duplicate
    problems; all rows are verified-correct, so any occurrence is fine).
    """
    ds = datasets.load_from_disk(path)
    traces: dict[str, dict] = {}
    kept = 0
    for problem, thinking, solution, score, model in zip(
        ds["problem"],
        ds["teacher_thinking_trajectory"],
        ds["teacher_solution"],
        ds["teacher_score"],
        ds["teacher_model"],
    ):
        if score is None or score < min_score:
            continue
        kept += 1
        traces.setdefault(problem.strip(), {
            "reasoning": thinking or None,
            "response": solution,
            "score": float(score),
            "teacher_model": model,
        })
    print(f"[old] {path}: {len(ds)} rows, {kept} with score >= {min_score}, "
          f"{len(traces)} unique problems")
    return traces


def make_record(ref: dict, trace: dict | None) -> dict:
    """Build one pipeline-format record for a reference problem.

    Identity fields come from the reference record; generation fields come from
    the old trace, or are empty/0 when the old dataset lacks this problem.
    """
    return {
        "id": ref["id"].rsplit("#", 1)[0] + "#0",
        "data_source": ref["data_source"],
        "ability": ref.get("ability"),
        "repeat_idx": 0,
        "seed": None,
        "prompt": ref["prompt"],
        "ground_truth": ref["ground_truth"],
        "extra_info": ref.get("extra_info"),
        "score": trace["score"] if trace else 0.0,
        "response": trace["response"] if trace else "",
        "reasoning": trace["reasoning"] if trace else None,
        "missing_reasoning": (trace["reasoning"] is None) if trace else None,
        "finish_reason": "stop" if trace else None,
        "completion_tokens": None,
        "verify_seconds": None,
        "teacher_model": trace["teacher_model"] if trace else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--old-dataset", required=True,
                   help="datasets.Dataset dir with the old verified traces.")
    p.add_argument("--reference", required=True,
                   help="results.jsonl of the current pipeline defining the problem set.")
    p.add_argument("--output", required=True, help="Output jsonl path.")
    p.add_argument("--min-score", type=float, default=1.0,
                   help="Keep old traces with teacher_score >= this (default 1.0).")
    args = p.parse_args()

    traces = load_old_traces(args.old_dataset, args.min_score)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    seen: set[str] = set()
    matched = 0
    with open(args.reference, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ref = json.loads(line)
            prompt_id = ref["id"].rsplit("#", 1)[0]
            if prompt_id in seen:  # reference holds one record per attempt
                continue
            seen.add(prompt_id)
            question = (ref.get("extra_info") or {}).get("question", "")
            trace = traces.get(question.strip())
            matched += trace is not None
            fout.write(json.dumps(make_record(ref, trace), ensure_ascii=False) + "\n")

    print(f"[done] {args.output}: {len(seen)} problems, "
          f"{matched} with recovered traces (score from old run), "
          f"{len(seen) - matched} empty (score 0.0)")


if __name__ == "__main__":
    main()
