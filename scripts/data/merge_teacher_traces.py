#!/usr/bin/env python3
"""Merge teacher trace results from multiple teacher models into one results file.

Every positively-scored, complete (finish_reason != "length"), non-empty attempt
from every teacher model is considered, and the merged file keeps exactly one
record per prompt. The winner is drawn from the models whose best attempt reaches
the highest (score, has-reasoning) tier for that prompt, picking the model with
the fewest selections so far, which keeps per-model contributions roughly
balanced wherever several models solve the same prompt equally well. Prompts
nobody solved above the threshold are dropped entirely.

Record ids are verified to be consistent across models (same id -> same prompt).
A record whose prompt disagrees with the id's canonical prompt is kept under a
rewritten id (`<prompt_id>~<prompt_hash>#<repeat>`) so it can never be joined
against the wrong prompt downstream; such conflicts are counted in the manifest.

Output records keep the input schema (so build_reasoning_sft_dataset.py consumes
the merged file directly) with reasoning/response normalized and a
`teacher_model` field naming the model each record came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reasoning_sft_dataset import is_truncated, log, normalized_generation, prompt_id


@dataclass(frozen=True)
class Candidate:
    """Best eligible attempt of one model for one prompt, addressed by file offset."""

    score: float
    has_reasoning: int
    neg_repeat: int
    file_index: int
    offset: int

    @property
    def tier(self) -> tuple[float, int]:
        return (self.score, self.has_reasoning)

    @property
    def rank(self) -> tuple[float, int, int]:
        return (self.score, self.has_reasoning, self.neg_repeat)


def prompt_digest(prompt: Any) -> str:
    """Deterministic short digest of a prompt message list."""
    payload = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def scan_model_files(
    model: str,
    files: list[tuple[int, Path]],
    *,
    score_threshold: float,
    candidates: dict[str, dict[str, Candidate]],
    canonical_prompts: dict[str, str],
    stats: dict[str, Any],
) -> None:
    """Stream one model's result files, keeping its best eligible attempt per prompt."""
    model_stats = stats["models"][model]
    for file_index, path in files:
        file_size = path.stat().st_size
        log(f"[scan] {model}: {path} ({file_size / 2**30:.2f} GiB)")
        progress = tqdm(
            total=file_size,
            desc=f"[scan] {model}/{path.parent.name}/{path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        )
        offset = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line_offset = offset
                offset += len(raw_line)
                if line_number % 1024 == 0:
                    progress.update(offset - progress.n)
                if not raw_line.strip():
                    continue
                try:
                    raw_record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if "id" not in raw_record:
                    raise ValueError(f"{path}:{line_number}: missing id")
                record = normalized_generation(raw_record)
                pid = prompt_id(str(record["id"]))
                model_stats["attempts"] += 1

                score = record.get("score")
                if score is None or float(score) <= score_threshold:
                    model_stats["discarded_low_score"] += 1
                    continue
                if not (record.get("response") or "").strip():
                    model_stats["discarded_empty_response"] += 1
                    continue
                if is_truncated(record):
                    model_stats["discarded_truncated"] += 1
                    continue

                digest = prompt_digest(record.get("prompt"))
                canonical = canonical_prompts.setdefault(pid, digest)
                key = pid
                if digest != canonical:
                    key = f"{pid}~{digest}"
                    model_stats["id_prompt_conflicts"] += 1

                candidate = Candidate(
                    score=float(score),
                    has_reasoning=1 if record.get("reasoning") else 0,
                    neg_repeat=-int(record.get("repeat_idx") or 0),
                    file_index=file_index,
                    offset=line_offset,
                )
                current = candidates[key].get(model)
                if current is None or candidate.rank > current.rank:
                    candidates[key][model] = candidate
            progress.update(file_size - progress.n)
            progress.close()


def assign_prompts(
    candidates: dict[str, dict[str, Candidate]],
) -> tuple[dict[str, tuple[str, Candidate]], dict[str, int]]:
    """Pick one model per prompt: best (score, has-reasoning) tier, least-used model first."""
    selected_counts: dict[str, int] = defaultdict(int)
    assignment: dict[str, tuple[str, Candidate]] = {}
    for key in tqdm(sorted(candidates), desc="[assign] prompts", unit="prompt", dynamic_ncols=True):
        entries = candidates[key]
        top_tier = max(candidate.tier for candidate in entries.values())
        tier_models = [model for model, candidate in entries.items() if candidate.tier == top_tier]
        model = min(tier_models, key=lambda name: (selected_counts[name], name))
        selected_counts[model] += 1
        assignment[key] = (model, entries[model])
    return assignment, dict(selected_counts)


def write_merged(
    assignment: dict[str, tuple[str, Candidate]],
    file_paths: list[Path],
    file_models: list[str],
    output: Path,
) -> int:
    """Re-read the selected records file by file and write the merged JSONL."""
    by_file: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for key, (model, candidate) in assignment.items():
        by_file[candidate.file_index].append((candidate.offset, key, model))

    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for file_index in sorted(by_file):
            path = file_paths[file_index]
            picks = sorted(by_file[file_index])
            desc = f"[write] {file_models[file_index]}/{path.parent.name}/{path.name}"
            with path.open("rb") as handle:
                for offset, key, model in tqdm(picks, desc=desc, unit="rec", dynamic_ncols=True):
                    handle.seek(offset)
                    record = normalized_generation(json.loads(handle.readline()))
                    if "~" in key:
                        repeat = int(record.get("repeat_idx") or 0)
                        record["id"] = f"{key}#{repeat}"
                    record["teacher_model"] = model
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
    return written


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        metavar="NAME=FILE[,FILE...]",
        help=(
            "Repeatable teacher spec: a model name and its result JSONL file(s) for one "
            "domain. List several files for the same model (e.g. a live run plus a "
            "recovered run) to treat them as one candidate pool."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="merged results.jsonl to write")
    parser.add_argument("--score-threshold", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")

    teachers: dict[str, list[Path]] = {}
    for spec in args.teacher:
        name, separator, files_part = spec.partition("=")
        if not separator or not name or not files_part:
            parser.error(f"invalid --teacher spec {spec!r}; expected NAME=FILE[,FILE...]")
        if name in teachers:
            parser.error(f"duplicate --teacher name {name!r}")
        files = [Path(part) for part in files_part.split(",")]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            parser.error(f"--teacher {name}: missing files: {', '.join(missing)}")
        teachers[name] = files
    args.teachers = teachers

    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    return args


def main() -> None:
    """Merge the teacher runs and write the output plus a manifest."""
    args = parse_args()
    teachers: dict[str, list[Path]] = args.teachers

    file_paths: list[Path] = []
    file_models: list[str] = []
    files_by_model: dict[str, list[tuple[int, Path]]] = {}
    for model in sorted(teachers):
        indexed = []
        for path in teachers[model]:
            indexed.append((len(file_paths), path))
            file_paths.append(path)
            file_models.append(model)
        files_by_model[model] = indexed

    stats: dict[str, Any] = {
        "score_threshold": args.score_threshold,
        "models": {
            model: {
                "files": [str(path) for path in teachers[model]],
                "attempts": 0,
                "discarded_low_score": 0,
                "discarded_empty_response": 0,
                "discarded_truncated": 0,
                "id_prompt_conflicts": 0,
            }
            for model in sorted(teachers)
        },
    }

    candidates: dict[str, dict[str, Candidate]] = defaultdict(dict)
    canonical_prompts: dict[str, str] = {}
    for model in sorted(teachers):
        scan_model_files(
            model,
            files_by_model[model],
            score_threshold=args.score_threshold,
            candidates=candidates,
            canonical_prompts=canonical_prompts,
            stats=stats,
        )

    assignment, selected_counts = assign_prompts(candidates)
    for model in stats["models"]:
        stats["models"][model]["selected"] = selected_counts.get(model, 0)
        conflicts = stats["models"][model]["id_prompt_conflicts"]
        if conflicts:
            log(f"[warn] {model}: {conflicts:,} records had a prompt conflicting with their id")

    written = write_merged(assignment, file_paths, file_models, args.output)
    stats["prompts_seen"] = len(canonical_prompts)
    stats["prompts_selected"] = written
    with (args.output.parent / "merge_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    log(
        f"[done] {written:,} merged records from {len(canonical_prompts):,} prompts seen; "
        "per-model: "
        + ", ".join(f"{model}={selected_counts.get(model, 0):,}" for model in sorted(teachers))
    )
    print(json.dumps({model: selected_counts.get(model, 0) for model in sorted(teachers)}, indent=2))


if __name__ == "__main__":
    main()
