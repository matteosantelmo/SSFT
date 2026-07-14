#!/usr/bin/env python3
"""Re-grade an existing results.jsonl from the generation pipeline.

Re-runs verification (src/verifier.py) on the stored ``response`` of every
record — useful after verifier fixes, e.g. stripping unparsed reasoning that
precedes a bare close-thinking token (Qwen3-style outputs). The input file is
never modified: records are re-scored concurrently and streamed to a new jsonl
(default: ``<input stem>.regraded.jsonl`` next to the input). The output is
resumable — ids already present in it are skipped on restart.

Every record is carried through unchanged except: ``score`` (plus any extra
keys the verifier returns) is recomputed, the previous score is preserved in
``prev_score``, and ``verify_seconds`` is refreshed. Records whose generation
failed (``response`` is null) pass through ungraded.

Launch on a compute node via scripts/generation/regrade.sbatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from verifier import verify


def load_records(path: str) -> dict[str, dict]:
    """Read records keyed by id (last occurrence wins), skipping bad lines."""
    records: dict[str, dict] = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # half-written trailing line from a previous interruption; ignore
                continue
            if "id" in rec:
                records[rec["id"]] = rec
    return records


def regrade(rec: dict) -> dict:
    """Re-verify one record; returns the updated copy (never raises)."""
    out = dict(rec)
    out["prev_score"] = rec.get("score")
    if rec.get("response") is None:
        return out  # generation failed; nothing to grade
    out.pop("error", None)  # superseded by the fresh verification below
    t0 = time.perf_counter()
    result = verify(
        rec.get("data_source"), rec["response"],
        rec.get("ground_truth"), rec.get("extra_info"),
    )
    out.update(result)
    out["verify_seconds"] = round(time.perf_counter() - t0, 4)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-grade a results.jsonl produced by the generation pipeline.")
    p.add_argument("--input", required=True, help="results.jsonl to re-grade (read-only).")
    p.add_argument("--output", default=None,
                   help="Output jsonl; default <input stem>.regraded.jsonl next to the input.")
    p.add_argument("--verify-concurrency", type=int, default=32, help="Max concurrent verifications.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(
        args.output or os.path.splitext(in_path)[0] + ".regraded.jsonl"
    )
    if out_path == in_path:
        sys.exit("[fatal] output path equals the input; refusing to overwrite it.")

    records = load_records(in_path)
    if not records:
        sys.exit(f"[fatal] no valid records found in {in_path}")
    done = set(load_records(out_path))
    pending = [rec for rid, rec in records.items() if rid not in done]
    print(f"[input] {in_path}: {len(records)} records"
          + (f"; resuming, {len(done)} already re-graded" if done else ""))
    if not pending:
        print("[done] nothing to do — all records already re-graded.")
        return

    changed = 0
    with open(out_path, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.verify_concurrency) as pool:
        futures = [pool.submit(regrade, rec) for rec in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="regrade", unit="sample"):
            rec = fut.result()
            if rec.get("score") != rec.get("prev_score"):
                changed += 1
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    print(f"[done] re-graded {len(pending)} records -> {out_path} ({changed} scores changed)")


if __name__ == "__main__":
    main()
