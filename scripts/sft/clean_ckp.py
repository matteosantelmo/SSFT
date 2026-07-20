#!/usr/bin/env python3

"""Remove resumable state from intermediate SFT checkpoints."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


CHECKPOINT_PATTERN = re.compile(r"global_step_(\d+)$")
HF_DIRECTORY = "huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove the resumable FSDP state from every intermediate SFT "
            "checkpoint while preserving its huggingface/ directory. The "
            "highest-numbered checkpoint in each output directory is left intact."
        ),
        epilog=(
            "Example:\n"
            "  scripts/sft/clean_ckp.sh outputs/run-a outputs/run-b"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dirs",
        metavar="OUTPUT_DIR",
        nargs="+",
        type=Path,
        help="SFT output directory containing global_step_<N> checkpoints.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print what would be removed without changing any files.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int | None:
    """Return the numeric step for a valid checkpoint directory name."""
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def find_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Find real checkpoint directories ordered by their numeric step."""
    checkpoints = []
    for candidate in output_dir.iterdir():
        step = checkpoint_step(candidate)
        if step is None or candidate.is_symlink() or not candidate.is_dir():
            continue
        checkpoints.append((step, candidate))
    return sorted(checkpoints, key=lambda item: (item[0], item[1].name))


def resolve_output_dirs(paths: list[Path]) -> list[Path]:
    """Resolve and validate all user-provided output directories."""
    output_dirs = []
    seen = set()

    for path in paths:
        try:
            output_dir = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"output directory does not exist: {path}") from exc

        if not output_dir.is_dir():
            raise ValueError(f"not an output directory: {output_dir}")
        if output_dir == Path("/"):
            raise ValueError("refusing to use / as an output directory")
        if output_dir not in seen:
            output_dirs.append(output_dir)
            seen.add(output_dir)

    return output_dirs


def plan_cleanup(output_dirs: list[Path]) -> list[Path]:
    """Validate every run and return its intermediate checkpoints."""
    checkpoints_to_clean = []

    for output_dir in output_dirs:
        checkpoints = find_checkpoints(output_dir)
        if not checkpoints:
            print(
                f"warning: no global_step_<N> checkpoints found in {output_dir}",
                file=sys.stderr,
            )
            continue

        latest_checkpoint = checkpoints[-1][1]
        print(f"Keeping latest checkpoint intact: {latest_checkpoint}")

        for _, checkpoint in checkpoints[:-1]:
            hf_dir = checkpoint / HF_DIRECTORY
            if not hf_dir.is_dir():
                raise ValueError(
                    f"refusing to clean checkpoint without huggingface/: {checkpoint}"
                )
            checkpoints_to_clean.append(checkpoint)

    return checkpoints_to_clean


def remove_entry(entry: Path) -> None:
    """Remove one checkpoint-root entry without following symlinks."""
    if entry.is_symlink() or not entry.is_dir():
        entry.unlink()
    else:
        shutil.rmtree(entry)


def clean(checkpoints: list[Path], *, dry_run: bool) -> int:
    """Clean planned checkpoints and return the number of removed entries."""
    removed_entries = 0

    for checkpoint in checkpoints:
        print(f"Cleaning intermediate checkpoint: {checkpoint}")
        for entry in sorted(checkpoint.iterdir(), key=lambda path: path.name):
            if entry.name == HF_DIRECTORY:
                continue

            if dry_run:
                print(f"  would remove: {entry}")
            else:
                remove_entry(entry)
                print(f"  removed: {entry}")
            removed_entries += 1

    return removed_entries


def main() -> int:
    args = parse_args()
    try:
        output_dirs = resolve_output_dirs(args.output_dirs)
        checkpoints = plan_cleanup(output_dirs)
        removed_entries = clean(checkpoints, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    action = "would be removed" if args.dry_run else "removed"
    prefix = "Dry run complete" if args.dry_run else "Done"
    print(
        f"{prefix}: {len(checkpoints)} checkpoint(s), "
        f"{removed_entries} root entry/entries {action}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
