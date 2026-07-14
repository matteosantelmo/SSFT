#!/bin/bash
#SBATCH --job-name=regrade
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#
# Re-grade an existing results.jsonl with the current verifiers (CPU-only).
#
#   sbatch scripts/generation/regrade.sbatch /path/to/results.jsonl [output.jsonl]
#
# or via env vars: INPUT_JSONL=... [OUTPUT_JSONL=...] sbatch scripts/generation/regrade.sbatch
#
# The output defaults to <input stem>.regraded.jsonl next to the input. The
# input is never overwritten, and an interrupted job can simply be resubmitted:
# already re-graded ids are skipped.

set -euo pipefail

REPO="/iopsstor/scratch/cscs/msantelmo/SSFT"
cd "$REPO"
source "$REPO/.venv/bin/activate"

INPUT_JSONL="/users/msantelmo/scratch/SSFT/outputs/teacher_qwen3.6-27b/code/results.jsonl"
OUTPUT_JSONL="/users/msantelmo/scratch/SSFT/outputs/teacher_qwen3.6-27b/code/results_regraded.jsonl"
VERIFY_CONCURRENCY=32

# Code verifiers (taco/apps/codeforces/...) run in the Kubernetes sandbox when
# this is set; unset it to fall back to local prime_code execution.
export KUBERNETES_SANDBOX_URL="${KUBERNETES_SANDBOX_URL:-https://sandbox-dev.swissai.svc.cscs.ch}"

out_arg=()
[[ -n "$OUTPUT_JSONL" ]] && out_arg=(--output "$OUTPUT_JSONL")

python "$REPO/src/regrade.py" \
  --input "$INPUT_JSONL" \
  --verify-concurrency "$VERIFY_CONCURRENCY" \
  "${out_arg[@]}"
