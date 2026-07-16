#!/bin/bash

set -euo pipefail

## SFT 0
# TODO: create reproducible data generation script for SFT-0 data
# ...

# Create the train-time evals data based on the configs (used both for SFT and RLVR)
python verl_rl/apertus/data_preprocess.py --config scripts/data/data_config/rl_0.yaml
python verl_rl/apertus/data_preprocess.py --config scripts/data/data_config/evals-small_nothink_nodisplay.yaml
python verl_rl/apertus/data_preprocess.py --config scripts/data/data_config/evals-full_nothink_nodisplay.yaml
python verl_rl/apertus/data_preprocess.py --config scripts/data/data_config/evals-small_think_nodisplay.yaml

python scripts/data/convert_rl_eval_to_sft.py ./data/evals-small_nothink_nodisplay/val.parquet ./data/sft_0/test.parquet
python scripts/data/convert_rl_eval_to_sft.py ./data/evals-full_nothink_nodisplay/val.parquet ./data/sft_0/test_full.parquet
python scripts/data/convert_rl_eval_to_sft.py ./data/evals-small_think_nodisplay/val.parquet ./data/sft_1/test_think.parquet


# Verifiable Data (for RL-0 and cap-filter+SFT)
python verl_rl/apertus/data_preprocess.py --config  scripts/data/data_config/capfilter_code.yaml
python verl_rl/apertus/data_preprocess.py --config  scripts/data/data_config/capfilter_if.yaml
python verl_rl/apertus/data_preprocess.py --config  scripts/data/data_config/capfilter_math.yaml

# Merge Teacher-CoTs and Model-Generated Responses on Verifiable Dataset for SFT
# Strategies: teacher-baseline = all-teacher with 50/50 CoT kept;
#             cap-filter-fill  = self-success where solvable (pass@8=1), teacher CoT otherwise;
#             cap-filter-hard  = solvable dropped, teacher CoT for unsolvable only.
# Each comes standalone and merged with SFT-0 (mixed, LSH prompt-dedup, reasoning rows win).
# No-CoT rows get enable_thinking via a 50/50 coin flip (--no-cot-enable-thinking-strategy random).
SFT0_TRAIN=data/sft_0/train.parquet
MAX_ATTEMPTS=8
DOMAINS=(math code if)
TEACHER_RUN=outputs/teacher_qwen3.6-27b
TEACHER_ARGS=()
for domain in "${DOMAINS[@]}"; do
    TEACHER_ARGS+=(--teacher-results "$TEACHER_RUN/$domain/results_regraded.jsonl")
done

# teacher-baseline (standalone + merged with SFT-0)
python scripts/data/build_reasoning_sft_dataset.py \
    "${TEACHER_ARGS[@]}" \
    --output-dir data/sft_1/teacher-baseline \
    --strategy teacher-only-5050 \
    --max-attempts "$MAX_ATTEMPTS" \
    --no-cot-enable-thinking-strategy random

python scripts/data/build_reasoning_sft_dataset.py \
    "${TEACHER_ARGS[@]}" \
    --sft0 "$SFT0_TRAIN" \
    --output-dir data/sft_1/sft0+teacher-baseline \
    --strategy teacher-only-5050 \
    --dataset-mode mixed \
    --max-attempts "$MAX_ATTEMPTS" \
    --no-cot-enable-thinking-strategy random

# cap-filter datasets, one set per student model (append new runs here to repeat)
declare -A STUDENT_RESULTS=(
    [apertus-8b-2509-sft0-step11776]="outputs/capability-filtering/Apertus-8B-2509__sft_0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260710-095910__global_step_11776"
    [apertus-1p5_8b-sft0-step11264]="outputs/capability-filtering/apertus-1p5_8b_seq_len_256k_7000__sft_0_lr5e-5-ratio03__global_step_11264"
)

for model in "${!STUDENT_RESULTS[@]}"; do
    student="${STUDENT_RESULTS[$model]}"
    # Student: one results.jsonl per domain-specific folder of the run.
    student_args=()
    for domain in "${DOMAINS[@]}"; do
        student_args+=(--student-results "$student/$domain/results.jsonl")
    done

    # cap-filter-fill (standalone + merged with SFT-0)
    python scripts/data/build_reasoning_sft_dataset.py \
        "${TEACHER_ARGS[@]}" \
        "${student_args[@]}" \
        --output-dir "data/sft_1/cap-filter-fill-${model}" \
        --strategy solvable-student-else-teacher \
        --max-attempts "$MAX_ATTEMPTS" \
        --no-cot-enable-thinking-strategy random

    python scripts/data/build_reasoning_sft_dataset.py \
        "${TEACHER_ARGS[@]}" \
        "${student_args[@]}" \
        --sft0 "$SFT0_TRAIN" \
        --output-dir "data/sft_1/cap-filter-fill-${model}-mix-sft0" \
        --strategy solvable-student-else-teacher \
        --dataset-mode mixed \
        --max-attempts "$MAX_ATTEMPTS" \
        --no-cot-enable-thinking-strategy random

    # cap-filter-hard (standalone + merged with SFT-0)
    python scripts/data/build_reasoning_sft_dataset.py \
        "${TEACHER_ARGS[@]}" \
        "${student_args[@]}" \
        --output-dir "data/sft_1/cap-filter-hard-${model}" \
        --strategy unsolvable-teacher-only \
        --max-attempts "$MAX_ATTEMPTS" \
        --no-cot-enable-thinking-strategy random

    python scripts/data/build_reasoning_sft_dataset.py \
        "${TEACHER_ARGS[@]}" \
        "${student_args[@]}" \
        --sft0 "$SFT0_TRAIN" \
        --output-dir "data/sft_1/cap-filter-hard-${model}-mix-sft0" \
        --strategy unsolvable-teacher-only \
        --dataset-mode mixed \
        --max-attempts "$MAX_ATTEMPTS" \
        --no-cot-enable-thinking-strategy random
done