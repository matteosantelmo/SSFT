#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SFT_SCRIPT="${SCRIPT_DIR}/sft_1.sh"

TRAIN_DATASET_ROOT="/iopsstor/scratch/cscs/msantelmo/SSFT/data/sft_1"

# MODEL_PATH="/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/apertus-1p5_8b_seq_len_256k_7000_steps"
# TOKENIZER_PATH="/iopsstor/scratch/cscs/msantelmo/tokenizers/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
MODEL_PATH="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-2509"
TOKENIZER_PATH="/iopsstor/scratch/cscs/msantelmo/tokenizers/apertus_2509_text_only_aligned_v3"


LEARNING_RATES=("5e-5" "1e-5") # Add "5e-6" here to include it in the sweep.
DATASET_NAMES=(
    # "sft0+cap-filter-fill-apertus-1p5_8b-sft0-step11264"
    # "sft0+cap-filter-hard-apertus-1p5_8b-sft0-step11264"
    # "sft0+teacher-baseline-apertus-1p5_8b-sft0-step11264"

    "sft0+teacher-baseline-apertus-8b-2509-sft0-step11776"
    "sft0+cap-filter-hard-apertus-8b-2509-sft0-step11776"
    "sft0+cap-filter-fill-apertus-8b-2509-sft0-step11776"
)

if [[ ! -x "$SFT_SCRIPT" ]]; then
    echo "Error: SFT launcher is not executable: $SFT_SCRIPT" >&2
    exit 1
fi

num_jobs=$((${#LEARNING_RATES[@]} * ${#DATASET_NAMES[@]}))
echo "Submitting ${num_jobs} SFT jobs..."

for dataset_name in "${DATASET_NAMES[@]}"; do
    dataset_path="${TRAIN_DATASET_ROOT}/${dataset_name}"

    for learning_rate in "${LEARNING_RATES[@]}"; do
        echo "Submitting dataset=${dataset_name} learning_rate=${learning_rate}"

        MODEL_PATH="$MODEL_PATH" \
        TOKENIZER_PATH="$TOKENIZER_PATH" \
        DATASET_PATH="$dataset_path" \
        LEARNING_RATE="$learning_rate" \
            "$SFT_SCRIPT"

        sleep 10
    done
done

echo "Submitted all ${num_jobs} SFT jobs."
