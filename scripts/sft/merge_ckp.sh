#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# Configuration
###############################################################################

# FSDP role checkpoint directory, e.g. .../global_step_260/actor.
# SOURCE_DIR="/iopsstor/scratch/cscs/msantelmo/SSFT/outputs/ssft/Apertus-8B-2509__sft_0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260710-095910/global_step_11776"
# TARGET_DIR="/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_0/Apertus-8B-2509__sft_0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260710-095910__global_step_11776"
# TOKENIZER_DIR="/iopsstor/scratch/cscs/msantelmo/checkpoints/Apertus-8B-Instruct-2509"

# SOURCE_DIR="/iopsstor/scratch/cscs/msantelmo/SSFT/outputs/ssft/apertus-1p5_8b_seq_len_256k_7000_steps__sft_0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260713-102218/global_step_11264"
# TARGET_DIR="/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_0/apertus-1p5_8b_seq_len_256k_7000_steps__sft_0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260713-102218__global_step_11264"
# TOKENIZER_DIR="/capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_mixthink_1606_480it"

# SOURCE_DIR="/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__cap-filter-fill-apertus-8b-2509-sft0-step11776-mix-sft0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-001239/global_step_12618"
# TOKENIZER_DIR="/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__cap-filter-fill-apertus-8b-2509-sft0-step11776-mix-sft0__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-001239/global_step_12618"
# TARGET_DIR="/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_1/Apertus-8B-2509__sft0-cap-filter-fill__20260716-001239__global_step_12618"

# SOURCE_DIR="/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft0+teacher-baseline__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-003107/global_step_12618"
# TOKENIZER_DIR="/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft0+teacher-baseline__sp2-lr5e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-003107/global_step_12618"
# TARGET_DIR="/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_1/Apertus-8B-2509__sft0-baseline__20260716-003107__global_step_12618"

# SOURCE_DIR=/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776__cap-filter-fill-apertus-8b-2509-sft0-step11776__sp2-lr1e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-212649/global_step_1010
# TOKENIZER_DIR=/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776__cap-filter-fill-apertus-8b-2509-sft0-step11776__sp2-lr1e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-212649/global_step_1010
# TARGET_DIR=/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_1/Apertus-8B-2509-sft_0-11776__cap-filter-fill__20260716-212649__global_step_1010

# SOURCE_DIR=/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776__cap-filter-hard-apertus-8b-2509-sft0-step11776__sp2-lr1e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-212621/global_step_344
# TOKENIZER_DIR=/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776__cap-filter-hard-apertus-8b-2509-sft0-step11776__sp2-lr1e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-212621/global_step_344
# TARGET_DIR=/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_1/Apertus-8B-2509-sft_0-11776__cap-filter-hard__20260716-212621__global_step_344

SOURCE_DIR=/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776__teacher-baseline__sp2-lr1e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-214546/global_step_1010
TOKENIZER_DIR=/users/msantelmo/scratch/SSFT/outputs/sft_1/Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776__teacher-baseline__sp2-lr1e-5-bs512-warmuplinear-lr_warmup_steps_ratio0.03__20260716-214546/global_step_1010
TARGET_DIR=/iopsstor/scratch/cscs/msantelmo/checkpoints/sft_1/Apertus-8B-2509-sft_0-11776__teacher__20260716-214546__global_step_1010


die() {
  echo "error: $*" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
verl_dir="${repo_root}/verl_rl"
cd "${verl_dir}"

source_dir="${SOURCE_DIR%/}"
target_dir="${TARGET_DIR%/}"
tokenizer_dir="${TOKENIZER_DIR%/}"

hf_dir="${source_dir}/huggingface"
fsdp_config="${source_dir}/fsdp_config.json"

[[ -d "${source_dir}" ]] || die "source directory does not exist: ${source_dir}"
[[ -f "${fsdp_config}" ]] || die "missing ${fsdp_config}"
[[ -d "${hf_dir}" ]] || die "missing HuggingFace config directory: ${hf_dir}"
[[ -f "${hf_dir}/config.json" ]] || die "missing ${hf_dir}/config.json"

world_size="$(
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["world_size"])' "${fsdp_config}"
)" || die "could not read world_size from ${fsdp_config}"
[[ "${world_size}" =~ ^[0-9]+$ && "${world_size}" -gt 0 ]] || die "invalid world_size: ${world_size}"

missing=0
for rank in $(seq 0 $((world_size - 1))); do
  shard="${source_dir}/model_world_size_${world_size}_rank_${rank}.pt"
  if [[ ! -f "${shard}" ]]; then
    echo "missing model shard: ${shard}" >&2
    missing=1
  fi
done
[[ "${missing}" -eq 0 ]] || die "source checkpoint is incomplete"

tokenizer_required=(tokenizer.json tokenizer_config.json special_tokens_map.json)
tokenizer_missing=()
for file in "${tokenizer_required[@]}"; do
  [[ -f "${hf_dir}/${file}" ]] || tokenizer_missing+=("${file}")
done

if [[ "${#tokenizer_missing[@]}" -gt 0 && -z "${tokenizer_dir}" ]]; then
  die "missing tokenizer files in ${hf_dir}: ${tokenizer_missing[*]}; pass TOKENIZER_DIR or set TOKENIZER_NAME_OR_PATH"
fi

if [[ -n "${tokenizer_dir}" ]]; then
  [[ -d "${tokenizer_dir}" ]] || die "tokenizer directory does not exist: ${tokenizer_dir}"

  for file in "${tokenizer_required[@]}"; do
    if [[ ! -f "${hf_dir}/${file}" ]]; then
      [[ -f "${tokenizer_dir}/${file}" ]] || die "missing tokenizer source file: ${tokenizer_dir}/${file}"
      cp "${tokenizer_dir}/${file}" "${hf_dir}/"
    fi
  done

  for file in chat_template.jinja audio_token_mapping.json vision_token_mapping.json; do
    if [[ -f "${tokenizer_dir}/${file}" && ! -f "${hf_dir}/${file}" ]]; then
      cp "${tokenizer_dir}/${file}" "${hf_dir}/"
    fi
  done
fi

pip install -e .

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${source_dir}" \
  --target_dir "${target_dir}"
