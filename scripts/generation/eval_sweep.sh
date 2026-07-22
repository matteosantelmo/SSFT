#!/usr/bin/env bash

# Submit one evaluation client/controller job per SFT run directory. Each
# controller evaluates that run's Hugging Face checkpoints sequentially, using
# one temporary serving job at a time. This deliberately avoids one idle client
# allocation per checkpoint.

set -euo pipefail

REPO="${REPO:-/iopsstor/scratch/cscs/msantelmo/SSFT}"
EVAL_SUITE=evals-small_nothink_nodisplay
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/outputs/evals/${EVAL_SUITE}}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/data/${EVAL_SUITE}/eval}"
EVALS=(
  aime2024
  aime2025
  math500
  gpqa_diamond
  gsm8k
  openai_humaneval
  ifeval
  ifbench
)

THINKING="${THINKING:-off}"
STOP_ON_FIRST_CORRECT=off
CORRECT_THRESHOLD=0.7
SEED="${SEED:-85}"
REPEATS=64
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=8192
CONCURRENCY="${CONCURRENCY:-512}"
VERIFY_CONCURRENCY="${VERIFY_CONCURRENCY:-64}"
START="${START:-0}"
END="${END:-}"

REPLICAS="${REPLICAS:-8}"
NODES_PER_REPLICA="${NODES_PER_REPLICA:-1}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
ENV_TOML="${ENV_TOML:-${REPO}/model_launch/src/swiss_ai_model_launch/assets/envs/sglang.toml}"
ROUTER_ARGS="${ROUTER_ARGS:-}"

PARTITION="${PARTITION:-normal}"
ACCOUNT="${ACCOUNT:-infra01}"
RESERVATION="${RESERVATION:-SD-69241-apertus-1-5-0}"
SERVING_TIME="${SERVING_TIME:-02:00:00}"
SWEEP_TIME="${SWEEP_TIME:-02:00:00}"
CLIENT_CPUS="${CLIENT_CPUS:-32}"
KUBERNETES_SANDBOX_URL="${KUBERNETES_SANDBOX_URL:-https://sandbox-dev.swissai.svc.cscs.ch}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${THINKING}" == "on" ]]; then
  SKIP_SPECIAL_TOKENS="${SKIP_SPECIAL_TOKENS:-false}"
else
  SKIP_SPECIAL_TOKENS="${SKIP_SPECIAL_TOKENS:-true}"
fi

usage() {
  cat <<'EOF'
Usage:
  eval_sweep.sh [--dry-run] SFT_OUTPUT_DIR [SFT_OUTPUT_DIR ...]
  eval_sweep.sh [--dry-run] --checkpoint HF_MODEL_DIR

Submit one evaluation controller job per SFT run. Within each controller job,
all global_step_<N>/huggingface checkpoints are served and evaluated
sequentially. Completed eval outputs are not submitted again.

With --checkpoint, evaluate one Hugging Face model directory directly instead
of discovering global_step_<N>/huggingface directories below an SFT run.

Examples:
  scripts/generation/eval_sweep.sh outputs/sft_1/run-a
  scripts/generation/eval_sweep.sh outputs/sft_1/run-a outputs/sft_1/run-b
  scripts/generation/eval_sweep.sh --dry-run outputs/sft_1/run-*
  scripts/generation/eval_sweep.sh --checkpoint checkpoints/baselines/model

Important environment overrides:
  OUTPUT_ROOT, EVAL_SUITE, THINKING, REPEATS, REPLICAS, SERVING_TIME,
  SWEEP_TIME, PARTITION, ACCOUNT, RESERVATION, DRY_RUN
EOF
}

die() {
  echo "[fatal] $*" >&2
  exit 1
}

enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

checkpoint_dirs() {
  local run_dir="$1"
  local candidate name
  local -a found=()

  # A direct Hugging Face model is a one-checkpoint sweep.
  if [[ -f "${run_dir}/config.json" ]]; then
    printf '%s\n' "${run_dir}"
    return 0
  fi

  shopt -s nullglob
  for candidate in "${run_dir}"/global_step_*; do
    [[ -d "${candidate}" && ! -L "${candidate}" ]] || continue
    name="${candidate##*/}"
    [[ "${name}" =~ ^global_step_[0-9]+$ ]] || continue
    found+=("${candidate}")
  done
  ((${#found[@]} > 0)) || return 0
  printf '%s\n' "${found[@]}" | sort -V
}

eval_output_dir() {
  local run_dir="$1" step="$2" eval_name="${3:-}"
  local stage run_name model_name
  stage="$(basename "$(dirname "${run_dir}")")"
  run_name="$(basename "${run_dir}")"
  if [[ -f "${run_dir}/config.json" ]]; then
    model_name="${run_name}__thinking-${THINKING}"
  else
    model_name="${run_name}__global_step_${step}__thinking-${THINKING}"
  fi
  printf '%s/%s/%s' "${OUTPUT_ROOT}" "${stage}" "${model_name}"
  [[ -z "${eval_name}" ]] || printf '/%s' "${eval_name}"
}

checkpoint_model_path() {
  local checkpoint="$1"
  if [[ -f "${checkpoint}/config.json" ]]; then
    printf '%s' "${checkpoint}"
  else
    printf '%s/huggingface' "${checkpoint}"
  fi
}

# A marker is authoritative for new runs. A non-empty legacy results file is
# also treated as already executed, so adopting this script never reruns old
# evaluations. The .eval_sweep_incomplete marker allows interrupted runs
# started by this script to resume instead of being mistaken for legacy output.
eval_is_done() {
  local eval_dir="$1"
  [[ -f "${eval_dir}/.complete" ]] || \
    [[ -s "${eval_dir}/results.jsonl" && ! -f "${eval_dir}/.eval_sweep_incomplete" ]]
}

checkpoint_has_pending_evals() {
  local run_dir="$1" step="$2" eval_name eval_dir
  for eval_name in "${EVALS[@]}"; do
    eval_dir="$(eval_output_dir "${run_dir}" "${step}" "${eval_name}")"
    eval_is_done "${eval_dir}" || return 0
  done
  return 1
}

validate_common_inputs() {
  local eval_name parquet
  [[ -d "${REPO}" ]] || die "repository does not exist: ${REPO}"
  for eval_name in "${EVALS[@]}"; do
    parquet="${EVAL_ROOT}/${eval_name}.parquet"
    [[ -f "${parquet}" ]] || die "missing eval parquet: ${parquet}"
  done
  case "${THINKING}" in on|off) ;; *) die "THINKING must be 'on' or 'off'" ;; esac
}

run_generate() {
  local job_id="$1" served_model_name="$2" parquet="$3" eval_dir="$4"
  local -a end_arg=() early_stop_arg=() extra_args=()
  [[ -z "${END}" ]] || end_arg=(--end "${END}")
  enabled "${STOP_ON_FIRST_CORRECT}" && early_stop_arg=(--stop-on-first-correct)
  [[ -z "${TOP_K:-}" ]] || extra_args+=(--top-k "${TOP_K}")
  [[ -z "${CHAT_TEMPLATE_KWARGS:-}" ]] || \
    extra_args+=(--chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
  [[ -z "${MAX_RETRIES:-}" ]] || extra_args+=(--max-retries "${MAX_RETRIES}")

  python "${REPO}/src/generate.py" \
    --job-id "${job_id}" \
    --served-model-name "${served_model_name}" \
    --input "${parquet}" \
    --output-dir "${eval_dir}" \
    --concurrency "${CONCURRENCY}" \
    --verify-concurrency "${VERIFY_CONCURRENCY}" \
    --repeats "${REPEATS}" \
    --correct-threshold "${CORRECT_THRESHOLD}" "${early_stop_arg[@]}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-tokens "${MAX_TOKENS}" \
    --enable-thinking "${THINKING}" \
    --skip-special-tokens "${SKIP_SPECIAL_TOKENS}" \
    --start "${START}" "${end_arg[@]}" "${extra_args[@]}"
}

worker_main() {
  local run_dir="$1"
  local checkpoint checkpoint_name step model_path checkpoint_output
  local eval_name eval_dir parquet submit_out job_id served_model_name
  local sglang_args client_rc=0 checkpoint_rc
  local current_serving_job=""
  local -a checkpoints=() reservation_args=() router_args_opt=()

  validate_common_inputs
  [[ -x "${REPO}/.venv/bin/sml" ]] || \
    die "sml not found in ${REPO}/.venv; run scripts/setup.sh first"
  # shellcheck disable=SC1091
  source "${REPO}/.venv/bin/activate"

  mapfile -t checkpoints < <(checkpoint_dirs "${run_dir}")
  ((${#checkpoints[@]} > 0)) || \
    die "no global_step_<N> checkpoints or Hugging Face model in ${run_dir}"

  [[ -z "${RESERVATION}" ]] || reservation_args=(--reservation "${RESERVATION}")
  [[ -z "${ROUTER_ARGS}" ]] || router_args_opt=(--router-args "${ROUTER_ARGS}")

  cleanup_serving() {
    if [[ -n "${current_serving_job}" ]]; then
      echo "[cleanup] cancelling serving job ${current_serving_job}"
      scancel "${current_serving_job}" || true
      current_serving_job=""
    fi
  }
  trap cleanup_serving EXIT
  trap 'cleanup_serving; exit 130' INT TERM

  echo "[sweep] run=${run_dir}; checkpoints=${#checkpoints[@]}; evals=${#EVALS[@]}"
  for checkpoint in "${checkpoints[@]}"; do
    checkpoint_name="${checkpoint##*/}"
    step="${checkpoint_name#global_step_}"
    model_path="$(checkpoint_model_path "${checkpoint}")"
    checkpoint_output="$(eval_output_dir "${run_dir}" "${step}")"

    if ! checkpoint_has_pending_evals "${run_dir}" "${step}"; then
      echo "[skip] step ${step}: every evaluation has already been executed"
      continue
    fi
    if [[ ! -f "${model_path}/config.json" ]]; then
      echo "[warn] step ${step}: no Hugging Face config at ${model_path}; skipping" >&2
      client_rc=1
      continue
    fi

    served_model_name="$(basename "${run_dir}")"
    served_model_name="${served_model_name:0:48}-step-${step}-${USER}-${SLURM_JOB_ID}"
    sglang_args="--model-path ${model_path} --served-model-name ${served_model_name} --tp-size ${TP_SIZE} --dp-size ${DP_SIZE} --host 0.0.0.0 --trust-remote-code --context-length ${MAX_MODEL_LEN} --mem-fraction-static ${GPU_MEM_UTIL} --tokenizer-path ${model_path}"
    [[ -z "${CHAT_TEMPLATE}" ]] || sglang_args+=" --chat-template ${CHAT_TEMPLATE}"

    mkdir -p "${checkpoint_output}/logs"
    echo "[serve] step ${step}: ${model_path}"
    # The controller has CLIENT_CPUS CPUs, while sml's serving tasks request
    # their own CPU shape. Do not leak the controller's task-level Slurm
    # variables into the nested serving allocation: recent Slurm versions
    # otherwise see SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK disagree.
    if ! submit_out="$(env \
      -u SLURM_CPUS_PER_TASK \
      -u SLURM_TRES_PER_TASK \
      sml advanced \
      --partition "${PARTITION}" \
      "${reservation_args[@]}" \
      --replicas "${REPLICAS}" \
      --nodes-per-replica "${NODES_PER_REPLICA}" \
      --framework sglang \
      --environment "${ENV_TOML}" \
      --time "${SERVING_TIME}" \
      --served-model-name "${served_model_name}" \
      --router sglang \
      "${router_args_opt[@]}" \
      --framework-args "${sglang_args}")"; then
      echo "[warn] step ${step}: serving submission failed; continuing" >&2
      client_rc=1
      continue
    fi
    echo "${submit_out}"
    job_id="$(sed -n 's/^Job submitted: \([0-9]\+\).*/\1/p' <<< "${submit_out}" | head -n1)"
    if [[ -z "${job_id}" ]]; then
      echo "[warn] step ${step}: could not parse serving job id; continuing" >&2
      client_rc=1
      continue
    fi
    current_serving_job="${job_id}"
    ln -sfn "${HOME}/.sml/logs/${job_id}" "${checkpoint_output}/logs/serving"

    checkpoint_rc=0
    for eval_name in "${EVALS[@]}"; do
      eval_dir="$(eval_output_dir "${run_dir}" "${step}" "${eval_name}")"
      parquet="${EVAL_ROOT}/${eval_name}.parquet"
      if eval_is_done "${eval_dir}"; then
        echo "[skip] step ${step}/${eval_name}: results already exist"
        continue
      fi

      mkdir -p "${eval_dir}"
      touch "${eval_dir}/.eval_sweep_incomplete"
      echo "[eval] step ${step}/${eval_name} -> ${eval_dir}"
      if run_generate "${job_id}" "${served_model_name}" "${parquet}" "${eval_dir}"; then
        mv -f "${eval_dir}/.eval_sweep_incomplete" "${eval_dir}/.complete"
      else
        echo "[warn] step ${step}/${eval_name} failed; later evals will still run" >&2
        checkpoint_rc=1
        client_rc=1
      fi
    done

    cleanup_serving
    if ((checkpoint_rc == 0)); then
      echo "[done] step ${step}"
    fi
  done

  echo "[done] sweep finished for ${run_dir} with status ${client_rc}"
  return "${client_rc}"
}

orchestrator_main() {
  local -a run_dirs=() validated_run_dirs=() checkpoints=() reservation_args=() summary=()
  local arg run_dir checkpoint checkpoint_name step stage run_name model_path
  local pending job_dir state_file previous_job submit_out job_id script_path

  while (($#)); do
    arg="$1"
    case "${arg}" in
      -n|--dry-run) DRY_RUN=1 ;;
      -h|--help) usage; return 0 ;;
      --checkpoint)
        (($# >= 2)) || die "--checkpoint requires a Hugging Face model directory"
        run_dirs+=("$2")
        shift
        ;;
      --) shift; run_dirs+=("$@"); break ;;
      -*) die "unknown option: ${arg}" ;;
      *) run_dirs+=("${arg}") ;;
    esac
    shift
  done
  ((${#run_dirs[@]} > 0)) || { usage >&2; return 2; }

  validate_common_inputs
  script_path="$(realpath "$0")"
  [[ -z "${RESERVATION}" ]] || reservation_args=(--reservation "${RESERVATION}")

  # Validate and plan every run before submitting the first job.
  for arg in "${run_dirs[@]}"; do
    run_dir="$(realpath "${arg}")" || die "SFT output directory does not exist: ${arg}"
    [[ -d "${run_dir}" ]] || die "not a directory: ${run_dir}"
    mapfile -t checkpoints < <(checkpoint_dirs "${run_dir}")
    ((${#checkpoints[@]} > 0)) || \
      die "no global_step_<N> checkpoints or Hugging Face model in ${run_dir}"
    for checkpoint in "${checkpoints[@]}"; do
      model_path="$(checkpoint_model_path "${checkpoint}")"
      [[ -f "${model_path}/config.json" ]] || \
        die "checkpoint has no Hugging Face config: ${model_path}"
    done
    validated_run_dirs+=("${run_dir}")
  done

  for run_dir in "${validated_run_dirs[@]}"; do
    mapfile -t checkpoints < <(checkpoint_dirs "${run_dir}")
    pending=0
    for checkpoint in "${checkpoints[@]}"; do
      checkpoint_name="${checkpoint##*/}"
      step="${checkpoint_name#global_step_}"
      checkpoint_has_pending_evals "${run_dir}" "${step}" && ((pending += 1))
    done
    if ((pending == 0)); then
      echo "[skip] ${run_dir}: all checkpoint evaluations already exist"
      summary+=("${run_dir}: complete")
      continue
    fi

    stage="$(basename "$(dirname "${run_dir}")")"
    run_name="$(basename "${run_dir}")"
    job_dir="${OUTPUT_ROOT}/${stage}/.sweep_logs/${run_name}__thinking-${THINKING}"
    state_file="${job_dir}/job_id"
    if [[ -s "${state_file}" ]] && command -v squeue >/dev/null 2>&1; then
      previous_job="$(<"${state_file}")"
      if [[ "${previous_job}" =~ ^[0-9]+$ ]] && \
        [[ -n "$(squeue -h -j "${previous_job}" -o '%i' 2>/dev/null)" ]]; then
        echo "[skip] ${run_dir}: sweep job ${previous_job} is already queued/running"
        summary+=("${run_dir}: already active as ${previous_job}")
        continue
      fi
    fi

    echo "[plan] ${run_dir}: ${pending}/${#checkpoints[@]} checkpoint(s) have pending evals"
    if enabled "${DRY_RUN}"; then
      summary+=("${run_dir}: dry-run, would submit one controller job")
      continue
    fi

    mkdir -p "${job_dir}"
    export REPO OUTPUT_ROOT EVAL_SUITE EVAL_ROOT THINKING STOP_ON_FIRST_CORRECT
    export CORRECT_THRESHOLD REPEATS SEED CONCURRENCY VERIFY_CONCURRENCY
    export TEMPERATURE TOP_P MAX_TOKENS START END REPLICAS
    export NODES_PER_REPLICA TP_SIZE DP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL
    export CHAT_TEMPLATE ROUTER_ARGS PARTITION ACCOUNT RESERVATION SERVING_TIME SWEEP_TIME
    export CLIENT_CPUS KUBERNETES_SANDBOX_URL SKIP_SPECIAL_TOKENS
    export TOP_K CHAT_TEMPLATE_KWARGS MAX_RETRIES ENV_TOML
    if ! submit_out="$(sbatch \
      --parsable \
      --job-name="eval-${run_name:0:32}" \
      --partition="${PARTITION}" \
      --account="${ACCOUNT}" \
      "${reservation_args[@]}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${CLIENT_CPUS}" \
      --time="${SWEEP_TIME}" \
      --output="${job_dir}/sweep_%j.log" \
      "${script_path}" --worker "${run_dir}")"; then
      echo "[warn] failed to submit sweep for ${run_dir}" >&2
      summary+=("${run_dir}: SUBMISSION FAILED")
      continue
    fi
    job_id="${submit_out%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || die "could not parse sbatch job id from: ${submit_out}"
    printf '%s\n' "${job_id}" > "${state_file}"
    echo "[ok] submitted controller job ${job_id} for ${run_dir}"
    summary+=("${run_dir}: controller=${job_id}, log=${job_dir}/sweep_${job_id}.log")
  done

  echo "============================================================"
  printf '  %s\n' "${summary[@]}"
  echo "============================================================"
}

if [[ "${1:-}" == "--worker" ]]; then
  (($# == 2)) || die "internal --worker mode requires exactly one run directory"
  worker_main "$2"
else
  orchestrator_main "$@"
fi
