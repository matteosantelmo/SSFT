#!/bin/bash
# =============================================================================
# SSFT eval sweep — orchestrator.
#
# Runs the EVALS suite against every checkpoint in MODELS. For each model:
#
#   1. launch it as REPLICAS vllm/sglang replicas via `sml advanced`; the
#      replicas register on the mesh and are load-balanced by the serving-api
#      gateway under one served-model-name (scaling = raise REPLICAS);
#   2. submit ONE client Slurm job (CPU-only: HTTP to the gateway +
#      verification). It waits for the serving model (via --job-id), then runs
#      src/generate.py once per eval, sequentially, against that same fleet — so
#      the model is launched once, not once per eval. Each eval streams with high
#      async concurrency and writes its own results.jsonl incrementally.
#
# All models are submitted in one pass and left to Slurm to schedule — note the
# fleet is REPLICAS x len(MODELS) nodes if everything runs at once. The script
# only submits and returns; nothing heavy runs on the login node, so the sweep
# survives your shell disconnecting.
#
# Run from a Clariden login node:
#
#       ./scripts/generation/run_pipeline_evals.sh          # submit the sweep
#       DRY_RUN=1 ./scripts/generation/run_pipeline_evals.sh  # print, submit nothing
#
# Outputs land under, one dir per model:
#   <OUTPUT_ROOT>/<stage>/<model-name>__thinking-<on|off>/
#     <eval>/results.jsonl generated responses + scores, one dir per eval, so a
#                          re-run resumes each eval independently
#     logs/client.log      the client job log for ALL evals (progress, errors)
#     logs/serving -> ~/.sml/logs/<job-id>   (symlink to sml serving logs)
#
# Everything below is overridable via environment variables.
# =============================================================================
set -euo pipefail

REPO="/iopsstor/scratch/cscs/msantelmo/SSFT"
cd "$REPO"

# ---- configuration (override via env) --------------------------------------
# Pick the evals to run — each is generated + verified by its own sequential
# generate.py call and lands in its own <output-dir>/<eval>/results.jsonl.
# Comment out whatever you don't want in this run.
EVAL_SUITE="evals-small_nothink_nodisplay"
EVAL_ROOT="$REPO/data/$EVAL_SUITE/eval"
EVALS=(
  aime2024
  aime2025
  math500
  gpqa_diamond
  gsm8k
  openai_humaneval
  ifeval
  ifbench
  # mmlu skipped because very large
)
THINKING=off

# Models to evaluate — one "<stage> <checkpoint-name>" per line, resolved under
# CKPT_ROOT/<stage>/<checkpoint-name>. Every model gets its own serving job and
# its own client job; all pairs are submitted in one go and Slurm queues them.
CKPT_ROOT="/iopsstor/scratch/cscs/msantelmo/checkpoints"
MODELS=(
  "sft_1 Apertus-8B-2509__sft0-cap-filter-fill__20260716-001239__global_step_12618"
  # "sft_1 Apertus-8B-2509__sft0-teacher__20260716-003107__global_step_12618"
  # "sft_1 Apertus-8B-2509-sft_0-11776__cap-filter-fill__20260716-212649__global_step_1010"
  # "sft_1 Apertus-8B-2509-sft_0-11776__cap-filter-hard__20260716-212621__global_step_344"
  # "sft_1 Apertus-8B-2509-sft_0-11776__teacher__20260716-214546__global_step_1010"
  # "sft_0 apertus-1p5_8b_seq_len_256k_7000__sft_0_lr5e-5-ratio03__global_step_11264"
  # "sft_0 Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_11776"
  # "sft_0 Apertus-8B-2509__sft_0_lr5e-5-ratio03__global_step_6656"
)
OUTPUT_ROOT="/users/msantelmo/scratch/SSFT/outputs/evals"
CHAT_TEMPLATE=""

if [[ "$THINKING" == "on" ]]; then
  # Must be false to parse reasoning <|inner_prefix|>/<|inner_suffix|>
  SKIP_SPECIAL_TOKENS=false
  THINKING_TAG="-think"
else
  SKIP_SPECIAL_TOKENS=true
  THINKING_TAG="-no-think"
fi

STOP_ON_FIRST_CORRECT=off
CORRECT_THRESHOLD="${CORRECT_THRESHOLD:-0.7}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
REPEATS=32
REPLICAS=4
NODES_PER_REPLICA=1
TP_SIZE=1
DP_SIZE=4
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.8
FRAMEWORK="sglang"

if [[ "$FRAMEWORK" == "sglang" ]]; then
  ENV_TOML="$REPO/model_launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
  ROUTER="sglang"
  ROUTER_ARGS="${ROUTER_ARGS:-}"
else
ENV_TOML="$REPO/model_launch/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml"
  ROUTER="opentela"
  ROUTER_ARGS=""
fi
PARTITION="normal"
RESERVATION="SD-69241-apertus-1-5-0"

# client-side knobs (forwarded to src/generate.py)
# Code verifiers (taco/apps/codeforces/...) run in the Kubernetes sandbox when
# this is set; unset it to fall back to local prime_code execution.
KUBERNETES_SANDBOX_URL="${KUBERNETES_SANDBOX_URL:-https://sandbox-dev.swissai.svc.cscs.ch}"
# Semantic response formatting: none, markdown, xml, or xml_think. An empty
# prompt uses the parser default; prompt role is system or user.
OUTPUT_FORMATTING_PARSER="${OUTPUT_FORMATTING_PARSER:-none}"
OUTPUT_FORMATTING_PROMPT="${OUTPUT_FORMATTING_PROMPT:-}"
OUTPUT_FORMATTING_PROMPT_ROLE="${OUTPUT_FORMATTING_PROMPT_ROLE:-system}"
SEED="85"
CONCURRENCY="${CONCURRENCY:-512}"
VERIFY_CONCURRENCY="${VERIFY_CONCURRENCY:-32}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS=8129
START="${START:-0}"
END="${END:-}"

# client.sbatch splits INPUT_PARQUET on whitespace into repeated --input args.
eval_parquets=()
for name in "${EVALS[@]}"; do
  parquet="$EVAL_ROOT/$name.parquet"
  [[ -f "$parquet" ]] || { echo "[fatal] no such eval parquet: $parquet" >&2; exit 1; }
  eval_parquets+=("$(realpath "$parquet")")
done
INPUT_PARQUET="${eval_parquets[*]}"

# Resolve every checkpoint up front: a typo should fail here, not after half the
# sweep is already queued.
model_paths=()
for entry in "${MODELS[@]}"; do
  read -r stage model_name <<< "$entry"
  ckpt="$CKPT_ROOT/$stage/$model_name"
  [[ -d "$ckpt" ]] || { echo "[fatal] no such checkpoint: $ckpt" >&2; exit 1; }
  [[ -f "$ckpt/config.json" ]] || { echo "[fatal] not a HF checkpoint (no config.json): $ckpt" >&2; exit 1; }
  model_paths+=("$(realpath "$ckpt")")
done

# ---- environment -----------------------------------------------------------
if [[ ! -x "$REPO/.venv/bin/sml" ]]; then
  echo "[fatal] sml not found in $REPO/.venv — run ./scripts/setup.sh first." >&2
  exit 1
fi
source "$REPO/.venv/bin/activate"

reservation_args=()
[[ -n "$RESERVATION" ]] && reservation_args=(--reservation "$RESERVATION")

router_args_opt=()
[[ -n "$ROUTER_ARGS" ]] && router_args_opt=(--router-args "$ROUTER_ARGS")

CLIENT_TIME="${CLIENT_TIME:-$TIME_LIMIT}"
CLIENT_CPUS="${CLIENT_CPUS:-32}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"
# One generate.py call per eval, sequentially against the model's serving fleet,
# each into its own <output-dir>/<eval>/results.jsonl (independent resume).
PER_INPUT_SUBDIR=1
# DRY_RUN=1 prints what each model would submit without touching Slurm.
DRY_RUN="${DRY_RUN:-0}"

########################################
# One serving job + one client job per model
########################################
summary=()
all_jobs=()
for i in "${!MODELS[@]}"; do
  read -r STAGE MODEL_NAME <<< "${MODELS[i]}"
  MODEL_PATH="${model_paths[i]}"
  TOKENIZER_PATH="$MODEL_PATH"
  # Served name must be unique across the sweep — the gateway routes by it.
  SERVED_MODEL_NAME="${MODEL_NAME}${THINKING_TAG}-$USER"
  OUTPUT_DIR="$OUTPUT_ROOT/$STAGE/${MODEL_NAME}__thinking-${THINKING}"

  if [[ "$FRAMEWORK" == "sglang" ]]; then
    FRAMEWORK_ARGS="--model-path $MODEL_PATH \
      --served-model-name $SERVED_MODEL_NAME \
      --tp-size $TP_SIZE \
      --dp-size $DP_SIZE \
      --host 0.0.0.0 \
      --trust-remote-code \
      --context-length $MAX_MODEL_LEN \
      --mem-fraction-static $GPU_MEM_UTIL"
    [[ -n "$CHAT_TEMPLATE" ]] && FRAMEWORK_ARGS="$FRAMEWORK_ARGS --chat-template $CHAT_TEMPLATE"
    [[ -n "$TOKENIZER_PATH" ]] && FRAMEWORK_ARGS="$FRAMEWORK_ARGS --tokenizer-path $TOKENIZER_PATH"
  else
    FRAMEWORK_ARGS="--model $MODEL_PATH \
      --served-model-name $SERVED_MODEL_NAME \
      --tensor-parallel-size $TP_SIZE \
      --data-parallel-size $DP_SIZE \
      --host 0.0.0.0 \
      --trust-remote-code \
      --trust-request-chat-template \
      --skip-mm-profiling \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEM_UTIL"
    [[ -n "$CHAT_TEMPLATE" ]] && FRAMEWORK_ARGS="$FRAMEWORK_ARGS --chat-template $CHAT_TEMPLATE"
    [[ -n "$TOKENIZER_PATH" ]] && FRAMEWORK_ARGS="$FRAMEWORK_ARGS --tokenizer $TOKENIZER_PATH"
  fi

  ########################################
  # 1. launch serving
  ########################################
  echo "============================================================"
  echo " [$((i + 1))/${#MODELS[@]}] $STAGE / $MODEL_NAME"
  echo "   model      : $MODEL_PATH"
  echo "   served as  : $SERVED_MODEL_NAME"
  echo "   evals      : ${#EVALS[@]} from $EVAL_SUITE -> ${EVALS[*]}"
  echo "   output     : $OUTPUT_DIR"
  echo "   layout     : $REPLICAS replicas x $NODES_PER_REPLICA node (TP=$TP_SIZE, DP=$DP_SIZE -> $((REPLICAS * DP_SIZE)) engines), $FRAMEWORK, router=$ROUTER"
  echo "   client     : concurrency=$CONCURRENCY repeats=$REPEATS thinking=$THINKING"
  echo "   early stop : $STOP_ON_FIRST_CORRECT (correct threshold=$CORRECT_THRESHOLD)"
  echo "   sandbox    : ${KUBERNETES_SANDBOX_URL:-<none — local prime_code>}"
  echo "   chat tmpl  : ${CHAT_TEMPLATE:-<model-dir default>}"
  echo "   tokenizer  : ${TOKENIZER_PATH:-<model-dir default>}"
  echo "   partition  : $PARTITION  reservation: ${RESERVATION:-<none>}  time: $TIME_LIMIT"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    summary+=("$STAGE/$MODEL_NAME  serving=<dry-run> client=<dry-run>  $OUTPUT_DIR")
    continue
  fi
  mkdir -p "$OUTPUT_DIR/logs"

  # One model failing to submit must not abandon the rest of the sweep.
  if ! submit_out="$(sml advanced \
    --partition "$PARTITION" \
    "${reservation_args[@]}" \
    --replicas "$REPLICAS" \
    --nodes-per-replica "$NODES_PER_REPLICA" \
    --framework "$FRAMEWORK" \
    --environment "$ENV_TOML" \
    --time "$TIME_LIMIT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --router "$ROUTER" \
    "${router_args_opt[@]}" \
    --framework-args "$FRAMEWORK_ARGS")"; then
    echo "[warn] sml advanced failed for $STAGE/$MODEL_NAME — skipping." >&2
    summary+=("$STAGE/$MODEL_NAME  SERVING SUBMIT FAILED")
    continue
  fi

  echo "$submit_out"
  JOB_ID="$(echo "$submit_out" | sed -n 's/^Job submitted: \([0-9]\+\).*/\1/p' | head -n1)"
  if [[ -z "$JOB_ID" ]]; then
    echo "[warn] could not parse a job id from sml output for $STAGE/$MODEL_NAME — skipping." >&2
    summary+=("$STAGE/$MODEL_NAME  NO SERVING JOB ID")
    continue
  fi
  echo "[ok] serving job id: $JOB_ID"

  # Surface the sml serving logs inside the run's output dir (live symlink).
  SML_LOG_DIR="$HOME/.sml/logs/$JOB_ID"
  ln -sfn "$SML_LOG_DIR" "$OUTPUT_DIR/logs/serving"

  ########################################
  # 2. Submit generate+verify job
  ########################################
  export REPO JOB_ID SERVED_MODEL_NAME INPUT_PARQUET OUTPUT_DIR \
    CONCURRENCY VERIFY_CONCURRENCY REPEATS SEED TEMPERATURE TOP_P MAX_TOKENS \
    SKIP_SPECIAL_TOKENS THINKING START END KEEP_ALIVE STOP_ON_FIRST_CORRECT \
    CORRECT_THRESHOLD KUBERNETES_SANDBOX_URL PER_INPUT_SUBDIR \
    OUTPUT_FORMATTING_PARSER OUTPUT_FORMATTING_PROMPT OUTPUT_FORMATTING_PROMPT_ROLE

  if ! client_submit="$(sbatch \
    --partition="$PARTITION" \
    "${reservation_args[@]}" \
    --cpus-per-task="$CLIENT_CPUS" \
    --time="$CLIENT_TIME" \
    --output="$OUTPUT_DIR/logs/client.log" \
    "$REPO/scripts/generation/client.sbatch")"; then
    echo "[warn] client sbatch failed for $STAGE/$MODEL_NAME — cancelling its serving job $JOB_ID." >&2
    scancel "$JOB_ID" || true
    summary+=("$STAGE/$MODEL_NAME  CLIENT SUBMIT FAILED (serving $JOB_ID cancelled)")
    continue
  fi
  echo "$client_submit"
  CLIENT_JOB_ID="$(echo "$client_submit" | awk '{print $NF}')"

  echo "[ok] serving=$JOB_ID client=$CLIENT_JOB_ID -> $OUTPUT_DIR"
  summary+=("$STAGE/$MODEL_NAME  serving=$JOB_ID client=$CLIENT_JOB_ID  $OUTPUT_DIR")
  all_jobs+=("$JOB_ID" "$CLIENT_JOB_ID")
  sleep 5
done

echo "============================================================"
echo "[ok] submitted ${#MODELS[@]} model(s) — nothing else runs on the login node."
for line in "${summary[@]}"; do
  echo "   $line"
done
if [[ ${#all_jobs[@]} -gt 0 ]]; then
  echo "   cancel all  : scancel ${all_jobs[*]}"
fi
echo "============================================================"
