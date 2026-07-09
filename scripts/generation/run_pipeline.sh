#!/bin/bash
# =============================================================================
# SSFT generation + verification pipeline — orchestrator.
#
#   1. launch the model as REPLICAS vllm replicas via `sml advanced`; the
#      replicas register on the OpenTela mesh and are load-balanced by the
#      serving-api gateway (scaling = raise REPLICAS, no endpoint discovery);
#   2. submit src/generate.py as its OWN 1-node Slurm job (CPU-only: HTTP to the
#      gateway + verification). That job waits for the serving model (via
#      --job-id), then streams prompts from INPUT_PARQUET with high async
#      concurrency, verifies each response, and writes results.jsonl
#      incrementally (resumable).
#
# This script only submits the two jobs and returns immediately — nothing heavy
# runs on the login node, so the run survives your shell disconnecting.
#
# Run from a Clariden login node:
#
#       INPUT_PARQUET=/path/to/data.parquet PROJECT_NAME=my_eval ./scripts/run_pipeline.sh
#
# Outputs land under:
#   outputs/<PROJECT_NAME>/<model-name>_<dataset-name>_<datetime>/
#     results.jsonl        generated responses + scores
#     logs/client.log      the generate.py client job log (progress, errors)
#     logs/serving -> ~/.sml/logs/<job-id>   (symlink to sml serving logs)
#
# Everything below is overridable via environment variables.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- configuration (override via env) --------------------------------------
INPUT_PARQUET=/users/msantelmo/scratch/SSFT/data/sft_reasoning_dataset/train.parquet
PROJECT_NAME="capability-filtering"

MODEL_PATH=/users/msantelmo/scratch/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed
# MODEL_PATH=/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200
TOKENIZER_PATH="/capstor/store/cscs/swissai/infra01/reasoning/models/tokenizers/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
CHAT_TEMPLATE=""
THINKING=off
OUTPUT_DIR="/users/msantelmo/scratch/SSFT/outputs/capability-filtering/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed-no-think__sft_reasoning_dataset_20260630-135930"

if [[ "$THINKING" == "on" ]]; then
  # Must be false to parse reasoning <|inner_prefix|>/<|inner_suffix|>
  SKIP_SPECIAL_TOKENS=false
  THINKING_TAG="-think"
else
  SKIP_SPECIAL_TOKENS=true
  THINKING_TAG="-no-think"
fi

STOP_ON_FIRST_CORRECT=on
CORRECT_THRESHOLD="${CORRECT_THRESHOLD:-0.7}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
REPEATS=8
REPLICAS=10
NODES_PER_REPLICA=1
TP_SIZE=1
DP_SIZE=4
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.8

FRAMEWORK="${FRAMEWORK:-sglang}"
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
SEED="85"
CONCURRENCY="${CONCURRENCY:-512}"
VERIFY_CONCURRENCY="${VERIFY_CONCURRENCY:-32}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS=16384
START="${START:-0}"
END="${END:-}"

MODEL_PATH="$(realpath "$MODEL_PATH")"
INPUT_PARQUET="$(realpath "$INPUT_PARQUET")"

# ---- environment -----------------------------------------------------------
if [[ ! -x "$REPO/.venv/bin/sml" ]]; then
  echo "[fatal] sml not found in $REPO/.venv — run ./scripts/setup.sh first." >&2
  exit 1
fi
source "$REPO/.venv/bin/activate"

# ---- output dir ------------------------------------------------------------
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")${THINKING_TAG}-$USER
MODEL_NAME="$(basename "$MODEL_PATH")"
DATASET_NAME="$(basename "$(dirname "$INPUT_PARQUET")")"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/$PROJECT_NAME/${MODEL_NAME}${THINKING_TAG}__${DATASET_NAME}_${STAMP}}"
mkdir -p "$OUTPUT_DIR/logs"

reservation_args=()
[[ -n "$RESERVATION" ]] && reservation_args=(--reservation "$RESERVATION")

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
echo " Launching generation pipeline"
echo "   model      : $MODEL_PATH"
echo "   served as  : $SERVED_MODEL_NAME"
echo "   input      : $INPUT_PARQUET"
echo "   output     : $OUTPUT_DIR"
echo "   layout     : $REPLICAS replicas x $NODES_PER_REPLICA node (TP=$TP_SIZE, DP=$DP_SIZE -> $((REPLICAS * DP_SIZE)) engines), $FRAMEWORK, router=$ROUTER"
echo "   client     : concurrency=$CONCURRENCY repeats=$REPEATS thinking=$THINKING"
echo "   early stop : $STOP_ON_FIRST_CORRECT (correct threshold=$CORRECT_THRESHOLD)"
echo "   chat tmpl  : ${CHAT_TEMPLATE:-<model-dir default>}"
echo "   tokenizer  : ${TOKENIZER_PATH:-<model-dir default>}"
echo "   partition  : $PARTITION  reservation: ${RESERVATION:-<none>}  time: $TIME_LIMIT"
echo "============================================================"

router_args_opt=()
[[ -n "$ROUTER_ARGS" ]] && router_args_opt=(--router-args "$ROUTER_ARGS")

submit_out="$(sml advanced \
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
  --framework-args "$FRAMEWORK_ARGS")"

echo "$submit_out"
JOB_ID="$(echo "$submit_out" | sed -n 's/^Job submitted: \([0-9]\+\).*/\1/p' | head -n1)"
if [[ -z "$JOB_ID" ]]; then
  echo "[fatal] could not parse a job id from sml output." >&2
  exit 1
fi
echo "[ok] serving job id: $JOB_ID"

# Surface the sml serving logs inside the run's output dir (live symlink).
SML_LOG_DIR="$HOME/.sml/logs/$JOB_ID"
ln -sfn "$SML_LOG_DIR" "$OUTPUT_DIR/logs/serving"

########################################
# 2. Submit generate+verify job
########################################
CLIENT_TIME="${CLIENT_TIME:-$TIME_LIMIT}"
CLIENT_CPUS="${CLIENT_CPUS:-32}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"

export REPO JOB_ID SERVED_MODEL_NAME INPUT_PARQUET OUTPUT_DIR \
  CONCURRENCY VERIFY_CONCURRENCY REPEATS SEED TEMPERATURE TOP_P MAX_TOKENS \
  SKIP_SPECIAL_TOKENS THINKING START END KEEP_ALIVE STOP_ON_FIRST_CORRECT \
  CORRECT_THRESHOLD

client_submit="$(sbatch \
  --partition="$PARTITION" \
  "${reservation_args[@]}" \
  --cpus-per-task="$CLIENT_CPUS" \
  --time="$CLIENT_TIME" \
  --output="$OUTPUT_DIR/logs/client.log" \
  "$REPO/scripts/client.sbatch")"
echo "$client_submit"
CLIENT_JOB_ID="$(echo "$client_submit" | awk '{print $NF}')"

echo "============================================================"
echo "[ok] submitted — nothing else runs on the login node."
echo "   serving job : $JOB_ID"
echo "   client job  : $CLIENT_JOB_ID"
echo "   results     : $OUTPUT_DIR/results.jsonl"
echo "   client log  : tail -f $OUTPUT_DIR/logs/client.log"
echo "   cancel both : scancel $JOB_ID $CLIENT_JOB_ID"
echo "============================================================"
