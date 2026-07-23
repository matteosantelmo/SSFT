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

REPO="/iopsstor/scratch/cscs/msantelmo/SSFT"
cd "$REPO"

# ---- configuration (override via env) --------------------------------------
DOMAIN=code
INPUT_PARQUET=/iopsstor/scratch/cscs/msantelmo/SSFT/data/cap_filter/${DOMAIN}/train.parquet
PROJECT_NAME="capability-filtering"

# MODEL_PATH=/capstor/scratch/cscs/msantelmo/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
# MODEL_PATH=/capstor/store/cscs/swissai/infra01/hf_models/models/MiniMaxAI/MiniMax-M2.7
MODEL_PATH=/capstor/store/cscs/swissai/infra01/hf_models/models/google/gemma-4-31B-it
OUTPUT_DIR="/users/msantelmo/scratch/SSFT/outputs/teacher_gemma-4-31B/${DOMAIN}"
TOKENIZER_PATH=""
CHAT_TEMPLATE=""
THINKING=on
FRAMEWORK=vllm  # sglang

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
REPLICAS=24
NODES_PER_REPLICA=1
TP_SIZE=4
DP_SIZE=1
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.7

# ---- model-family presets ----------------------------------------------------
# DeepSeek-V4-Flash (284B-A13B, FP4 experts + FP8, ~149GB) and MiniMax-M2.7
# (230B-A10B, FP8, ~215GB) both fit on ONE 4xGH200 node (4x96GB), so replicas
# stay single-node (no cross-node TP) and throughput scales via REPLICAS.
# Presets follow the vendor/vLLM-recipe serving commands; sampling defaults are
# the model-card recommendations (override via env as usual).
TOP_K="${TOP_K:-}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-}"
EXTRA_FRAMEWORK_ARGS="${EXTRA_FRAMEWORK_ARGS:-}"
case "$(basename "$MODEL_PATH")" in
  [Dd]eep[Ss]eek-V4*)
    FRAMEWORK=vllm
    NODES_PER_REPLICA=1
    # MLA-style attention (1 KV head): TP would replicate the KV cache on every
    # rank, so the vLLM recipe recommends DP attention + EP experts instead.
    # Fallback if the DP path misbehaves: TP_SIZE=4 DP_SIZE=1 (keep EP on).
    TP_SIZE=1
    DP_SIZE=4
    EXTRA_FRAMEWORK_ARGS="--enable-expert-parallel --kv-cache-dtype fp8 --block-size 256 \
      --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice"
    SKIP_SPECIAL_TOKENS=true  # reasoning comes back separately via the parser
    TEMPERATURE="${TEMPERATURE:-1.0}"
    TOP_P="${TOP_P:-1.0}"
    # thinking modes: {"thinking": false} = non-think, {"thinking": true} = think
    # high; think max additionally needs {"reasoning_effort": "max"} + >=384K ctx.
    if [[ "$THINKING" == "on" ]]; then
      CHAT_TEMPLATE_KWARGS='{"thinking": true}'
    else
      CHAT_TEMPLATE_KWARGS='{"thinking": false}'
    fi
    PRE_LAUNCH_CMDS=""  # stock vllm image serves deepseek_v4; don't touch transformers
    ;;
  [Mm]ini[Mm]ax-M2*)
    FRAMEWORK=vllm
    NODES_PER_REPLICA=1
    TP_SIZE=4
    DP_SIZE=1
    GPU_MEM_UTIL=0.85
    EXTRA_FRAMEWORK_ARGS="--reasoning-parser minimax_m2_append_think \
      --tool-call-parser minimax_m2 --enable-auto-tool-choice"
    SKIP_SPECIAL_TOKENS=true
    TEMPERATURE="${TEMPERATURE:-1.0}"
    TOP_P="${TOP_P:-0.95}"
    TOP_K="${TOP_K:-40}"
    CHAT_TEMPLATE_KWARGS='{}'  # template always opens <think>; there is no toggle
    [[ "$THINKING" != "on" ]] && echo "[warn] MiniMax-M2 always thinks; THINKING=$THINKING has no effect." >&2
    PRE_LAUNCH_CMDS=""
    ;;
  [Gg]emma-4-*)
    # Gemma 4 emits reasoning as <|channel>thought ... <channel|>. Let
    # vLLM split it into reasoning_content and final content instead of
    # treating its structural tokens as part of the answer.
    EXTRA_FRAMEWORK_ARGS="${EXTRA_FRAMEWORK_ARGS:+$EXTRA_FRAMEWORK_ARGS }--reasoning-parser gemma4"
    SKIP_SPECIAL_TOKENS=true
    ;;
esac

if [[ "$FRAMEWORK" == "sglang" ]]; then
  ENV_TOML="$REPO/model_launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
  ROUTER="sglang"
  ROUTER_ARGS="${ROUTER_ARGS:-}"
else
  if [[ "${MODEL_PATH,,}" == *apertus* ]]; then
    ENV_TOML="$REPO/model_launch/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml"
  else
    ENV_TOML="$REPO/model_launch/src/swiss_ai_model_launch/assets/envs/vllm.toml"
  fi
  ROUTER="opentela"
  ROUTER_ARGS=""
fi
PARTITION="normal"
RESERVATION="SD-69241-apertus-1-5-0"
EXCLUDE_NODES="${EXCLUDE_NODES:-nid007277,nid006065,nid006076,nid006080,nid006081,nid006082,nid006085,nid006086}"

# Runs inside each replica's container (writable overlay) before vllm starts.
# The image's transformers is too old for gemma-4 (`model type gemma4` not
# recognized) — upgrade it at startup. 
PRE_LAUNCH_CMDS="${PRE_LAUNCH_CMDS-python3 -m pip install --no-cache-dir --upgrade 'transformers==5.13.1'}"

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

MAX_RETRIES="${MAX_RETRIES:-2}"
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
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")${THINKING_TAG}-${DOMAIN}-$USER
MODEL_NAME="$(basename "$MODEL_PATH")"
DATASET_NAME="$(basename "$(dirname "$INPUT_PARQUET")")"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/$PROJECT_NAME/${MODEL_NAME}${THINKING_TAG}__${DATASET_NAME}_${STAMP}}"
mkdir -p "$OUTPUT_DIR/logs"

reservation_args=()
[[ -n "$RESERVATION" ]] && reservation_args=(--reservation "$RESERVATION")
exclude_nodes_args=()
[[ -n "$EXCLUDE_NODES" ]] && exclude_nodes_args=(--exclude-nodes "$EXCLUDE_NODES")

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
[[ -n "$EXTRA_FRAMEWORK_ARGS" ]] && FRAMEWORK_ARGS="$FRAMEWORK_ARGS $EXTRA_FRAMEWORK_ARGS"

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
echo "   client     : concurrency=$CONCURRENCY max_retries=$MAX_RETRIES repeats=$REPEATS thinking=$THINKING"
echo "   sampling   : temp=$TEMPERATURE top_p=$TOP_P top_k=${TOP_K:-<off>} max_tokens=$MAX_TOKENS tmpl_kwargs=${CHAT_TEMPLATE_KWARGS:-<default>}"
echo "   early stop : $STOP_ON_FIRST_CORRECT (correct threshold=$CORRECT_THRESHOLD)"
echo "   sandbox    : ${KUBERNETES_SANDBOX_URL:-<none — local prime_code>}"
echo "   chat tmpl  : ${CHAT_TEMPLATE:-<model-dir default>}"
echo "   tokenizer  : ${TOKENIZER_PATH:-<model-dir default>}"
echo "   partition  : $PARTITION  reservation: ${RESERVATION:-<none>}  exclude: ${EXCLUDE_NODES:-<none>}  time: $TIME_LIMIT"
echo "============================================================"

router_args_opt=()
[[ -n "$ROUTER_ARGS" ]] && router_args_opt=(--router-args "$ROUTER_ARGS")

pre_launch_opt=()
[[ -n "$PRE_LAUNCH_CMDS" ]] && pre_launch_opt=(--pre-launch-cmds "$PRE_LAUNCH_CMDS")

submit_out="$(sml advanced \
  --partition "$PARTITION" \
  "${reservation_args[@]}" \
  "${exclude_nodes_args[@]}" \
  --replicas "$REPLICAS" \
  --nodes-per-replica "$NODES_PER_REPLICA" \
  --framework "$FRAMEWORK" \
  --environment "$ENV_TOML" \
  --time "$TIME_LIMIT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --router "$ROUTER" \
  "${router_args_opt[@]}" \
  "${pre_launch_opt[@]}" \
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
  CONCURRENCY VERIFY_CONCURRENCY REPEATS SEED TEMPERATURE TOP_P TOP_K MAX_TOKENS \
  SKIP_SPECIAL_TOKENS THINKING CHAT_TEMPLATE_KWARGS START END KEEP_ALIVE \
  STOP_ON_FIRST_CORRECT CORRECT_THRESHOLD KUBERNETES_SANDBOX_URL MAX_RETRIES \
  OUTPUT_FORMATTING_PARSER OUTPUT_FORMATTING_PROMPT OUTPUT_FORMATTING_PROMPT_ROLE

client_submit="$(sbatch \
  --partition="$PARTITION" \
  "${reservation_args[@]}" \
  --cpus-per-task="$CLIENT_CPUS" \
  --time="$CLIENT_TIME" \
  --output="$OUTPUT_DIR/logs/client.log" \
  "$REPO/scripts/generation/client.sbatch")"
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
