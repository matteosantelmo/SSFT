#!/bin/bash
# =============================================================================
# SSFT serving PoC — orchestrator (the *launching* half of the pipeline).
#
#   1. launch the model as 2 replicas (2 nodes x 4 GPUs, TP=4) via `sml advanced`
#      (Swiss AI model-launch); the replicas register on the OpenTela mesh and
#      are load-balanced by the serving-api gateway;
#   2. hand the Slurm job id to src/client.py, which waits for the model to be
#      ready and then sends example OpenAI-API queries and prints the outputs.
#
# Run this from a Clariden login node (it submits a Slurm job and then runs the
# lightweight client locally):
#
#       ./scripts/run_poc.sh
#
# Everything below is overridable via environment variables.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- configuration (override via env) --------------------------------------
MODEL_PATH="${MODEL_PATH:-/users/msantelmo/scratch/checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816-thinking-token-fixed}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-apertus-8b-thinking-$USER}"
ENV_TOML="${ENV_TOML:-$REPO/model_launch/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml}"
FRAMEWORK="${FRAMEWORK:-vllm}"
PARTITION="${PARTITION:-normal}"
RESERVATION="${RESERVATION:-SD-69241-apertus-1-5-0}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
REPLICAS="${REPLICAS:-2}"
NODES_PER_REPLICA="${NODES_PER_REPLICA:-1}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

MODEL_PATH="$(realpath "$MODEL_PATH")"

# ---- environment -----------------------------------------------------------
if [[ ! -x "$REPO/.venv/bin/sml" ]]; then
  echo "[fatal] sml not found in $REPO/.venv — run ./setup.sh first." >&2
  exit 1
fi
source "$REPO/.venv/bin/activate"

reservation_args=()
[[ -n "$RESERVATION" ]] && reservation_args=(--reservation "$RESERVATION")

# --served-model-name is forwarded *inside* --framework-args so vllm itself
# advertises this id on /v1/models (the top-level flag only sets mesh labels).
FRAMEWORK_ARGS="--model $MODEL_PATH \
  --served-model-name $SERVED_MODEL_NAME \
  --tensor-parallel-size $TP_SIZE \
  --host 0.0.0.0 \
  --trust-remote-code \
  --trust-request-chat-template \
  --skip-mm-profiling \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL"

echo "============================================================"
echo " Launching serving job"
echo "   model      : $MODEL_PATH"
echo "   served as  : $SERVED_MODEL_NAME"
echo "   layout     : $REPLICAS replicas x $NODES_PER_REPLICA node (TP=$TP_SIZE), OpenTela-routed via gateway"
echo "   partition  : $PARTITION  reservation: ${RESERVATION:-<none>}  time: $TIME_LIMIT"
echo "============================================================"

# ---- (1) launch serving ----------------------------------------------------
submit_out="$(sml advanced \
  --partition "$PARTITION" \
  "${reservation_args[@]}" \
  --replicas "$REPLICAS" \
  --nodes-per-replica "$NODES_PER_REPLICA" \
  --framework "$FRAMEWORK" \
  --environment "$ENV_TOML" \
  --time "$TIME_LIMIT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --framework-args "$FRAMEWORK_ARGS")" 

echo "$submit_out"
JOB_ID="$(echo "$submit_out" | sed -n 's/^Job submitted: \([0-9]\+\).*/\1/p' | head -n1)"
if [[ -z "$JOB_ID" ]]; then
  echo "[fatal] could not parse a job id from sml output." >&2
  exit 1
fi
echo "[ok] serving job id: $JOB_ID"
echo "[hint] cancel later with:  scancel $JOB_ID"

# ---- (2) wait for readiness + query (all waiting logic lives in src/client.py)
python "$REPO/src/client.py" \
  --job-id "$JOB_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P"

# Keep the service up after the client finishes unless asked to tear it down.
if [[ "${KEEP_ALIVE:-1}" == "0" ]]; then
  echo "[cleanup] KEEP_ALIVE=0 -> scancel $JOB_ID"
  scancel "$JOB_ID"
else
  echo "[note] serving job $JOB_ID still running; scancel $JOB_ID when done."
fi
