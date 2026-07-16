#!/bin/bash
#SBATCH --job-name=generation-client
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -uo pipefail

cd "$REPO"
source "$REPO/.venv/bin/activate"

end_arg=()
[[ -n "${END:-}" ]] && end_arg=(--end "$END")

early_stop_arg=()
case "${STOP_ON_FIRST_CORRECT:-0}" in
  1|true|yes|on) early_stop_arg=(--stop-on-first-correct) ;;
esac

# optional per-model sampling / template / retry knobs
extra_args=()
[[ -n "${TOP_K:-}" ]] && extra_args+=(--top-k "$TOP_K")
[[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]] && extra_args+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
[[ -n "${MAX_RETRIES:-}" ]] && extra_args+=(--max-retries "$MAX_RETRIES")

python "$REPO/src/generate.py" \
  --job-id "$JOB_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --input "$INPUT_PARQUET" \
  --output-dir "$OUTPUT_DIR" \
  --concurrency "$CONCURRENCY" \
  --verify-concurrency "$VERIFY_CONCURRENCY" \
  --repeats "$REPEATS" \
  --correct-threshold "$CORRECT_THRESHOLD" "${early_stop_arg[@]}" \
  --seed "$SEED" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --enable-thinking "$THINKING" \
  --skip-special-tokens "$SKIP_SPECIAL_TOKENS" \
  --start "$START" "${end_arg[@]}" "${extra_args[@]}"
client_rc=$?

# Keep the service up after the client finishes unless asked to tear it down.
if [[ "${KEEP_ALIVE:-1}" == "0" ]]; then
  echo "[cleanup] KEEP_ALIVE=0 -> scancel $JOB_ID (serving)"
  scancel "$JOB_ID" || true
fi
exit $client_rc
