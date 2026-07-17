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

# INPUT_PARQUET may hold several whitespace-separated parquets (eval suites).
read -r -a input_parquets <<< "$INPUT_PARQUET"

early_stop_arg=()
case "${STOP_ON_FIRST_CORRECT:-0}" in
  1|true|yes|on) early_stop_arg=(--stop-on-first-correct) ;;
esac

# optional per-model sampling / template / retry knobs
extra_args=()
[[ -n "${TOP_K:-}" ]] && extra_args+=(--top-k "$TOP_K")
[[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]] && extra_args+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
[[ -n "${MAX_RETRIES:-}" ]] && extra_args+=(--max-retries "$MAX_RETRIES")

run_generate() {  # run_generate <output-dir> <parquet>...
  local out="$1"; shift
  python "$REPO/src/generate.py" \
    --job-id "$JOB_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --input "$@" \
    --output-dir "$out" \
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
}

if [[ "${PER_INPUT_SUBDIR:-0}" == "1" ]]; then
  # Evals: one generate.py call per parquet, sequentially against the one serving
  # fleet, each into its own <output-dir>/<eval>/results.jsonl so they resume
  # independently. A failing eval must not skip the ones after it.
  client_rc=0
  for parquet in "${input_parquets[@]}"; do
    name="$(basename "$parquet" .parquet)"
    echo "[eval] === $name ($parquet) -> $OUTPUT_DIR/$name"
    run_generate "$OUTPUT_DIR/$name" "$parquet"
    rc=$?
    [[ $rc -ne 0 ]] && { echo "[warn] $name exited $rc; continuing." >&2; client_rc=$rc; }
  done
else
  run_generate "$OUTPUT_DIR" "${input_parquets[@]}"
  client_rc=$?
fi

# Keep the service up after the client finishes unless asked to tear it down.
if [[ "${KEEP_ALIVE:-1}" == "0" ]]; then
  echo "[cleanup] KEEP_ALIVE=0 -> scancel $JOB_ID (serving)"
  scancel "$JOB_ID" || true
fi
exit $client_rc
