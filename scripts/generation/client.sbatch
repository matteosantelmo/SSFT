#!/bin/bash
#SBATCH --job-name=generation-client
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=nid006634,nid006701,nid006948,nid006588,nid006629,nid006910,nid007254,nid007078,nid006619,nid006840,nid006905,nid006941,nid006947,nid006922,nid007074,nid007131,nid007189,nid007129,nid007184,nid007176,nid007177,nid007183,nid007090,nid007551,nid007531,nid007539,nid007558,nid006988,nid006990,nid006987,nid006989,nid007363,nid006606,nid007410,nid007096,nid007566,nid006774,nid007343,nid006867,nid007323,nid007489,nid006676,nid006677,nid007411,nid006848,nid006681,nid007626,nid007612,nid006887,nid006577,nid006729,nid006831,nid007520,nid007589,nid007614,nid006955,nid007592,nid007344,nid007374,nid007134,nid007628,nid007382,nid007141,nid007155,nid007286,nid006589,nid007024,nid007025,nid006065,nid006076,nid006080,nid006081,nid006082,nid006085,nid006086

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
