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

python "$REPO/src/generate.py" \
  --job-id "$JOB_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --input "$INPUT_PARQUET" \
  --output-dir "$OUTPUT_DIR" \
  --concurrency "$CONCURRENCY" \
  --verify-concurrency "$VERIFY_CONCURRENCY" \
  --repeats "$REPEATS" \
  --seed "$SEED" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --enable-thinking "$THINKING" \
  --skip-special-tokens "$SKIP_SPECIAL_TOKENS" \
  --start "$START" "${end_arg[@]}"
client_rc=$?

# Keep the service up after the client finishes unless asked to tear it down.
if [[ "${KEEP_ALIVE:-1}" == "0" ]]; then
  echo "[cleanup] KEEP_ALIVE=0 -> scancel $JOB_ID (serving)"
  scancel "$JOB_ID" || true
fi
exit $client_rc
