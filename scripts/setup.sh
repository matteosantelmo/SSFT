#!/bin/bash
# One-time setup: create a venv and install the model-launch CLI (sml) + openai.
# Re-run safely; it's idempotent.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if ! command -v uv >/dev/null 2>&1; then
  echo "[fatal] 'uv' not found on PATH. Install it: https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Pull the model-launch submodule if it hasn't been checked out yet.
if [[ ! -f "$REPO/model_launch/pyproject.toml" ]]; then
  git -C "$REPO" submodule update --init --recursive
fi

# Create the venv and install sml (editable, from the submodule) + the openai client.
uv venv --python 3.12 "$REPO/.venv"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
uv pip install -e "$REPO/model_launch" openai

# Dependencies for the generation+verification pipeline
uv pip install \
  pandas "pyarrow>=19.0.0" tqdm \
  math-verify latex2sympy2-extended pylatexenc "numpy<2.0.0" \
  nltk langdetect immutabledict emoji syllapy "setuptools<81"

# The instruction-following verifier tokenizes with nltk; fetch its data once.
python - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as exc:
        print(f"[warn] nltk.download({pkg!r}) failed: {exc}")
PY

# Remember: `sml init` (once) configures the slurm launcher + CSCS API key,
# then `./scripts/run_poc.sh` launches the PoC, or `./scripts/run_pipeline.sh`
# runs the scalable generation+verification pipeline.
