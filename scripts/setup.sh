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

# Remember: `sml init` (once) configures the slurm launcher + CSCS API key,
# then `./scripts/run_poc.sh` launches the PoC.
