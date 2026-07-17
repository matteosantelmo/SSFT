#!/bin/bash
#SBATCH --job-name=prepare-data
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_data-%j.out
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=reasoning_fixed
set -euo pipefail

REPO="/iopsstor/scratch/cscs/msantelmo/SSFT"
cd "$REPO"

bash "$REPO/scripts/data/prepare_data.sh"
