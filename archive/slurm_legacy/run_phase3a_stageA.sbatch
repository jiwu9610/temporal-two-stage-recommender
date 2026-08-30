#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=phase3a_stageA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase3a_stageA_%j.out
#SBATCH --error=logs/phase3a_stageA_%j.err

# Phase 3A Stage A: retrieval-only sweep on the model-selection snapshot.
# Usage:  CAT=Video_Games sbatch run_phase3a_stageA.sbatch
#         CAT=Books MAXK=1000 sbatch run_phase3a_stageA.sbatch

set -eo pipefail   # NOT -u: /etc/bashrc references BASHRCSOURCED unbound on this cluster
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
MAXK="${MAXK:-500}"

python -m scripts.retrieval.phase3a_sweep --category "$CAT" --max-k "$MAXK"
echo "=== done: results/phase3a/${CAT}_stageA.json"
