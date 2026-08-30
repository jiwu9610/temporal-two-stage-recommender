#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=phase3a_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase3a_final_%j.out
#SBATCH --error=logs/phase3a_final_%j.err

# Phase 3A LOCKED FINAL TEST. Runs ONLY after frozen_config.json exists.
# Builds the test-snapshot candidates for the frozen variant (train/selection
# candidates already exist from Stage B) and runs the full ranker protocol
# (selection -> refit -> single test eval).
# Usage: CAT=Video_Games sbatch run_phase3a_final.sbatch

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
VARIANT=C2_k500_recent
test -f results/phase3a/frozen_config.json || { echo "frozen config missing"; exit 1; }

echo "=== [1/2] test-snapshot candidates ($CAT / $VARIANT)"
python -m scripts.ranker.temporal_candidate_builder --category "$CAT" \
    --variant "$VARIANT" --config-json "configs/phase3a/${CAT}_${VARIANT}.json" \
    --snapshots test

echo "=== [2/2] full ranker: selection -> refit -> LOCKED single test eval"
python -m scripts.ranker.train_temporal_ranker --category "$CAT" --variant "$VARIANT"

echo "=== done: results/phase2_temporal/${CAT}_ranker_${VARIANT}.json"
