#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=phase3a_stageB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase3a_stageB_%j.out
#SBATCH --error=logs/phase3a_stageB_%j.err

# Phase 3A Stage B: build variant candidates (ranker_train + model_selection
# ONLY -- the test snapshot stays locked) and run the ranker in --stage-b mode.
# Usage:  CAT=Video_Games VARIANT=C2_k500_recent sbatch run_phase3a_stageB.sbatch

set -eo pipefail   # NOT -u: /etc/bashrc references BASHRCSOURCED unbound on this cluster
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
VARIANT="${VARIANT:?must set VARIANT}"
CFG="configs/phase3a/${CAT}_${VARIANT}.json"

echo "=== [1/2] variant candidates ($CAT / $VARIANT, train+selection only)"
python -m scripts.ranker.temporal_candidate_builder --category "$CAT" \
    --variant "$VARIANT" --config-json "$CFG" \
    --snapshots ranker_train,model_selection

echo "=== [2/2] ranker stage-B confirmation ($CAT / $VARIANT)"
python -m scripts.ranker.train_temporal_ranker --category "$CAT" \
    --variant "$VARIANT" --stage-b

echo "=== done: results/phase3a/${CAT}_stageB_${VARIANT}.json"
