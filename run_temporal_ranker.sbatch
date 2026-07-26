#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=temporal_ranker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/temporal_ranker_%j.out
#SBATCH --error=logs/temporal_ranker_%j.err

# Full three-snapshot temporal ranker chain for one category:
#   data layer -> two-tower A/B/C -> candidates x3 -> selection/refit/test.
#
# Usage:
#   CAT=Video_Games sbatch run_temporal_ranker.sbatch
#   CAT=All_Beauty EPOCHS=6 sbatch run_temporal_ranker.sbatch

set -eo pipefail   # NOT -u: /etc/bashrc references BASHRCSOURCED unbound on this cluster
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
EPOCHS="${EPOCHS:-8}"

echo "=== [1/4] three-snapshot data layer ($CAT)"
python -m scripts.data.temporal_ranker_pipeline --category "$CAT"

echo "=== [2/4] periodic two-tower A/B/C ($CAT, dev epochs=$EPOCHS)"
python -m scripts.retrieval.temporal_two_tower --category "$CAT" \
    --snapshot all --epochs "$EPOCHS"

echo "=== [3/4] snapshot candidates x3 ($CAT)"
python -m scripts.ranker.temporal_candidate_builder --category "$CAT"

echo "=== [4/4] temporal ranker selection + refit + final test ($CAT)"
python -m scripts.ranker.train_temporal_ranker --category "$CAT"

echo "=== done: results/phase2_temporal/${CAT}_ranker.json"
