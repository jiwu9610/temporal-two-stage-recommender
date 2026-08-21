#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=w3_seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/w3_seq_%A_%a.out
#SBATCH --error=logs/w3_seq_%A_%a.err

# Wave 3 selection sweep (advisor spec docs/advanced_seq.pdf): sequence module
# at the ranking position, selection-only (--stage-b, test never read).
# Each array task = one (category, seq spec); the grid inside = the 4 frozen
# archs (mlp/deep_cross x 2 lr) + the seq spec x 2 lr, so every report shows
# whether the seq arm beats the frozen ranker on the SAME candidates.
# Candidates = P4.4 frozen per-category retrieval config
# (Books A2 / Electronics A5 / Video_Games A0). AB stays popularity baseline.
# Index = cat_idx * N_SPECS + spec_idx; specs in configs/wave3_seq_grid.txt.
# Model = DCN branch (frozen DeepCrossConfig) || seq module -> overarch, so
# the in-run deep_cross arm is the exact seq-off ablation.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

mapfile -t SPECS < configs/wave3_seq_grid.txt
N=${#SPECS[@]}
CATS=(Books Electronics Video_Games); VARS=(A2 A5 A0)
CI=$(( SLURM_ARRAY_TASK_ID / N )); SI=$(( SLURM_ARRAY_TASK_ID % N ))
CAT=${CATS[$CI]}; VAR=${VARS[$CI]}; SPEC=${SPECS[$SI]}
TAG=$(echo "$SPEC" | tr ':=' '__')
RD="results/wave3_seq/${CAT}"
mkdir -p "$RD"
# Two sibling arrays (a100-only + any-gpu) race the same index space; the
# report file is the lock -- if it exists (or another task claimed it) skip.
CLAIM="${RD}/.claim_${TAG}"
if ls "${RD}"/${CAT}_stageB_${VAR}_${TAG}.json >/dev/null 2>&1 || ! mkdir "$CLAIM" 2>/dev/null; then
    echo "=== ${CAT} ${SPEC} already done/claimed -- skip ==="; exit 0
fi
trap 'rmdir "$CLAIM" 2>/dev/null' EXIT

HEADREF=$(cat .git/HEAD); case "$HEADREF" in ref:*) COMMIT=$(cat ".git/${HEADREF#ref: }");; *) COMMIT=$HEADREF;; esac
echo "=== code commit: $COMMIT ==="
echo "=== CHECKPOINT: w3 ${CAT} ${VAR} ${SPEC} start $(date) ==="
python -u -m scripts.ranker.train_temporal_ranker --category "$CAT" \
    --variant "$VAR" --label-mode all_positive --stage-b \
    --seq-arm --seq-grid "$SPEC" --results-dir "$RD"
# report lands in ${RD}/${CAT}_stageB_${VAR}_<spec tag>.json (results_dir-aware)
ls -1 "${RD}"/${CAT}_stageB_${VAR}_*.json | tail -1
echo "=== CHECKPOINT: w3 ${CAT} ${VAR} ${SPEC} done $(date) ==="
echo "=== Job finished: $(date) ==="
