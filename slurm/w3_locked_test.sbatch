#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=w3_locked
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/w3_locked_%A_%a.out
#SBATCH --error=logs/w3_locked_%A_%a.err

# Wave-3 LOCKED TEST: one full run per frozen seq winner
# (results/wave3_seq/frozen_winners.json; freeze commit 655974f precedes this
# read). Grid = the frozen spec only (--seq-only); lr/epoch chosen on
# model_selection; refit; ONE test eval. Candidates = the P4.4-frozen
# variants whose test snapshots were built in P5. Elec/AB: no seq module.
#   0 Books A2 mh_pool:pos=delta:L=2 | 1 Video_Games A0 hstu:pos=delta:d=64

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

test -f results/wave3_seq/frozen_winners.json || { echo "FATAL: wave-3 freeze missing"; exit 1; }
PAIRS=("Books A2 seq:mh_pool:pos=delta:L=2" "Video_Games A0 seq:hstu:pos=delta:d=64")
read -r CAT VAR SPEC <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
RD="results/wave3_seq/locked_test/${CAT}"
mkdir -p "$RD"
# Sibling arrays on gengpu / gengpu-long race the same arms. The LOCKED TEST
# may be executed exactly once per arm: mkdir is the atomic claim; the loser
# (or a rerun after a report exists) exits without ever reading the test
# snapshot. A claim without a report (crashed run) must be investigated
# manually, NOT auto-cleared -- the test read may already have happened.
CLAIM="${RD}/.claim"
if ls "${RD}"/${CAT}_ranker_*.json >/dev/null 2>&1; then
    echo "=== ${CAT}: locked-test report already exists -- skip ==="; exit 0
fi
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "=== ${CAT}: another task holds the locked-test claim -- skip ==="; exit 0
fi
HEADREF=$(cat .git/HEAD); case "$HEADREF" in ref:*) COMMIT=$(cat ".git/${HEADREF#ref: }");; *) COMMIT=$HEADREF;; esac
echo "=== code commit: $COMMIT ==="

echo "=== CHECKPOINT: w3 LOCKED TEST ${CAT} ${VAR} ${SPEC} start $(date) ==="
python -u -m scripts.ranker.train_temporal_ranker --category "$CAT" \
    --variant "$VAR" --label-mode all_positive \
    --seq-arm --seq-only --seq-grid "$SPEC" --results-dir "$RD"
echo "=== CHECKPOINT: done ${CAT} $(date) ==="
echo "=== Job finished: $(date) ==="
