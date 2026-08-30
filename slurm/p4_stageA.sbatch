#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=normal
#SBATCH --job-name=p4_stageA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p4_stageA_%A_%a.out
#SBATCH --error=logs/p4_stageA_%A_%a.err

# Stage A: retrieval-only 6-config sweep on the rebuilt data layer.
#   A0  thr5, no content   == frozen C2_k500_recent rebuilt post-C1 (P4.1)
#   A1  thr5 + content(catalog)
#   A2  thr3 + content(catalog)
#   A3  thr2 + content(catalog)
#   A4  thr3 + content(history,min1)  -- dilution control
#   A5  thr3, no content              -- threshold-attribution control
# VG/Electronics content source carries top_k=200 in its config JSON (D8).
# Snapshots: ranker_train + model_selection ONLY (test stays locked).
# A0 runs first: it re-tunes rule weights on the new T0 window (the pre-C1
# rule_weights.json was archived) and every later config reuses the frozen
# file. The k500 two-tower parquets come from slurm/p4_k500_emit.sbatch (GPU);
# the guard below fails fast instead of silently burning hours of CPU
# regeneration. Array index -> category matches slurm/p3_freeze.sbatch.
# Probe-calibrated budget/category: AB ~2h, VG ~3h, Elec ~6h, Books ~8h.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

CATS=(All_Beauty Video_Games Books Electronics)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"

for s in ranker_train model_selection; do
    f="data/processed/${CAT}/temporal_ranker/two_tower_predictions_${s}_k500.parquet"
    test -f "$f" || { echo "FATAL: missing $f (run p4_k500_emit first)"; exit 1; }
done

# 2026-08-21 REBUILD on the ALL-POSITIVE frame (project objective). The
# first run of this sweep (Aug 14) used the builder default = first-positive
# frame: labels, coverage ceilings and rule-weight tuning all answered the
# next-item question. Variants are written as {V}_ap so nothing is shared
# with the first-positive artifacts; rule weights go to
# rule_weights_all_positive.json.
LM="${LABEL_MODE:-all_positive}"; SUF="${VARIANT_SUFFIX:-_ap}"
for V in A0 A1 A2 A3 A4 A5; do
    echo "=== CHECKPOINT: stageA ${CAT} ${V}${SUF} (${LM}) start $(date) ==="
    python -u -m scripts.ranker.temporal_candidate_builder \
        --category "$CAT" --variant "${V}${SUF}" --label-mode "$LM" \
        --config-json "configs/p4_stageA/${CAT}_${V}.json" \
        --snapshots ranker_train,model_selection
    echo "=== CHECKPOINT: stageA ${CAT} ${V}${SUF} done $(date) ==="
done
echo "=== Job finished: $(date) ==="
