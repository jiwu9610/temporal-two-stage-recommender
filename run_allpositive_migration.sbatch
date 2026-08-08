#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=allpos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/allpos_%j.out
#SBATCH --error=logs/allpos_%j.err

# Objective migration: first-positive -> all-positive ground truth.
#
# The research question is "did the recommender cover what the user actually
# bought over the next window", but the frozen protocol labels only each user's
# FIRST positive and feeds the rest to the ranker as label=0. This retrains the
# ranker against the corrected objective and refits calibration on top.
#
# What is NOT rerun: the two-tower. Its training pairs come from `history`, not
# from the groundtruth frame, so its representation is label-independent. Its
# frozen epoch counts were selected on single-positive dev recall, but an audit
# found 3 of 4 of those counts sit inside run-to-run noise, so re-selecting
# would trade a documented arbitrary choice for an undocumented one while
# invalidating the freeze. Left frozen deliberately.
#
# Nothing frozen is overwritten: candidates come from the *_allpos variant
# built by relabel_candidates_all_positive, prediction dumps land beside them,
# and every report is tagged with the new variant name.
#
# Usage: CAT=Video_Games sbatch run_allpositive_migration.sbatch

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results/label_modes/ranker_allpos
source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
VARIANT="${VARIANT:-C2_k500_recent_allpos}"
MODE=all_positive

CANDS="data/processed/${CAT}/temporal_ranker/variants/${VARIANT}"
for s in ranker_train model_selection test; do
    [ -f "${CANDS}/candidates_${s}.parquet" ] || {
        echo "missing ${CANDS}/candidates_${s}.parquet -- run relabel first" >&2
        exit 1; }
    [ -f "data/processed/${CAT}/temporal_ranker/groundtruth_all_${s}.parquet" ] || {
        echo "missing groundtruth_all_${s}.parquet -- run make_all_positive_groundtruth" >&2
        exit 1; }
done

echo "=== [1/3] ranker: selection -> refit -> single locked test ($CAT / $VARIANT / $MODE)"
python -m scripts.ranker.train_temporal_ranker --category "$CAT" \
    --variant "$VARIANT" --label-mode "$MODE" --dump-predictions \
    --results-dir results/label_modes/ranker_allpos

echo "=== [2/3] calibration fit (model_selection only; test dump untouched)"
python -m scripts.evaluation.temporal_calibration --category "$CAT" \
    --variant "$VARIANT" --label-mode "$MODE" --stage fit

echo "=== [3/3] frozen chain -> test (single locked application)"
python -m scripts.evaluation.temporal_calibration --category "$CAT" \
    --variant "$VARIANT" --label-mode "$MODE" --stage test

echo "=== done"
echo "  ranker : results/label_modes/ranker_allpos/${CAT}_ranker_${VARIANT}.json"
echo "  calib  : results/calibration/${CAT}_${VARIANT}_{fit,frozen,test}.json"
