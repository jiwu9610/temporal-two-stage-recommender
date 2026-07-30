#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=calib_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/calib_eval_%j.out
#SBATCH --error=logs/calib_eval_%j.err

# Evaluation & Calibration (advisor spec 2026-07-26).
# [1] Rerun the frozen C2_k500_recent ranker protocol with --dump-predictions.
#     Identical config + seed; the report goes to results/calibration/
#     ranker_rerun so the locked Phase 3A test report is never overwritten.
# [2] Fit calibration (prior/Platt + per-cohort thresholds) on the SELECTION
#     model's out-of-sample model_selection predictions; freeze the config.
#     The test dump is not read at this stage (enforced in code).
# [3] Single locked application of the frozen chain to the test predictions.
# Usage: CAT=Video_Games sbatch run_calibration_eval.sbatch

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
VARIANT="${VARIANT:-C2_k500_recent}"

echo "=== [1/3] ranker rerun with prediction dump ($CAT / $VARIANT)"
python -m scripts.ranker.train_temporal_ranker --category "$CAT" \
    --variant "$VARIANT" --dump-predictions \
    --results-dir results/calibration/ranker_rerun

echo "=== [2/3] calibration fit (model_selection only)"
python -m scripts.evaluation.temporal_calibration --category "$CAT" \
    --variant "$VARIANT" --stage fit

echo "=== [3/3] frozen chain -> test (single locked application)"
python -m scripts.evaluation.temporal_calibration --category "$CAT" \
    --variant "$VARIANT" --stage test

echo "=== done: results/calibration/${CAT}_${VARIANT}_{fit,frozen,test}.json"
