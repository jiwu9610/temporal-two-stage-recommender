#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=phase2_ranker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_ranker_%j.out
#SBATCH --error=logs/phase2_ranker_%j.err

# End-to-end Phase 2 pipeline for one category:
#   1. candidate_builder.py  -> data/processed/{cat}/candidates.parquet
#   2. train_mlp_ranker.py    -> results/phase2/{cat}_mlp_ranker.json
#   3. train_complex_ranker.py-> results/phase2/{cat}_complex_ranker.json
#
# Usage:
#   CAT=Video_Games sbatch run_phase2_ranker.sbatch
#   CAT=Books TOPK=50 EPOCHS=15 sbatch run_phase2_ranker.sbatch
#
# Vars:
#   CAT      (required)   All_Beauty | Video_Games | Books | Electronics
#   TOPK     default 100
#   EPOCHS   default 20
#   VARIANT  default metadata_and_ids   (two-tower variant to consume)
#
# `set -e` (NOT -u): /etc/bashrc references BASHRCSOURCED unbound on this cluster.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
TOPK="${TOPK:-100}"
EPOCHS="${EPOCHS:-100}"
VARIANT="${VARIANT:-metadata_and_ids}"

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CAT=$CAT TOPK=$TOPK EPOCHS=$EPOCHS VARIANT=$VARIANT"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

CANDS_PATH="data/processed/${CAT}/candidates.parquet"
if [ "${SKIP_BUILDER:-0}" = "1" ] && [ -f "$CANDS_PATH" ]; then
    echo "--- step 1: SKIP candidate_builder (SKIP_BUILDER=1, $CANDS_PATH exists) ---"
    RC1=0
else
    echo "--- step 1: candidate_builder ---"
    python -u -m scripts.ranker.candidate_builder \
        --category "$CAT" \
        --top-k "$TOPK" \
        --two-tower-variant "$VARIANT"
    RC1=$?
fi
echo "step 1 exit=$RC1"
[ "$RC1" -ne 0 ] && exit "$RC1"

echo "--- step 2: train_mlp_ranker ---"
python -u -m scripts.ranker.train_mlp_ranker \
    --category "$CAT" \
    --epochs "$EPOCHS"
RC2=$?
echo "step 2 exit=$RC2"

echo "--- step 3: train_complex_ranker ---"
python -u -m scripts.ranker.train_complex_ranker \
    --category "$CAT" \
    --epochs "$EPOCHS"
RC3=$?
echo "step 3 exit=$RC3"

EXIT=$(( RC2 | RC3 ))
echo "=== Job finished: $(date), combined_exit=$EXIT ==="
exit $EXIT
