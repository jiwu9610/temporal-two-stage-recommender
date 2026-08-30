#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=two_tower
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/two_tower_%j.out
#SBATCH --error=logs/two_tower_%j.err

# Phase 1 two-tower v1 trainer launcher.
#
# Usage:
#   CAT=Video_Games VARIANT=metadata_and_ids sbatch run_two_tower.sbatch
#   CAT=Books      VARIANT=metadata_only     EXTRA="--epochs 8" sbatch run_two_tower.sbatch
#
# Vars:
#   CAT       (required)   All_Beauty | Video_Games | Books | Electronics
#   VARIANT   (required)   metadata_only | metadata_and_ids
#   EPOCHS    default 5
#   DIM       default 64
#   N_NEG     default 4
#   EXTRA     extra CLI args appended verbatim (e.g. "--use-hard-negatives")

set -eo pipefail   # NOT -u: /etc/bashrc references BASHRCSOURCED unbound on this cluster
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
VARIANT="${VARIANT:?must set VARIANT}"
EPOCHS="${EPOCHS:-5}"
DIM="${DIM:-64}"
N_NEG="${N_NEG:-4}"
EXTRA="${EXTRA:-}"

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CAT=$CAT VARIANT=$VARIANT EPOCHS=$EPOCHS DIM=$DIM N_NEG=$N_NEG EXTRA=[$EXTRA]"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -u -m scripts.retrieval.train_two_tower \
    --category "$CAT" \
    --variant "$VARIANT" \
    --epochs "$EPOCHS" \
    --embedding-dim "$DIM" \
    --num-soft-negatives "$N_NEG" \
    $EXTRA

EXIT=$?
echo "=== Job finished: $(date), exit=$EXIT ==="
exit $EXIT
