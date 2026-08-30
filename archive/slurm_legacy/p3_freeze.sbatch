#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=p3_freeze
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/p3_freeze_%A_%a.out
#SBATCH --error=logs/p3_freeze_%A_%a.err

# P3.4 freeze wave: Scheme B with the Stage-1/Stage-2 winner.
#   config (all categories): loss_mode=softmax, use_hist_pool, AdamW
#   (zero-decay embeddings), pair_recency_decay_days=90, uniform pooling,
#   ids on. Books additionally hard_negative_lambda=0.5 (the only category
#   where the 1-D sweep showed a consistent gain, +8% across both lambda
#   values; per-category freeze approved as decision D2).
# --snapshot all: dev early-stop picks epochs -> frozen -> model_selection and
# test retrain FROM SCRATCH on the frozen schedule; config_hash asserted
# identical across the three checkpoints; predictions parquet emitted per
# snapshot. Sweep artifacts from results/p3_dev are NOT reused: the freeze
# wave retrains everything from scratch on the production paths.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

CATS=(All_Beauty Video_Games Books Electronics)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
EXTRA=""
[ "$CAT" = "Books" ] && EXTRA="--hard-negative-lambda 0.5"

echo "=== CHECKPOINT: freeze wave ${CAT} start ==="
python -u -m scripts.retrieval.temporal_two_tower \
    --category "$CAT" --snapshot all --epochs 15 \
    --loss-mode softmax --use-hist-pool --optimizer adamw \
    --pair-recency-decay-days 90 $EXTRA
echo "=== CHECKPOINT: freeze wave ${CAT} done ==="
echo "=== Job finished: $(date) ==="
