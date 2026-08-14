#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=p3_dev
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p3_dev_%A_%a.out
#SBATCH --error=logs/p3_dev_%A_%a.err

# P3.2 Stage-1 dev arms (spec + review correction #7): decouple the loss
# rewrite from the taste channel before any freeze. THREE arms per category,
# dev snapshot (ranker_train) ONLY, early stopping allowed here and nowhere
# else. Dev artifacts are moved to results/p3_dev/ after each arm so arms do
# not clobber each other and the production paths stay clean for P3.4.
#
#   arm1_bce_base       v1 turned on the new data layer (anchors the delta)
#   arm2_bce_taste      + hist-pool taste channel + AdamW(no-decay-emb) + recency
#   arm3_softmax_taste  arm2 with loss_mode=softmax (in-batch + logQ)

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results/p3_dev
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

CATS=(All_Beauty Video_Games Books Electronics)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
TDIR="data/processed/${CAT}/temporal_ranker"
OUT="results/p3_dev"

run_arm () {
  ARM="$1"; shift
  echo "=== CHECKPOINT: ${CAT} / ${ARM} start ==="
  python -u -m scripts.retrieval.temporal_two_tower \
      --category "$CAT" --snapshot ranker_train --epochs 15 "$@"
  mv "$TDIR/two_tower_ranker_train_meta.json"    "$OUT/${CAT}_${ARM}_meta.json"
  mv "$TDIR/two_tower_frozen_config.json"        "$OUT/${CAT}_${ARM}_frozen.json"
  rm -f "$TDIR/two_tower_ranker_train.pt" \
        "$TDIR/two_tower_predictions_ranker_train.parquet"
  echo "=== CHECKPOINT: ${CAT} / ${ARM} done ==="
}

run_arm arm1_bce_base
run_arm arm2_bce_taste   --use-hist-pool --optimizer adamw --pair-recency-decay-days 90
run_arm arm3_softmax_taste --loss-mode softmax --use-hist-pool --optimizer adamw \
                           --pair-recency-decay-days 90

echo "=== Job finished: $(date) ==="
