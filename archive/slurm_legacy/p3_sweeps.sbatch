#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=p3_sweep
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/p3_sweep_%A_%a.out
#SBATCH --error=logs/p3_sweep_%A_%a.err

# P3.3 Stage-2 1-D sweeps around the arm3 winner (softmax + taste + AdamW +
# recency90). STRICTLY one dimension per run; dev snapshot only. Spec's "MNS
# tail {0}" arm is skipped: the knob is not CLI-exposed (documented deviation).
# Spec gate: if pair-decay=90 wins somewhere, check it is not an M1-ramp/dev-
# window interaction before freezing it.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results/p3_dev
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

CATS=(All_Beauty Video_Games Books Electronics)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"
TDIR="data/processed/${CAT}/temporal_ranker"
OUT="results/p3_dev"
BASE="--loss-mode softmax --use-hist-pool --optimizer adamw"

run_sweep () {
  TAG="$1"; shift
  echo "=== CHECKPOINT: ${CAT} / ${TAG} start ==="
  python -u -m scripts.retrieval.temporal_two_tower \
      --category "$CAT" --snapshot ranker_train --epochs 15 $BASE "$@"
  mv "$TDIR/two_tower_ranker_train_meta.json" "$OUT/${CAT}_${TAG}_meta.json"
  mv "$TDIR/two_tower_frozen_config.json"     "$OUT/${CAT}_${TAG}_frozen.json"
  rm -f "$TDIR/two_tower_ranker_train.pt" \
        "$TDIR/two_tower_predictions_ranker_train.parquet"
  echo "=== CHECKPOINT: ${CAT} / ${TAG} done ==="
}

# base arm3 = pair-decay 90, lambda 0, uniform pooling, ids on. One axis each:
run_sweep sw_decay_none
run_sweep sw_decay_365   --pair-recency-decay-days 365
run_sweep sw_lambda_05   --pair-recency-decay-days 90 --hard-negative-lambda 0.5
run_sweep sw_lambda_2    --pair-recency-decay-days 90 --hard-negative-lambda 2.0
run_sweep sw_poolrec_365 --pair-recency-decay-days 90 --hist-pool-recency-days 365
run_sweep sw_ids_off     --pair-recency-decay-days 90 --variant metadata_only

echo "=== Job finished: $(date) ==="
