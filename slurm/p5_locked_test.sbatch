#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=p5_locked
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/p5_locked_%A_%a.out
#SBATCH --error=logs/p5_locked_%A_%a.err

# P5 -- the ONLY locked test of the rebuild wave (spec P5.1-P5.2). Runs ONLY
# after the P4.4 freeze commit (results/p4_freeze/frozen_config.json).
# Per category, two arms: {frozen winner, A0 post-C1 reference} -- the
# reference decouples "identity fix" from "new source" in the test-time
# attribution. Per arm:
#   [0] GPU inference-only re-emit of the frozen v2 test checkpoint at K=500
#       (skip-if-exists; shared across the two arms of a category)
#   [1] test-snapshot candidates for the arm's frozen retrieval config
#       (first and only test-window read of the wave)
#   [2] full ranker protocol: selection -> refit -> ONE locked test eval,
#       all_positive, with prediction dumps (results dir outside the locked
#       report dir by the trainer's own guard)
#   [3] calibration fit on the selection dump (test dump not opened)
#   [4] frozen chain -> test predictions, once
# Array index -> (category, arm):
#   0 Books A2 | 1 Books A0 | 2 Electronics A5 | 3 Electronics A0
#   4 Video_Games A0 (winner == reference; single arm) |
#   5 All_Beauty A1 | 6 All_Beauty A0

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

test -f results/p4_freeze/frozen_config.json || { echo "FATAL: P4.4 freeze missing"; exit 1; }
PAIRS=("Books A2" "Books A0" "Electronics A5" "Electronics A0" \
       "Video_Games A0" "All_Beauty A1" "All_Beauty A0")
read -r CAT V <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
CFG="configs/p4_stageA/${CAT}_${V}.json"
test -f "$CFG" || { echo "FATAL: $CFG missing"; exit 1; }
RD="results/p5_locked/${CAT}_${V}"
mkdir -p "$RD"

echo "=== CHECKPOINT: [0] test k500 emit ${CAT} $(date) ==="
python -u - <<PY
from pathlib import Path
from scripts.retrieval.temporal_two_tower import infer_snapshot_checkpoint
cat = "${CAT}"
out = Path("data/processed") / cat / "temporal_ranker" / "two_tower_predictions_test_k500.parquet"
if out.exists():
    print(f"[emit] {out} present -- skip", flush=True)
else:
    df = infer_snapshot_checkpoint(cat, "test", k=500, out_path=out)
    print(f"[emit] wrote {out} rows={len(df):,}", flush=True)
PY

echo "=== CHECKPOINT: [1] test candidates ${CAT} ${V} $(date) ==="
python -u -m scripts.ranker.temporal_candidate_builder --category "$CAT" \
    --variant "$V" --config-json "$CFG" --snapshots test

echo "=== CHECKPOINT: [2] full ranker + dump ${CAT} ${V} $(date) ==="
python -u -m scripts.ranker.train_temporal_ranker --category "$CAT" \
    --variant "$V" --label-mode all_positive --dump-predictions \
    --results-dir "$RD"

echo "=== CHECKPOINT: [3] calibration fit ${CAT} ${V} $(date) ==="
python -u -m scripts.evaluation.temporal_calibration --category "$CAT" \
    --variant "$V" --label-mode all_positive --stage fit

echo "=== CHECKPOINT: [4] LOCKED TEST ${CAT} ${V} $(date) ==="
python -u -m scripts.evaluation.temporal_calibration --category "$CAT" \
    --variant "$V" --label-mode all_positive --stage test

echo "=== CHECKPOINT: done ${CAT} ${V} $(date) ==="
echo "=== Job finished: $(date) ==="
