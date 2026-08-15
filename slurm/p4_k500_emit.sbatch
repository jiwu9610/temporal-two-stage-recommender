#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=p4_k500_emit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/p4_k500_emit_%A_%a.out
#SBATCH --error=logs/p4_k500_emit_%A_%a.err

# P4 prologue: inference-only re-emit of the frozen v2 two-tower checkpoints
# at K=500 for the Stage-A sweep (ranker_train + model_selection ONLY; the
# test-snapshot emit happens after the P4.4 freeze, exactly like Phase 3A).
# GPU because the Books catalog (4.2M items) makes CPU scoring infeasible;
# the CPU sweep then finds the parquet and skips regeneration (the builder's
# generate-if-missing contract). NO training, NO selection, NO test labels.
# Array index -> category matches slurm/p3_freeze.sbatch.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

CATS=(All_Beauty Video_Games Books Electronics)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"

echo "=== CHECKPOINT: k500 emit ${CAT} start ==="
python -u - <<PY
from pathlib import Path
from scripts.retrieval.temporal_two_tower import infer_snapshot_checkpoint

cat = "${CAT}"
tdir = Path("data/processed") / cat / "temporal_ranker"
for snap in ["ranker_train", "model_selection"]:
    out = tdir / f"two_tower_predictions_{snap}_k500.parquet"
    if out.exists():
        print(f"[emit] {out} already present -- skip", flush=True)
        continue
    print(f"[emit] {cat}/{snap} K=500 from frozen checkpoint...", flush=True)
    df = infer_snapshot_checkpoint(cat, snap, k=500, out_path=out)
    print(f"[emit] wrote {out} rows={len(df):,} users={df['user_id'].nunique():,}",
          flush=True)
PY
echo "=== CHECKPOINT: k500 emit ${CAT} done ==="
echo "=== Job finished: $(date) ==="
