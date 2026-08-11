#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p2_%x_%j.out
#SBATCH --error=logs/p2_%x_%j.err

# P2 rebuild trunk, one STAGE per submission (driven by STAGE + CAT env vars):
#   STAGE=archive            rsync pre-C1 artifacts to data/archive_preC1/
#                            (excludes variants/ -- frozen candidate parquets
#                            are not overwritten by the rebuild and stay put)
#   STAGE=phase0   CAT=...   Phase 0 rebuild under canon v2 (9d1baf8)
#   STAGE=snapshots CAT=...  three-snapshot walk-forward rebuild (pinned cutoffs,
#                            incl. the approved AB t3 exception)
#   STAGE=embed              regenerate item_text_embeddings on the new identity
#                            (submit with --partition=gengpu --gres=gpu:a100:1)
# Chaining is done at submission time via --dependency=afterok.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

STAGE="${STAGE:?must set STAGE}"

case "$STAGE" in
  archive)
    DEST=data/archive_preC1
    mkdir -p "$DEST"
    for CAT in All_Beauty Video_Games Books Electronics; do
      rsync -a --exclude 'variants/' "data/processed/$CAT/" "$DEST/$CAT/"
    done
    # Compute nodes have no git binary (exit 127 killed job 9075594 here,
    # AFTER the rsyncs). Read the commit straight out of .git instead.
    REF=$(awk '{print $2}' .git/HEAD)
    { cat ".git/$REF" 2>/dev/null || cat .git/HEAD; } > "$DEST/ARCHIVED_AT_COMMIT"
    date -u +%FT%TZ >> "$DEST/ARCHIVED_AT_COMMIT"
    du -sh "$DEST"/*
    ;;
  phase0)
    CAT="${CAT:?must set CAT}"
    python -m scripts.data.preprocessing_pipeline --category "$CAT" --clean-phase0
    ;;
  snapshots)
    CAT="${CAT:?must set CAT}"
    python -m scripts.data.temporal_ranker_pipeline --category "$CAT"
    ;;
  embed)
    python -m scripts.data.build_item_text_embeddings
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 1;;
esac
echo "=== P2/$STAGE ${CAT:-} done"
