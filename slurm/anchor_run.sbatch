#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=anchor_run
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/anchor_run_%A_%a.out
#SBATCH --error=logs/anchor_run_%A_%a.err

# Track C anchor runs (REBUILD_WAVE_SPEC P1.2) — TEMPLATE, DO NOT SUBMIT until
# scripts/anchor/run_recbole_anchor.py + train_two_tower_anchor.py have landed
# and the diff has passed human review (GPU submission is gated on review).
#
# Array task -> category (one A100 / 8 CPU / 64G / 8h per category, per spec).
#
# Two phases per category, two different environments:
#   1. RecBole legs (Pop / ItemKNN / SASRec / SASRec-len20) run inside the
#      dedicated GPFS venv /projects/PROJECT_ALLOC/envs/recbole-anchor
#      (recbole==1.2.1; numpy-alias + torch.load weights_only=False shims are
#      pre-installed in its sitecustomize.py). Video_Games additionally runs
#      the official-split legs (leg E: official full-5-core SASRec, gate
#      R@10 in [0.070, 0.097]; leg F per run_recbole_anchor.py's leg registry)
#      against data/anchor/_official/Video_Games/*.inter (benchmark_filename
#      mode; downloaded + gated at setup by scripts.anchor.fetch_official_vg —
#      compute nodes have no internet, so the download MUST already be on disk).
#   2. Two-tower leg D runs in the production conda env (tae) via
#      train_two_tower_anchor.py (PROCESSED_DIR + RESULTS_DIR monkeypatched to
#      data/anchor + results/anchor, variant=anchor_v1).
#
# Vars:
#   RECBOLE_LEGS  default "pop,itemknn,sasrec,sasrec_len20" (+ ",official_sasrec,official_sasrec_len20" for VG)
#   RUN_TWO_TOWER default 1 (set 0 to skip leg D)
#   EXTRA         extra CLI args appended verbatim to run_recbole_anchor
#
# ISOLATION: everything written by this job goes to results/anchor/ only.
# Anchor artifacts must never enter production candidate/ranker pipelines.

set -eo pipefail   # NOT -u: /etc/bashrc references BASHRCSOURCED unbound on this cluster
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CATS=(All_Beauty Books Electronics Video_Games)
CAT="${CATS[$SLURM_ARRAY_TASK_ID]}"

RECBOLE_VENV=/projects/PROJECT_ALLOC/envs/recbole-anchor
RECBOLE_LEGS="${RECBOLE_LEGS:-pop,itemknn,sasrec,sasrec_len20}"
if [ "$CAT" = "Video_Games" ]; then
    RECBOLE_LEGS="${RECBOLE_LEGS},official_sasrec,official_sasrec_len20"
fi
RUN_TWO_TOWER="${RUN_TWO_TOWER:-1}"
EXTRA="${EXTRA:-}"

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  ARRAY_TASK: $SLURM_ARRAY_TASK_ID  CAT: $CAT"
echo "RECBOLE_LEGS=$RECBOLE_LEGS RUN_TWO_TOWER=$RUN_TWO_TOWER EXTRA=[$EXTRA]"
nvidia-smi || true

# Preflight: anchor exports + (for VG) official benchmark files must exist.
test -f "data/anchor/${CAT}/${CAT}.inter" || { echo "missing data/anchor/${CAT}/${CAT}.inter — run scripts.anchor.export_anchor_splits first"; exit 1; }
if [ "$CAT" = "Video_Games" ]; then
    test -f "data/anchor/_official/Video_Games/Video_Games.test.inter" || { echo "missing official benchmark .inter — run scripts.anchor.fetch_official_vg on the login node first"; exit 1; }
fi

# ---- Phase 1: RecBole legs (dedicated venv) --------------------------------
source "${RECBOLE_VENV}/bin/activate"
python -c "import recbole, torch; print('recbole', recbole.__version__, '| torch', torch.__version__, '| cuda?', torch.cuda.is_available())"
python -u -m scripts.anchor.run_recbole_anchor \
    --category "$CAT" \
    --legs "$RECBOLE_LEGS" \
    $EXTRA
deactivate

# ---- Phase 2: two-tower leg D (production conda env, isolated outputs) -----
if [ "$RUN_TWO_TOWER" = "1" ]; then
    source ~/.bashrc
    conda activate "${CONDA_ENV:-tae}"
    python -u -m scripts.anchor.train_two_tower_anchor --category "$CAT"
    python -u -m scripts.anchor.score_predictions --category "$CAT"
fi

echo "=== Job finished: $(date) ==="
