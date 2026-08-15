#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=p4_stageB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p4_stageB_%A_%a.out
#SBATCH --error=logs/p4_stageB_%A_%a.err

# P4.3 Stage B: ranker retrain on the Stage-A shortlist, selection-only
# (--stage-b: the test snapshot is neither read nor required to exist --
# freeze P4.4 happens strictly before any test read). all_positive protocol.
# Decision metric: selection-snapshot ranker R@100 macro (primary) plus the
# R@10 no-regression guardrail vs A0 -- hence A0 is retrained here too as
# the reference arm. Memory: 128G per the corrected allpos anchor
# (~1.1-1.2 KB/row; Books A3 projects ~116 GiB raw -- allpos precedent
# completed at 126.9/128).
# Pairs are appended per category as Stage-A gates land:
#   0-2 Books {A3,A2,A0}, 3-4 Electronics {A5,A0}; VG/AB pairs go in a
#   second submission once their sweeps finish.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"

# VG content rejected (136-137G projection > 128G hard wall); A0/A5 run on
# the empirical precedent (pre-C1 allpos: 126.9/128 GiB COMPLETED at MORE
# rows than the new A0) despite the 120G soft gate -- baseline is mandatory.
PAIRS=("Books A3" "Books A2" "Books A0" "Electronics A5" "Electronics A0" \
       "Video_Games A5" "Video_Games unused" "Video_Games A0" \
       "All_Beauty A3" "All_Beauty A1" "All_Beauty A0")
read -r CAT VARIANT <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"
case "$VARIANT" in __*__) echo "placeholder $VARIANT not resolved"; exit 1;; esac

echo "=== CHECKPOINT: stageB ${CAT} ${VARIANT} start $(date) ==="
python -u -m scripts.ranker.train_temporal_ranker \
    --category "$CAT" --variant "$VARIANT" \
    --label-mode all_positive --stage-b
echo "=== CHECKPOINT: stageB ${CAT} ${VARIANT} done $(date) ==="
echo "=== Job finished: $(date) ==="
