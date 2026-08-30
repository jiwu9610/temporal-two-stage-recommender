#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=normal
#SBATCH --job-name=p4_gate_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p4_gate_full_%j.out
#SBATCH --error=logs/p4_gate_full_%j.err
# Full-parquet Stage-A gate pass (per-source R@500, content unique-hit share,
# hit-by-item-history buckets). Report-only shortlists already ran inline;
# this adds the full report detail. A login node cannot hold the 40-54M-row
# candidate parquets (6G cgroup).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
set +e; source ~/.bashrc; set -e
conda activate "${CONDA_ENV:-tae}"
for CAT in "$@"; do
  echo "=== CHECKPOINT: gate full ${CAT} start $(date) ==="
  python -u -m scripts.analysis.p4_stageA_gate --category "$CAT"
  echo "=== CHECKPOINT: gate full ${CAT} done $(date) ==="
done
