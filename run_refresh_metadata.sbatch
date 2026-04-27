#!/bin/bash
#SBATCH --account=PROJECT_ALLOC
#SBATCH --partition=normal
#SBATCH --job-name=refresh_meta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/refresh_meta_%j.out
#SBATCH --error=logs/refresh_meta_%j.err

# Re-pull full metadata + rebuild item_features.parquet + write per-category
# metadata_schema_report.json. Books streaming + schema-report on 4.4M rows
# of long-text columns needs >100 GB; use the normal partition with 180G.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-recsys}"

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Step 1: rebuild VG + Electronics from already-pulled metadata.
echo "--- step 1: rebuild Video_Games + Electronics (skip repull) ---"
python -m scripts.data.refresh_metadata --skip-repull --categories Video_Games Electronics
RC1=$?
echo "step 1 exit=$RC1"

# Step 2: full repull + rebuild Books, then schema reports for all 4 categories.
echo "--- step 2: full repull + rebuild Books ---"
python -m scripts.data.refresh_metadata --categories Books
RC2=$?
echo "step 2 exit=$RC2"

EXIT=$(( RC1 | RC2 ))
echo "=== Job finished: $(date), combined_exit=$EXIT ==="
exit $EXIT
