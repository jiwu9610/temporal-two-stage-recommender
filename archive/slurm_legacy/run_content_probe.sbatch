#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=normal
#SBATCH --job-name=content_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/content_probe_%j.out
#SBATCH --error=logs/content_probe_%j.err

# P1.4 pre-C1 go/no-go probe: content_i2i on the existing (poisoned)
# item_text_embeddings.npz. CPU-only, one job per category (parallel):
#   CAT=All_Beauty  sbatch run_content_probe.sbatch     (~minutes)
#   CAT=Video_Games sbatch run_content_probe.sbatch     (~minutes-1h)
#   CAT=Electronics sbatch run_content_probe.sbatch     (~1-3h; 2.4G matrix)
#   CAT=Books       sbatch run_content_probe.sbatch     (~4-6h; 5.9G matrix)
# Output: results/phase3a/${CAT}_content_probe.json
# Gate (spec P1.4): Books/Electronics content unique-hit substantively > 0
# => content source proceeds; ~0 => stop and re-review the embedding recipe.
# NO freeze decision either way.

set -eo pipefail   # NOT -u: /etc/bashrc references BASHRCSOURCED unbound on this cluster
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.bashrc
conda activate "${CONDA_ENV:-tae}"

CAT="${CAT:?must set CAT}"
CONFIGS="${CONFIGS:-A0,A1,A5}"
TOPK="${TOPK:-500}"

python -m scripts.retrieval.content_probe --category "$CAT" \
    --configs "$CONFIGS" --top-k "$TOPK"
echo "=== done: results/phase3a/${CAT}_content_probe.json"
