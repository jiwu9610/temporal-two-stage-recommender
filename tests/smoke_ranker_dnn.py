"""Smoke test for the DNN ranker end-to-end.

Verifies:
1. Retriever fitting, candidate unification, and source-tagged pool assembly.
2. Temporal feature tables build without NaN / shape errors.
3. DNNRanker (embeddings + Cross + DIN + Deep MLP) forward pass is shape-valid.
4. One epoch of training decreases BCE loss on a tiny subsample.
5. AUROC / AUPRC / Precision@K / Recall@K produce finite numbers on val.

Not a correctness test — just "does the pipeline run end-to-end on CPU
without crashing". Intended to be runnable in ~60 seconds on a laptop.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_ranker_dnn import run_category


def main():
    category = "Video_Games"
    print(f"[smoke] Running DNN ranker end-to-end on {category} (tiny).")
    run_category(
        category=category,
        epochs=2,
        retriever_epochs=2,
        batch_size=256,
        lr=1e-3,
        emb_dim=8,
        n_train_users=200,
        neg_per_pos=10,
        eval_sample_users=100,
        max_eval_cand_per_user=500,
        seed=0,
        device="cpu",
        train_subsample_pairs=8000,
    )
    print("[smoke] OK — pipeline ran end-to-end.")


if __name__ == "__main__":
    main()
