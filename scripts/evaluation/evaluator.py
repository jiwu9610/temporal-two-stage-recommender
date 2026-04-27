"""
Main Evaluator class — single entry point for all recommendation metrics.

Wraps the individual metric functions from metrics.py into a convenient
callable that computes everything in one shot. This is what the training
loop and evaluation scripts should use.

Usage:
    evaluator = Evaluator(ks=[5, 10, 20])

    # Prepare inputs as padded tensors:
    #   predictions: [B, max_len] — model scores (higher = more relevant)
    #   labels:      [B, max_len] — binary ground truth (1 = user interacted)
    #   lengths:     [B]          — number of real items per user (rest is padding)
    metrics = evaluator(predictions, labels, lengths)
    # Returns: {"auc": 0.82, "gauc": 0.79, "mrr": 0.45, "ndcg@5": 0.38,
    #           "ndcg@10": ..., "precision@5": ..., "recall@5": ..., ...}
"""

import torch
from typing import Dict, List

from .metrics import (
    make_mask,
    auc_per_sample,
    global_auc,
    gauc,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class Evaluator:
    """Vectorized recommendation evaluator with jagged tensor support.

    Computes all standard recommendation metrics in a single forward pass:
      - AUC (per-user average), gAUC (impression-weighted), global AUC (pooled)
      - MRR (mean reciprocal rank)
      - nDCG@k, Precision@k, Recall@k for each cutoff k

    All operations are batched tensor operations (no Python loops over users),
    making this efficient even for large evaluation sets.

    Args:
        ks: list of cutoff values for top-k metrics (default: [5, 10, 20])
    """

    def __init__(self, ks: List[int] = None):
        if ks is None:
            ks = [5, 10, 20]
        self.ks = sorted(ks)

    def __call__(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all metrics for a batch of users.

        Args:
            predictions: [B, max_len] float tensor of model scores.
                         Padding positions can be any value (will be masked out).
            labels:      [B, max_len] binary tensor (1=relevant, 0=irrelevant).
                         Padding positions should be 0.
            lengths:     [B] int tensor — number of valid (non-padding) items per user.

        Returns:
            Dict mapping metric name to scalar float value (averaged over the batch).
            Keys: auc, gauc, global_auc, mrr, ndcg@k, precision@k, recall@k for each k.
        """
        B, max_len = predictions.shape
        mask = make_mask(lengths, max_len)  # [B, max_len] — True for real items

        results: Dict[str, float] = {}

        # ---- AUC variants ----
        # Per-user AUC: averaged only over users that have both pos and neg items
        auc_vals, valid = auc_per_sample(predictions, labels, mask)
        if valid.any():
            results["auc"] = auc_vals[valid].mean().item()
        else:
            results["auc"] = 0.0

        # gAUC: per-user AUC weighted by list length (standard industrial metric)
        results["gauc"] = gauc(predictions, labels, mask, lengths).item()
        # Global AUC: pools all items into one list (ignores user boundaries)
        results["global_auc"] = global_auc(predictions, labels, mask).item()

        # ---- MRR ----
        mrr_vals = mrr(predictions, labels, mask)
        results["mrr"] = mrr_vals.mean().item()

        # ---- Top-k metrics at each cutoff ----
        for k in self.ks:
            ndcg_vals = ndcg_at_k(predictions, labels, mask, k)
            prec_vals = precision_at_k(predictions, labels, mask, k)
            rec_vals = recall_at_k(predictions, labels, mask, k)

            results[f"ndcg@{k}"] = ndcg_vals.mean().item()
            results[f"precision@{k}"] = prec_vals.mean().item()
            results[f"recall@{k}"] = rec_vals.mean().item()

        return results

    def __repr__(self) -> str:
        return f"Evaluator(ks={self.ks})"
