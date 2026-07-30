"""Bridge from flat per-row ranker predictions to the vectorized metrics in
scripts/evaluation/metrics.py.

metrics.py exposes a pure tensor interface — padded 2D [B, max_len] tensors
plus a lengths vector, with no notion of user ids. Ranker prediction dumps are
flat rows (user_id, score, label). gAUC / MRR / nDCG are only defined per
user, so this module groups flat rows by user, packs them into padded batches,
runs the metric primitives batch by batch, and aggregates exact numerators /
denominators. Results are invariant to `batch_users`.

Denominator semantics (they differ per metric — read before quoting numbers):

    auc                  mean per-user AUC over users with >=1 pos AND >=1 neg
    gauc                 list-length-weighted mean over those same users
    global_auc           pooled over all rows in one flat pass (ignores users)
    mrr                  mean over users with >=1 positive in their list
    ndcg@k               mean over users with >=1 positive in their list
    precision@k          mean over ALL listed users; denominator is always k
                         (lists shorter than k count the missing slots as
                         misses — enforced by padding batches to >= max(ks))
    recall@k_conditional mean over users with >=1 positive in their list
                         (== conditional on the target being retrieved)
    recall@k_listed      mean over all listed users (missed-retrieval users
                         count as 0; users absent from the table do not count)
    recall@k_overall     hit_sum / n_gt_users when n_gt_users is given —
                         comparable to build_split_report's overall recall
                         under the one-positive-per-user snapshot protocol

Tie policy — deterministic and conservative, enforced HERE (metrics.py's own
argsort is not stable, so its tie behavior is unspecified): rows are
lexsorted (score descending, negatives before positives among equal scores)
and the metric primitives are fed each row's within-user RANK instead of the
raw score. Ordering of distinct scores is unchanged; tied pairs earn zero
AUC credit and tied negatives fill top-k slots before tied positives. This
matters because the calibration rule layer zeroes large fractions of each
list: a threshold that zeroes the target's score demotes it below every
co-zeroed negative — always a miss at k below the list length, never an
argsort accident. (Edge: a user whose whole list is shorter than k still has
all rows inside top-k by construction; at K=500 candidate scale this is
negligible and documented rather than special-cased.)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from scripts.evaluation.metrics import (
    auc_per_sample,
    global_auc,
    make_mask,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def evaluate_flat(
    user_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    ks: Sequence[int] = (10, 50, 100),
    batch_users: int = 4096,
    n_gt_users: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute all metrics over flat (user, score, label) rows.

    `n_gt_users` — number of ground-truth users the recall denominator should
    use (>= number of distinct users present here; users whose target was
    never generated as a candidate may have no rows at all).
    """
    n = len(user_ids)
    if not (len(scores) == len(labels) == n):
        raise ValueError("user_ids / scores / labels length mismatch")
    if n == 0:
        return {"n_rows": 0, "n_users_listed": 0}

    # Group rows by user AND apply the tie policy in one deterministic sort:
    # user asc, then score desc, then label asc (negatives before positives
    # among equal scores). np.lexsort: last key is primary.
    codes = np.unique(np.asarray(user_ids), return_inverse=True)[1]
    scores64 = np.asarray(scores, dtype=np.float64)
    labels64 = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(scores64).all():
        raise ValueError("evaluate_flat: non-finite scores would corrupt "
                         "the deterministic tie ordering")
    order = np.lexsort((labels64, -scores64, codes))
    codes_s = codes[order]
    scores_s = scores64[order]
    labels_s = labels64[order].astype(np.float32)
    counts = np.bincount(codes_s)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_users = len(counts)
    if n_gt_users is not None and n_gt_users < n_users:
        raise ValueError(
            f"n_gt_users={n_gt_users} < distinct listed users={n_users}")

    ks = sorted(ks)
    acc = {
        "sum_auc": 0.0, "n_auc_valid": 0,
        "sum_auc_len": 0.0, "sum_len_valid": 0.0,
        "sum_mrr": 0.0, "n_pos_users": 0,
    }
    for k in ks:
        acc[f"sum_ndcg@{k}"] = 0.0
        acc[f"sum_prec@{k}"] = 0.0
        acc[f"sum_rec@{k}"] = 0.0
        acc[f"sum_hits@{k}"] = 0.0

    for b0 in range(0, n_users, batch_users):
        b1 = min(b0 + batch_users, n_users)
        r0, r1 = offsets[b0], offsets[b1]
        batch_counts = counts[b0:b1]
        # Pad width to at least max(ks): metrics.py uses eff_k = min(k, width)
        # as the precision@k denominator, so without this floor a batch of
        # short lists would silently change P@k semantics (and break batch
        # invariance). With the floor, precision@k is always hits / k.
        B, L = b1 - b0, max(int(batch_counts.max()), ks[-1])

        # Scatter the contiguous per-user runs into padded [B, L] arrays.
        # Rows are already in final ranked order (tie policy applied), so the
        # within-user position IS the rank; feeding -rank as the score makes
        # every row's score distinct and lets metrics.py's unstable argsort
        # reproduce exactly this deterministic order.
        row_of = np.repeat(np.arange(B), batch_counts)
        col_of = np.arange(r1 - r0) - np.repeat(
            offsets[b0:b1] - r0, batch_counts)
        preds_np = np.zeros((B, L), dtype=np.float32)
        labs_np = np.zeros((B, L), dtype=np.float32)
        preds_np[row_of, col_of] = -col_of.astype(np.float32)
        labs_np[row_of, col_of] = labels_s[r0:r1]

        preds = torch.from_numpy(preds_np).to(device)
        labs = torch.from_numpy(labs_np).to(device)
        lengths = torch.from_numpy(batch_counts.astype(np.int64)).to(device)
        mask = make_mask(lengths, L)

        auc_vals, valid = auc_per_sample(preds, labs, mask)
        acc["sum_auc"] += float(auc_vals[valid].sum())
        acc["n_auc_valid"] += int(valid.sum())
        acc["sum_auc_len"] += float((auc_vals * lengths.float() * valid.float()).sum())
        acc["sum_len_valid"] += float((lengths.float() * valid.float()).sum())

        n_pos = (labs * mask.float()).sum(dim=1)
        has_pos = n_pos > 0
        acc["n_pos_users"] += int(has_pos.sum())
        acc["sum_mrr"] += float(mrr(preds, labs, mask).sum())

        for k in ks:
            acc[f"sum_ndcg@{k}"] += float(ndcg_at_k(preds, labs, mask, k).sum())
            acc[f"sum_prec@{k}"] += float(precision_at_k(preds, labs, mask, k).sum())
            rec = recall_at_k(preds, labs, mask, k)
            acc[f"sum_rec@{k}"] += float(rec.sum())
            acc[f"sum_hits@{k}"] += float((rec * n_pos).sum())

    out: Dict[str, float] = {
        "n_rows": int(n),
        "n_users_listed": int(n_users),
        "n_users_with_pos": int(acc["n_pos_users"]),
        "n_users_auc_valid": int(acc["n_auc_valid"]),
    }
    if n_gt_users is not None:
        out["n_gt_users"] = int(n_gt_users)
    out["auc"] = acc["sum_auc"] / acc["n_auc_valid"] if acc["n_auc_valid"] else 0.0
    out["gauc"] = (acc["sum_auc_len"] / acc["sum_len_valid"]
                   if acc["sum_len_valid"] else 0.0)
    out["mrr"] = acc["sum_mrr"] / acc["n_pos_users"] if acc["n_pos_users"] else 0.0

    # Global AUC pooled over all rows — same concordance formula as
    # metrics.global_auc but in int64/float64 (its float32 cumsum loses
    # integer precision past ~16.7M rows; full-scale dumps are ~60M) and
    # with the strict tie policy: ascending score with positives FIRST among
    # equal scores, so a tied negative is never counted as concordant.
    flat_order = np.lexsort((-labels_s.astype(np.float64), scores_s))
    sorted_flat_labels = labels_s[flat_order].astype(np.int64)
    n_pos_total = int(sorted_flat_labels.sum())
    n_neg_total = len(sorted_flat_labels) - n_pos_total
    if n_pos_total and n_neg_total:
        cum_neg = np.cumsum(1 - sorted_flat_labels)
        concordant = int((sorted_flat_labels * cum_neg).sum())
        out["global_auc"] = concordant / (n_pos_total * n_neg_total)
    else:
        out["global_auc"] = 0.0

    for k in ks:
        np_users = acc["n_pos_users"]
        out[f"ndcg@{k}"] = acc[f"sum_ndcg@{k}"] / np_users if np_users else 0.0
        out[f"precision@{k}"] = acc[f"sum_prec@{k}"] / n_users
        out[f"recall@{k}_conditional"] = (
            acc[f"sum_rec@{k}"] / np_users if np_users else 0.0)
        out[f"recall@{k}_listed"] = acc[f"sum_rec@{k}"] / n_users
        if n_gt_users:
            out[f"recall@{k}_overall"] = acc[f"sum_hits@{k}"] / n_gt_users
    return out
