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
                         (== conditional on the target being retrieved).
                         DIAGNOSTIC ONLY — its denominator is decided by our
                         own retrieval, so shrinking the candidate set raises
                         it while the system gets worse. Never optimise it.
    recall@k_listed      mean over all listed users (missed-retrieval users
                         count as 0; users absent from the table do not count)
    recall@k_overall     macro-average per-user recall over n_gt_users —
                         each user contributes |top-k ∩ their positives| /
                         |their positives|, and a user with no rows at all
                         contributes 0. This is the headline number: it is the
                         product of candidate coverage and conditional recall,
                         and it cannot be gamed by narrowing retrieval.
    hit_rate@k_overall   fraction of n_gt_users with AT LEAST ONE positive in
                         top-k. Equals recall@k_overall when every user has
                         exactly one positive; diverges above it otherwise.
    hits@k_per_gt_user   mean COUNT of positives in top-k per gt user (can
                         exceed 1 under multi-positive) — not a rate.

Multi-positive: every metric here is defined for users with several positives.
Under the one-positive-per-user protocol recall@k_overall, hit_rate@k_overall
and hits@k_per_gt_user all collapse to the same number, and precision@k
degenerates to recall@k_listed / k, carrying no independent information.

AUC scale context (auc_implied_mean_rank, auc_needed_for_topK): AUC is a
percentile, Recall@k is an absolute rank cutoff, and the two decouple as the
candidate list grows. On a ~1000-item list AUC 0.81 puts the average positive
near rank 200 — outside the top 100 — so "high AUC" does not imply "high
recall@100" at this scale. These derived fields make that explicit; see the
computation at the bottom of evaluate_flat for the exact formulas.

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
    true_positives_per_user: Optional[Dict[str, int]] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute all metrics over flat (user, score, label) rows.

    `n_gt_users` — number of ground-truth users the recall denominator should
    use (>= number of distinct users present here; users whose target was
    never generated as a candidate may have no rows at all).

    `true_positives_per_user` — |P_u| for every user, counting positives that
    retrieval MISSED and which therefore have no row here. Required whenever a
    user can have more than one positive: `labels` only marks the positives
    that made it into the candidate list, so without this the per-user recall
    denominator is the retrieved-positive count and a user who bought 4 items
    of which we retrieved 1 and ranked it first scores 1.0 instead of 0.25.
    Under one-positive-per-user it is unnecessary — a retrieved positive is
    the whole truth, and a missed one leaves the user with no rows at all.
    """
    n = len(user_ids)
    if not (len(scores) == len(labels) == n):
        raise ValueError("user_ids / scores / labels length mismatch")
    if n == 0:
        return {"n_rows": 0, "n_users_listed": 0}

    # Group rows by user AND apply the tie policy in one deterministic sort:
    # user asc, then score desc, then label asc (negatives before positives
    # among equal scores). np.lexsort: last key is primary.
    uniq_users, codes = np.unique(np.asarray(user_ids), return_inverse=True)
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

    # |P_u| including positives retrieval never surfaced, in uniq_users order.
    true_np: Optional[np.ndarray] = None
    if true_positives_per_user is not None:
        true_np = np.array(
            [float(true_positives_per_user.get(u, 0.0)) for u in uniq_users],
            dtype=np.float64,
        )
        in_list = np.bincount(codes, weights=labels64, minlength=len(uniq_users))
        short = in_list > true_np + 1e-9
        if short.any():
            bad = uniq_users[short][:5]
            raise ValueError(
                "true_positives_per_user is smaller than the positives already "
                f"present in the rows for users {list(bad)} — a retrieved "
                "positive must be part of the user's ground truth"
            )

    ks = sorted(ks)
    acc = {
        "sum_auc": 0.0, "n_auc_valid": 0,
        "sum_auc_len": 0.0, "sum_len_valid": 0.0,
        "sum_mrr": 0.0, "n_pos_users": 0, "sum_neg_valid": 0.0,
    }
    for k in ks:
        acc[f"sum_ndcg@{k}"] = 0.0
        acc[f"sum_prec@{k}"] = 0.0
        acc[f"sum_rec@{k}"] = 0.0
        acc[f"sum_hits@{k}"] = 0.0
        acc[f"n_hit_users@{k}"] = 0
        acc[f"sum_rec_true@{k}"] = 0.0

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
        # Negatives per AUC-valid user: the scale AUC must be read against.
        acc["sum_neg_valid"] += float(
            ((lengths.float() - n_pos) * valid.float()).sum())

        true_batch = (torch.from_numpy(true_np[b0:b1]).to(device)
                      if true_np is not None else None)
        for k in ks:
            acc[f"sum_ndcg@{k}"] += float(
                ndcg_at_k(preds, labs, mask, k,
                          n_true_positives=true_batch).sum())
            acc[f"sum_prec@{k}"] += float(precision_at_k(preds, labs, mask, k).sum())
            rec = recall_at_k(preds, labs, mask, k)
            acc[f"sum_rec@{k}"] += float(rec.sum())
            # rec is a fraction of the user's positives; rec * n_pos restores
            # the raw hit COUNT, and rec > 0 iff at least one positive landed
            # in top-k. Under one-positive-per-user all three coincide.
            hits_k = rec * n_pos
            acc[f"sum_hits@{k}"] += float(hits_k.sum())
            acc[f"n_hit_users@{k}"] += int((rec > 0).sum())
            if true_batch is not None:
                # Recall against the FULL ground truth, not just the slice of
                # it that retrieval happened to surface.
                denom = true_batch.to(hits_k.dtype).clamp(min=1.0)
                acc[f"sum_rec_true@{k}"] += float((hits_k / denom).sum())

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
        # _conditional deliberately keeps the retrieved-positive denominator:
        # it isolates the RANKER by asking "of the positives retrieval handed
        # us, how many did we rank into top-k". _listed and _overall measure
        # the system, so they use |P_u| whenever it is available.
        rec_num = (acc[f"sum_rec_true@{k}"] if true_np is not None
                   else acc[f"sum_rec@{k}"])
        out[f"recall@{k}_conditional"] = (
            acc[f"sum_rec@{k}"] / np_users if np_users else 0.0)
        out[f"recall@{k}_listed"] = rec_num / n_users
        if n_gt_users:
            # Macro-average of per-user recall. Was sum_hits/n_gt_users, which
            # is identical while every user has exactly one positive but
            # becomes mean hit COUNT (unbounded above 1) once they do not.
            # With true_positives_per_user the denominator is |P_u| rather than
            # the retrieved-positive count, so missed positives cost recall
            # instead of vanishing from the average.
            out[f"recall@{k}_overall"] = rec_num / n_gt_users
            out[f"hit_rate@{k}_overall"] = acc[f"n_hit_users@{k}"] / n_gt_users
            out[f"hits@{k}_per_gt_user"] = acc[f"sum_hits@{k}"] / n_gt_users

    # AUC scale context — see the module docstring. For one positive among M
    # negatives the positive outranks AUC*M of them, so its expected rank is
    # 1 + (1-AUC)*M; inverting, reaching top-k needs AUC >= 1 - (k-1)/M. Exact
    # for one positive per user, first-order otherwise. Reported so nobody
    # reads a percentile as if it were a top-k guarantee.
    mean_neg = (acc["sum_neg_valid"] / acc["n_auc_valid"]
                if acc["n_auc_valid"] else 0.0)
    out["mean_negatives_per_auc_user"] = mean_neg
    if mean_neg > 0:
        out["auc_implied_mean_rank"] = 1.0 + (1.0 - out["auc"]) * mean_neg
        for k in ks:
            out[f"auc_needed_for_top{k}"] = max(0.0, 1.0 - (k - 1) / mean_neg)
    return out
