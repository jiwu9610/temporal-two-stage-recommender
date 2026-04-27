"""Phase 1 retrieval evaluator.

Held-out-positive Recall@K / Precision@K. Spec contract:

    gt[u] = held-out positive items for user u
            (val/test rows with label == 1; built from val.parquet or test.parquet)

    Recall@K(u)    = |TopK(u) intersect gt[u]| / |gt[u]|
    Precision@K(u) = |TopK(u) intersect gt[u]| / K

    Aggregate only over users with |gt[u]| > 0.
    The metric never depends on training batch size, in-batch sampling, or any
    sampled negative scheme. Batch is a training/computation convenience, not
    part of the metric definition.

`build_split_report` adds the contextual fields needed for downstream review
alongside Recall (n_total_eval_users, candidate_pool size, held-out coverage,
etc.) so a JSON report tells the whole story.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd


@dataclass
class SplitReport:
    """Per-split summary for a single retriever's predictions."""
    split: str                                       # "val" or "test"
    n_total_eval_users: int
    n_positive_eval_users: int
    positive_eval_user_rate: float
    candidate_pool_type: str
    candidate_pool_size: int
    heldout_positive_coverage_by_candidate_pool: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "n_total_eval_users": self.n_total_eval_users,
            "n_positive_eval_users": self.n_positive_eval_users,
            "positive_eval_user_rate": self.positive_eval_user_rate,
            "candidate_pool_type": self.candidate_pool_type,
            "candidate_pool_size": self.candidate_pool_size,
            "heldout_positive_coverage_by_candidate_pool":
                self.heldout_positive_coverage_by_candidate_pool,
            "metrics": self.metrics,
        }


def build_groundtruth(split_df: pd.DataFrame,
                      user_col: str = "user_id",
                      item_col: str = "parent_asin") -> Dict[str, Set[str]]:
    """gt[u] = set of items u positively interacted with in this split (label==1).

    Users whose only held-out interaction is a hard negative get gt[u] == set();
    they are excluded from the recall denominator.
    """
    pos = split_df[split_df["label"] == 1]
    gt: Dict[str, Set[str]] = {}
    for u, g in pos.groupby(user_col):
        gt[u] = set(g[item_col].astype(str))
    return gt


def coverage_of_pool(gt: Mapping[str, Set[str]], pool: Set[str]) -> float:
    """Fraction of held-out positive items (across users) that lie in the pool.

    Computed at the (user, item) row level, not item-set level: a user with 2
    held-out positives where 1 is in the pool contributes 0.5.
    """
    n_total = sum(len(items) for items in gt.values())
    if n_total == 0:
        return 0.0
    n_in = sum(len(items & pool) for items in gt.values())
    return n_in / n_total


def recall_precision_at_k(
    topk_per_user: Mapping[str, Sequence[str]],
    gt_per_user: Mapping[str, Set[str]],
    k: int,
) -> tuple[float, float, int]:
    """Average Recall@K and Precision@K over users with |gt[u]| > 0.

    Returns (mean_recall, mean_precision, n_eval_users).
    A user is "eval" iff |gt[u]| > 0; users with empty gt are skipped.
    """
    recalls: List[float] = []
    precisions: List[float] = []
    for u, gt in gt_per_user.items():
        if not gt:
            continue
        topk = topk_per_user.get(u, [])
        topk_k = list(topk)[:k]
        hits = len(gt & set(topk_k))
        recalls.append(hits / len(gt))
        precisions.append(hits / k)
    if not recalls:
        return 0.0, 0.0, 0
    return float(np.mean(recalls)), float(np.mean(precisions)), len(recalls)


def build_split_report(
    split: str,
    topk_per_user: Mapping[str, Sequence[str]],
    gt_per_user: Mapping[str, Set[str]],
    candidate_pool: Set[str],
    candidate_pool_type: str,
    ks: Iterable[int] = (10, 50, 100),
) -> SplitReport:
    """Run all (Recall@K, Precision@K) for a split and pack the report."""
    n_total = len(topk_per_user)
    n_positive = sum(1 for u in topk_per_user if gt_per_user.get(u))
    coverage = coverage_of_pool(gt_per_user, candidate_pool)

    metrics: Dict[str, float] = {}
    for k in ks:
        r, p, _ = recall_precision_at_k(topk_per_user, gt_per_user, k)
        metrics[f"Recall@{k}"] = r
        metrics[f"Precision@{k}"] = p

    return SplitReport(
        split=split,
        n_total_eval_users=n_total,
        n_positive_eval_users=n_positive,
        positive_eval_user_rate=(n_positive / n_total) if n_total else 0.0,
        candidate_pool_type=candidate_pool_type,
        candidate_pool_size=len(candidate_pool),
        heldout_positive_coverage_by_candidate_pool=coverage,
        metrics=metrics,
    )
