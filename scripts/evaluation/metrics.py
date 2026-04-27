"""
Vectorized recommendation evaluation metrics using PyTorch.

Implements standard information retrieval / recommendation metrics in a fully
batched, GPU-friendly manner. All operations avoid Python loops over samples —
everything is expressed as tensor operations for efficiency.

**Jagged tensor semantics:**
Different users have different numbers of candidate items. Rather than using
ragged/jagged tensors, we use padded 2D tensors + an explicit lengths vector:
    predictions: [B, max_len]  — model scores (higher = more likely to be relevant)
    labels:      [B, max_len]  — binary relevance (1=relevant, 0=not relevant)
    lengths:     [B]           — number of valid items per user (rest is padding)
    mask:        [B, max_len]  — True for valid positions, False for padding

Padding positions are masked out and never affect metric computation.

**Metrics implemented:**
    - AUC (per-sample): Wilcoxon-Mann-Whitney concordance — fraction of (pos, neg)
      pairs where the positive item is scored higher
    - Global AUC: pools all items across the batch into one big list
    - gAUC: group AUC — weighted average of per-user AUCs by list length
    - MRR: mean reciprocal rank of the first relevant item
    - nDCG@k: normalized discounted cumulative gain at cutoff k
    - Precision@k: fraction of top-k items that are relevant
    - Recall@k: fraction of all relevant items captured in top-k
"""

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create a boolean mask [B, max_len] from a lengths vector [B].

    Example: lengths=[3, 2], max_len=4  →  [[T, T, T, F], [T, T, F, F]]
    """
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _masked_sort_descending(predictions: torch.Tensor, mask: torch.Tensor):
    """Sort predictions descending within valid positions, pushing padding to the end.

    Used by MRR, nDCG@k, Precision@k, Recall@k to rank items by predicted score.

    Returns:
        sorted_indices: [B, max_len] — indices that sort each row's predictions
                        in descending order (padding positions get the lowest rank)
    """
    # Set padding positions to -inf so they sort to the end
    masked_preds = predictions.masked_fill(~mask, float("-inf"))
    sorted_indices = masked_preds.argsort(dim=1, descending=True)
    return sorted_indices


# ---------------------------------------------------------------------------
# AUC  (per-sample via Wilcoxon-Mann-Whitney concordance)
# ---------------------------------------------------------------------------
# The key insight: sort items by predicted score in ascending order, then for
# each positive item, count how many negatives have LOWER predicted scores.
# This is equivalent to the Wilcoxon-Mann-Whitney U statistic.

def auc_per_sample(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample AUC using the rank-sum / concordance approach.

    For each user (row), counts the fraction of (positive, negative) pairs
    where the positive item is scored higher than the negative item.
    AUC = 0.5 means random ranking, AUC = 1.0 means all positives ranked above all negatives.

    Returns:
        auc: [B] per-sample AUC values.  Samples with n_pos=0 or n_neg=0
             are assigned 0.0 (caller should filter via valid_mask).
        valid_mask: [B] True for samples where AUC is well-defined
                    (i.e., user has both positive AND negative items).
    """
    # Sort items by predicted score in ASCENDING order
    # After sorting: position 0 = lowest score, position L-1 = highest score
    masked_preds = predictions.masked_fill(~mask, float("-inf"))
    sorted_indices = masked_preds.argsort(dim=1)  # ascending
    sorted_labels = labels.gather(1, sorted_indices)
    sorted_mask = mask.gather(1, sorted_indices)

    # Count negatives seen so far as we walk from low to high predicted score
    neg_flags = (1.0 - sorted_labels) * sorted_mask.float()
    cum_neg = neg_flags.cumsum(dim=1)

    # For each positive item at position i, cum_neg[i] = number of negatives
    # with lower predicted score → these are concordant pairs
    pos_flags = sorted_labels * sorted_mask.float()
    concordant = (pos_flags * cum_neg).sum(dim=1)          # [B]

    # AUC = concordant_pairs / total_pairs
    n_pos = (labels * mask.float()).sum(dim=1)              # [B]
    n_neg = ((1.0 - labels) * mask.float()).sum(dim=1)      # [B]
    denom = n_pos * n_neg

    valid = denom > 0
    auc = torch.where(valid, concordant / denom.clamp(min=1), torch.zeros_like(denom))
    return auc, valid


def global_auc(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pooled (global) AUC — flattens all valid items across the batch into one big list.

    Unlike per-sample AUC which computes AUC per user then averages, global AUC
    pools all (prediction, label) pairs together. This gives higher weight to
    users with more items.
    """
    valid_preds = predictions[mask]
    valid_labels = labels[mask]

    n_pos = valid_labels.sum()
    n_neg = valid_labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Same concordance approach as auc_per_sample, but on the flattened 1D tensors
    order = valid_preds.argsort()
    sorted_labels = valid_labels[order].float()
    cum_neg = (1.0 - sorted_labels).cumsum(0)
    concordant = (sorted_labels * cum_neg).sum()
    return concordant / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# gAUC  (group AUC: impression-weighted mean of per-user AUCs)
# ---------------------------------------------------------------------------
# gAUC is the standard metric in industrial recommendation systems (used by
# Alibaba, Huawei, etc.). It weights each user's AUC by their list length,
# giving more influence to users with more interactions.

def gauc(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Group AUC — weighted average of per-user AUCs, weighted by list length.

    gAUC = sum(auc_i * len_i) / sum(len_i)  for users with valid AUC
    """
    auc_vals, valid = auc_per_sample(predictions, labels, mask)
    weights = lengths.float() * valid.float()
    total_weight = weights.sum()
    if total_weight == 0:
        return torch.tensor(0.0, device=predictions.device)
    return (auc_vals * weights).sum() / total_weight


# ---------------------------------------------------------------------------
# MRR  (Mean Reciprocal Rank)
# ---------------------------------------------------------------------------

def mrr(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample MRR: 1 / rank_of_first_relevant_item (descending by score).

    Returns:
        mrr_vals: [B]  (0.0 if no relevant item exists)
    """
    # Sort items by predicted score (highest first)
    sorted_indices = _masked_sort_descending(predictions, mask)
    sorted_labels = labels.gather(1, sorted_indices)
    sorted_mask = mask.gather(1, sorted_indices)
    sorted_labels = sorted_labels * sorted_mask.float()  # Zero out padding positions

    B, L = sorted_labels.shape
    # 1-indexed rank positions: [1, 2, 3, ..., L]
    ranks = torch.arange(1, L + 1, device=labels.device).float().unsqueeze(0)  # [1, L]

    # reciprocal_ranks[i,j] = label[i,j] / rank[j]
    # For relevant items: 1/rank. For irrelevant: 0.
    # Taking max gives 1/(rank of first relevant item) = MRR for that user
    reciprocal_ranks = sorted_labels / ranks                    # [B, L]
    mrr_vals = reciprocal_ranks.max(dim=1).values               # [B]
    return mrr_vals


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------

def ndcg_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-sample Normalized Discounted Cumulative Gain at cutoff k.

    nDCG measures ranking quality by giving higher weight to relevant items
    that appear earlier in the ranked list. The discount factor is 1/log2(rank+1).

    nDCG@k = DCG@k / IDCG@k where:
      DCG@k  = sum_{i=1}^{k} rel_i / log2(i+1)        (actual ranking)
      IDCG@k = same formula but with labels sorted by relevance (ideal ranking)

    Returns:
        ndcg: [B]  (0.0 if no relevant items or k=0)
    """
    B, L = predictions.shape
    sorted_indices = _masked_sort_descending(predictions, mask)
    sorted_labels = labels.gather(1, sorted_indices).float()
    sorted_mask = mask.gather(1, sorted_indices)
    sorted_labels = sorted_labels * sorted_mask.float()

    # Only consider the top-k positions
    eff_k = min(k, L)
    topk_labels = sorted_labels[:, :eff_k]                     # [B, eff_k]

    # Discount factors: 1/log2(2), 1/log2(3), ..., 1/log2(k+1)
    # Using (i+2) because i is 0-indexed and log2(1)=0 is undefined
    discounts = torch.log2(
        torch.arange(2, eff_k + 2, device=labels.device).float()
    ).unsqueeze(0)                                              # [1, eff_k]
    dcg = (topk_labels / discounts).sum(dim=1)                  # [B]

    # Ideal DCG: what DCG would be if we ranked all relevant items first
    ideal_labels = labels.masked_fill(~mask, 0.0).float()
    ideal_sorted = ideal_labels.sort(dim=1, descending=True).values[:, :eff_k]
    idcg = (ideal_sorted / discounts).sum(dim=1)                # [B]

    # Normalize: nDCG = DCG / IDCG (handle edge case where IDCG = 0)
    valid = idcg > 0
    ndcg = torch.where(valid, dcg / idcg.clamp(min=1e-12), torch.zeros_like(dcg))
    return ndcg


# ---------------------------------------------------------------------------
# Precision@k  &  Recall@k
# ---------------------------------------------------------------------------

def precision_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-sample Precision@k: of the top-k ranked items, what fraction are relevant?

    Precision@k = |relevant items in top-k| / k
    Returns [B].
    """
    B, L = predictions.shape
    sorted_indices = _masked_sort_descending(predictions, mask)
    sorted_labels = labels.gather(1, sorted_indices).float()
    sorted_mask = mask.gather(1, sorted_indices)
    sorted_labels = sorted_labels * sorted_mask.float()

    eff_k = min(k, L)
    hits = sorted_labels[:, :eff_k].sum(dim=1)                 # [B]
    return hits / eff_k


def recall_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-sample Recall@k: of all relevant items, what fraction appear in top-k?

    Recall@k = |relevant items in top-k| / |all relevant items|
    Returns [B] (0.0 if the user has no relevant items).
    """
    B, L = predictions.shape
    sorted_indices = _masked_sort_descending(predictions, mask)
    sorted_labels = labels.gather(1, sorted_indices).float()
    sorted_mask = mask.gather(1, sorted_indices)
    sorted_labels = sorted_labels * sorted_mask.float()

    eff_k = min(k, L)
    hits = sorted_labels[:, :eff_k].sum(dim=1)                 # [B]
    total_relevant = (labels * mask.float()).sum(dim=1)          # [B]
    valid = total_relevant > 0
    return torch.where(valid, hits / total_relevant.clamp(min=1), torch.zeros_like(hits))
