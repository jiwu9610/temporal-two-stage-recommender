"""Phase 1 rule-based retrieval.

PDF-aligned: rank items by user metadata signals + a popularity prior. Amazon
data lacks demographic fields (age/country/gender), so we use what's actually
available -- a user's train-positive store and main_category histograms -- and
leave the demographic hooks for later if real fields appear.

    score(u, item) = w_store * user_store_aff[u, store(item)]
                   + w_cat   * user_cat_aff[u, category(item)]
                   + w_pop   * train_positive_popularity_prior[item]

Inputs (all train-only):
    train_df         -- train.parquet (label, user_id, parent_asin)
    item_features    -- item_features.parquet (parent_asin, store, main_category)
    candidate_pool   -- explicit set of allowed parent_asins (default: in_train_catalog)
    user_seen[u]     -- items u has seen in train (excluded from recommendations)

Tuning: small grid over (w_store, w_cat, w_pop) on a sampled subset of val
positive users (for cost), evaluated by mean Recall at the largest K. The
chosen weights are then re-evaluated on full val + test downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .evaluator import build_groundtruth, recall_precision_at_k


# ---------------------------------------------------------------------------
# Train-only affinity / popularity helpers
# ---------------------------------------------------------------------------

def _normalize_by_max(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return d
    z = max(d.values())
    if z <= 0:
        return d
    return {k: v / z for k, v in d.items()}


def build_user_facet_affinity(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    facet: str,                          # "store" or "main_category"
    user_col: str = "user_id",
    item_col: str = "parent_asin",
) -> Dict[str, Dict[str, float]]:
    """{user_id: {facet_value: normalized_affinity}} from train positives only.

    Affinity = (count of train positives for user with that facet value)
              / (max count across the user's facet values).
    Users with no train positives end up missing from the dict; downstream
    treats their affinity as zero everywhere.
    """
    pos = train_df[train_df["label"] == 1]
    item_facet = item_features.set_index(item_col)[facet]
    joined = pos.join(item_facet, on=item_col).dropna(subset=[facet])
    counts = joined.groupby([user_col, facet]).size()
    # Iterating .groupby(level=0) is clearer (and safer) than .apply().to_dict():
    # the latter folds dict-valued results back into a MultiIndex Series instead
    # of preserving the per-user dict structure we want.
    by_user: Dict[str, Dict[str, float]] = {
        u: _normalize_by_max({k: float(v) for k, v in grp.droplevel(0).items()})
        for u, grp in counts.groupby(level=0)
    }
    return by_user


def normalized_popularity_prior(
    train_df: pd.DataFrame,
    item_index: Sequence[str],
    item_col: str = "parent_asin",
) -> np.ndarray:
    """Per-item popularity prior in [0, 1], aligned to item_index order.

    Counts only train positives. Items absent from train get 0. Normalized by
    the max count so the dominant item scores 1.0.
    """
    pos = train_df[train_df["label"] == 1]
    counts = pos[item_col].value_counts()
    aligned = counts.reindex(item_index, fill_value=0).to_numpy(dtype=np.float32)
    z = aligned.max()
    return aligned / z if z > 0 else aligned


# ---------------------------------------------------------------------------
# Scoring + recommend
# ---------------------------------------------------------------------------

def _affinity_vector(
    user_pref: Dict[str, float],
    item_facet_array: np.ndarray,
) -> np.ndarray:
    """Map item_facet_array (str array) -> per-item affinity score vector."""
    if not user_pref:
        return np.zeros(len(item_facet_array), dtype=np.float32)
    # Vectorized lookup via dict.get over a Python list comprehension.
    # Faster alternatives exist (pd.Series.map) but this is clearest and
    # fine at the scales we hit (~30k items per category).
    return np.fromiter(
        (user_pref.get(s, 0.0) for s in item_facet_array),
        dtype=np.float32,
        count=len(item_facet_array),
    )


def recommend_rule_based(
    user_ids: Iterable[str],
    candidate_items: np.ndarray,             # parent_asin array, ordered
    item_store_arr: np.ndarray,              # same length, store per item
    item_cat_arr: np.ndarray,                # same length, main_category per item
    item_pop_prior: np.ndarray,              # same length, [0,1] popularity
    user_store_aff: Mapping[str, Dict[str, float]],
    user_cat_aff: Mapping[str, Dict[str, float]],
    user_seen: Mapping[str, Set[str]],
    weights: Tuple[float, float, float],     # (w_store, w_cat, w_pop)
    k: int = 100,
) -> Dict[str, List[str]]:
    """Score every candidate per user, return top-K minus seen.

    All inputs are pre-aligned to `candidate_items` order so per-user scoring
    is just three numpy vector adds + an argpartition.
    """
    w_store, w_cat, w_pop = weights
    pop_term = w_pop * item_pop_prior        # constant across users

    # Pre-compute set membership index for fast seen-masking.
    item_to_idx = {it: i for i, it in enumerate(candidate_items)}

    out: Dict[str, List[str]] = {}
    for u in user_ids:
        store_score = _affinity_vector(user_store_aff.get(u, {}), item_store_arr)
        cat_score = _affinity_vector(user_cat_aff.get(u, {}), item_cat_arr)
        score = w_store * store_score + w_cat * cat_score + pop_term

        seen = user_seen.get(u)
        if seen:
            for it in seen:
                idx = item_to_idx.get(it)
                if idx is not None:
                    score[idx] = -np.inf

        if k < len(score):
            top_idx = np.argpartition(-score, k)[:k]
            top_idx = top_idx[np.argsort(-score[top_idx])]
        else:
            top_idx = np.argsort(-score)
        out[u] = list(candidate_items[top_idx])
    return out


# ---------------------------------------------------------------------------
# Generic weighted-facets scorer (used by ablations)
# ---------------------------------------------------------------------------

def recommend_weighted_facets(
    user_ids: Iterable[str],
    candidate_items: np.ndarray,
    facet_arrays: Mapping[str, np.ndarray],          # {facet_name: per-item value array}
    user_affinities: Mapping[str, Mapping[str, Dict[str, float]]],
                                                     # {facet_name: {user: {value: aff}}}
    facet_weights: Mapping[str, float],              # {facet_name: weight}
    item_pop_prior: np.ndarray,
    pop_weight: float,
    user_seen: Mapping[str, Set[str]],
    k: int = 100,
) -> Dict[str, List[str]]:
    """Generalized rule-based scorer over an arbitrary set of facets.

    `recommend_rule_based` is a 2-facet special case. This version is what the
    Electronics ablation uses to swap `main_category` for `deeper_category` or
    drop `store` entirely without rewriting the scoring loop. All inputs must
    be pre-aligned to `candidate_items` order (same contract as
    recommend_rule_based).
    """
    pop_term = pop_weight * item_pop_prior
    item_to_idx = {it: i for i, it in enumerate(candidate_items)}
    active_facets = [(f, w) for f, w in facet_weights.items() if w != 0]

    out: Dict[str, List[str]] = {}
    for u in user_ids:
        score = pop_term.copy()
        for facet, weight in active_facets:
            score = score + weight * _affinity_vector(
                user_affinities[facet].get(u, {}),
                facet_arrays[facet],
            )
        seen = user_seen.get(u)
        if seen:
            for it in seen:
                idx = item_to_idx.get(it)
                if idx is not None:
                    score[idx] = -np.inf
        if k < len(score):
            top_idx = np.argpartition(-score, k)[:k]
            top_idx = top_idx[np.argsort(-score[top_idx])]
        else:
            top_idx = np.argsort(-score)
        out[u] = list(candidate_items[top_idx])
    return out


# ---------------------------------------------------------------------------
# Weight tuning on val
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    best_weights: Tuple[float, float, float]
    best_recall: float
    grid: List[dict]                         # one row per (weights, recall) try


def tune_weights(
    val_df: pd.DataFrame,
    user_ids_for_tuning: Sequence[str],
    candidate_items: np.ndarray,
    item_store_arr: np.ndarray,
    item_cat_arr: np.ndarray,
    item_pop_prior: np.ndarray,
    user_store_aff: Mapping[str, Dict[str, float]],
    user_cat_aff: Mapping[str, Dict[str, float]],
    user_seen: Mapping[str, Set[str]],
    grid: Iterable[Tuple[float, float, float]] | None = None,
    k_for_tuning: int = 100,
) -> TuningResult:
    """Pick (w_store, w_cat, w_pop) by val Recall@k_for_tuning.

    `user_ids_for_tuning` is typically a sampled subset of val-positive users
    -- 64 grid points x 60k users x 30k items is slow if we use all users.
    """
    if grid is None:
        grid = [
            (ws, wc, wp)
            for ws in (0.5, 1.0, 2.0)
            for wc in (0.5, 1.0, 2.0)
            for wp in (0.0, 0.5, 1.0)
        ]
    grid = list(grid)

    # Build val gt restricted to the tuning user set.
    gt_full = build_groundtruth(val_df)
    gt = {u: items for u, items in gt_full.items() if u in set(user_ids_for_tuning)}

    rows: List[dict] = []
    best = (None, -1.0)
    for weights in grid:
        topk = recommend_rule_based(
            user_ids=user_ids_for_tuning,
            candidate_items=candidate_items,
            item_store_arr=item_store_arr,
            item_cat_arr=item_cat_arr,
            item_pop_prior=item_pop_prior,
            user_store_aff=user_store_aff,
            user_cat_aff=user_cat_aff,
            user_seen=user_seen,
            weights=weights,
            k=k_for_tuning,
        )
        recall, _, _ = recall_precision_at_k(topk, gt, k_for_tuning)
        rows.append({"w_store": weights[0], "w_cat": weights[1],
                     "w_pop": weights[2], f"recall@{k_for_tuning}": recall})
        if recall > best[1]:
            best = (weights, recall)

    return TuningResult(best_weights=best[0], best_recall=best[1], grid=rows)
