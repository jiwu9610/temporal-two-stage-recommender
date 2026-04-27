"""Phase 1 popularity baseline.

Train-only, observed positives only. Recommend the globally most-popular items
the user has not yet seen, ranked within an explicit candidate_pool.

  popularity(item) = count of train rows with label == 1 and parent_asin == item

The function lives bare on purpose -- popularity is a statistic, not a learned
model. No fit() / predict() ceremony, no class.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Set

import numpy as np
import pandas as pd


def compute_popularity(train_df: pd.DataFrame,
                       item_col: str = "parent_asin") -> pd.Series:
    """Series indexed by item, value = positive train interaction count, sorted desc."""
    pos = train_df[train_df["label"] == 1]
    counts = pos[item_col].value_counts()  # already sorted descending
    counts.index.name = item_col
    counts.name = "popularity"
    return counts


def user_seen_from_train(train_df: pd.DataFrame,
                         user_col: str = "user_id",
                         item_col: str = "parent_asin",
                         users: Iterable[str] | None = None) -> Dict[str, Set[str]]:
    """Per-user set of items seen in train (positives + hard negatives).

    These must be excluded from the recommendation list -- the user already
    interacted with them. If `users` is given, only build the dict for that
    subset (saves memory when eval_user_ids is small).
    """
    if users is not None:
        train_df = train_df[train_df[user_col].isin(set(users))]
    return {u: set(g) for u, g in train_df.groupby(user_col)[item_col].agg(set).items()}


def recommend_popularity(
    popularity: pd.Series,
    candidate_pool: Set[str],
    user_ids: Iterable[str],
    user_seen: Mapping[str, Set[str]],
    k: int = 100,
) -> Dict[str, List[str]]:
    """Per-user top-K = popularity-ranked candidate_pool items minus user_seen[u].

    Ranks ALL items in candidate_pool, not just items with positive train count.
    Items absent from `popularity` (zero positive interactions) are appended
    with score 0; among any tied score the order is alphabetical by parent_asin
    so the ranking is fully deterministic. Top-K is unlikely to dip into the
    zero-score tail in practice, but the contract now matches the
    `in_train_catalog` candidate-pool semantics exactly.
    """
    pool_arr = np.array(sorted(candidate_pool))                 # alphabetical -> stable tiebreak
    pop_aligned = popularity.reindex(pool_arr, fill_value=0).to_numpy()
    # Stable sort by descending count keeps pool_arr alphabetical within ties.
    order = np.argsort(-pop_aligned, kind="stable")
    ranked_global = pool_arr[order]

    out: Dict[str, List[str]] = {}
    for u in user_ids:
        seen = user_seen.get(u, set())
        if not seen:
            out[u] = list(ranked_global[:k])
            continue
        unseen_mask = ~np.isin(ranked_global, list(seen))
        out[u] = list(ranked_global[unseen_mask][:k])
    return out
