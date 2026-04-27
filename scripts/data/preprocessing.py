"""
Preprocessing utilities for exploratory dataset analysis.

These functions are used in the EDA notebook (01_eda_dataset_stats.ipynb) to
understand dataset characteristics before building the full preprocessing pipeline.

Three main analyses:
  1. **Core-filtering** (filter_by_counts, compute_filtering_table)
     Iteratively remove items/users with too few reviews until the dataset stabilizes.
     This is the standard "k-core" filtering used in rec-sys literature.

  2. **Bought-together hit rate** (compute_bought_together_hit_rate)
     Measures what fraction of users have actually reviewed products that appear in
     each other's "bought_together" lists — validates that collaborative signal exists.

  3. **Cross-category user overlap** (cross_category_users)
     Analyzes how many users review products in multiple categories — informs whether
     cross-category features would be valuable for the recommendation model.
"""

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .loader import stream_user_ids


def filter_by_counts(
    df: pd.DataFrame,
    min_item_reviews: int = 5,
    min_user_reviews: int = 3,
    item_col: str = "parent_asin",
    user_col: str = "user_id",
    max_iters: int = 20,
) -> pd.DataFrame:
    """Iterative core-filtering: remove items/users below thresholds until stable.

    Removing low-count items may cause some users to drop below threshold
    and vice versa, so we iterate until no more rows are removed.

    Args:
        df: reviews DataFrame
        min_item_reviews: minimum reviews an item must have
        min_user_reviews: minimum reviews a user must have
        max_iters: safety limit on iterations

    Returns:
        Filtered DataFrame (copy)
    """
    df = df.copy()
    prev_len = len(df) + 1  # Initialize to force at least one iteration

    # Iterate: removing low-count items may cause users to drop below threshold
    # (and vice versa), so we repeat until the dataset size stabilizes.
    for i in range(max_iters):
        if len(df) == prev_len:
            break  # Converged — no more rows removed
        prev_len = len(df)

        # Remove items with fewer than min_item_reviews reviews
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_reviews].index
        df = df[df[item_col].isin(valid_items)]

        # Remove users with fewer than min_user_reviews reviews
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        df = df[df[user_col].isin(valid_users)]

    return df.reset_index(drop=True)


def compute_filtering_table(
    stats: dict,
    item_thresholds: list[int] = None,
    user_thresholds: list[int] = None,
) -> pd.DataFrame:
    """Estimate filtered dataset sizes from streaming stats (without full DataFrame).

    Uses count dictionaries to estimate how many items/users survive each threshold.
    This is an upper-bound estimate (doesn't account for iterative removal).

    Args:
        stats: output of stream_stats() with item_counts and user_counts
        item_thresholds: list of min-item-review thresholds
        user_thresholds: list of min-user-review thresholds

    Returns:
        DataFrame with columns: item_thresh, user_thresh, n_items, n_users, est_reviews
    """
    if item_thresholds is None:
        item_thresholds = [3, 5, 10]
    if user_thresholds is None:
        user_thresholds = [3, 5]

    # Extract count arrays from the Counter objects produced by stream_stats()
    item_counts = stats["item_counts"]
    user_counts = stats["user_counts"]
    item_vals = np.array(list(item_counts.values()))  # review counts per item
    user_vals = np.array(list(user_counts.values()))  # review counts per user

    # For each (item_thresh, user_thresh) pair, estimate survivors
    # Note: this is an upper bound because it doesn't account for the iterative
    # cascade (removing items may cause users to drop below threshold, etc.)
    rows = []
    for it in item_thresholds:
        for ut in user_thresholds:
            n_items = int((item_vals >= it).sum())
            n_users = int((user_vals >= ut).sum())
            # Upper-bound review count: sum of all reviews for surviving items
            est_reviews = int(item_vals[item_vals >= it].sum())
            rows.append({
                "item_thresh": it,
                "user_thresh": ut,
                "n_items": n_items,
                "n_users": n_users,
                "est_reviews": est_reviews,
            })

    return pd.DataFrame(rows)


def compute_bought_together_hit_rate(
    reviews_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    item_col: str = "parent_asin",
    user_col: str = "user_id",
) -> Dict[str, float]:
    """Compute bought-together hit rate.

    For each user, check if any pair of items they reviewed appears in
    each other's bought_together list.

    Args:
        reviews_df: reviews DataFrame with user_id and parent_asin
        meta_df: metadata DataFrame with parent_asin and bought_together

    Returns:
        dict with:
            hit_rate: fraction of users with at least one bought-together hit
            hit_rate_conditional: same but only for users who reviewed items
                                  that have non-empty bought_together lists
            n_users_total: total users analyzed
            n_users_with_bt_items: users who reviewed items with bought_together data
            n_users_hit: users with at least one hit
    """
    # Build lookup: for each item, what other items are in its "bought_together" list?
    # bought_together is a metadata field listing ASINs frequently co-purchased
    bt_lookup = {}
    for _, row in meta_df.iterrows():
        asin = row[item_col]
        bt = row.get("bought_together")
        if bt and isinstance(bt, list) and len(bt) > 0:
            bt_lookup[asin] = set(bt)

    # Build per-user set of all items they've reviewed
    user_items = reviews_df.groupby(user_col)[item_col].apply(set).to_dict()

    n_total = len(user_items)
    n_with_bt = 0  # Users who reviewed at least one item with bought_together data
    n_hit = 0      # Users where a bought_together recommendation matches another review

    for user, items in tqdm(user_items.items(), desc="Bought-together hits"):
        has_bt = False
        hit = False
        for item in items:
            if item in bt_lookup:
                has_bt = True
                # Set intersection: do any of this item's "bought together" ASINs
                # appear in the user's reviewed item set?
                if bt_lookup[item] & items:
                    hit = True
                    break  # One hit is enough — we only need a boolean per user
        if has_bt:
            n_with_bt += 1
        if hit:
            n_hit += 1

    return {
        "hit_rate": n_hit / n_total if n_total > 0 else 0.0,
        "hit_rate_conditional": n_hit / n_with_bt if n_with_bt > 0 else 0.0,
        "n_users_total": n_total,
        "n_users_with_bt_items": n_with_bt,
        "n_users_hit": n_hit,
    }


def cross_category_users(
    categories: List[str],
) -> Dict[str, any]:
    """Analyze user overlap across categories by streaming user_ids.

    Args:
        categories: list of category names

    Returns:
        dict with:
            category_user_counts: dict of category -> n_unique_users
            overlap_matrix: dict of (cat_a, cat_b) -> n_shared_users
            cross_category_distribution: dict of n_categories -> n_users
            n_total_unique_users: total unique users across all categories
    """
    # Step 1: Stream user_id sets for each category (avoids loading full DataFrames)
    category_users: Dict[str, set] = {}
    for cat in categories:
        print(f"[cross_category] Streaming user_ids for {cat}...")
        category_users[cat] = stream_user_ids(cat)
        print(f"  -> {len(category_users[cat]):,} unique users")

    # Step 2: For each user, count how many categories they appear in
    # This tells us about cross-category browsing behavior
    all_users = set()
    for users in category_users.values():
        all_users.update(users)

    # Distribution: how many users appear in exactly 1, 2, 3, ... categories
    user_cat_count = Counter()
    for user in tqdm(all_users, desc="Counting categories per user"):
        count = sum(1 for users in category_users.values() if user in users)
        user_cat_count[count] += 1

    # Step 3: Pairwise overlap — how many users are shared between each pair
    overlap = {}
    cats = list(categories)
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            shared = len(category_users[cats[i]] & category_users[cats[j]])
            overlap[(cats[i], cats[j])] = shared

    return {
        "category_user_counts": {cat: len(users) for cat, users in category_users.items()},
        "cross_category_distribution": dict(user_cat_count),
        "overlap_matrix": overlap,
        "n_total_unique_users": len(all_users),
    }
