"""Snapshot-specific feature stores for global temporal mode.

Every interaction-derived aggregate is recomputed PER SNAPSHOT from that
snapshot's history only (events strictly before the snapshot cutoff), with the
cutoff itself as the "as of" reference time. Aggregates carry the ``_hist``
suffix -- the temporal analog of Phase 0's ``_train`` suffix -- so the origin
of every column stays auditable.

Deliberately EXCLUDED from item features in temporal mode:
    average_rating, rating_number
These come from the 2023 metadata dump and aggregate the item's ENTIRE review
history -- future interactions relative to any historical cutoff. The
history-derived replacements are avg_rating_hist / n_reviews_hist.

Static catalog attributes (main_category, store, price, list-length counts,
has_bought_together, missing_flag) are kept: they are not interaction-derived.
Assumption to keep documented: the dump reflects them as of 2023 and we treat
them as time-invariant.

Temporal features (only these three for now, per the migration plan):
    days_since_last_interaction   (as_of - last history event) / days
    interactions_last_30d         history events in [as_of - 30d, as_of)
    positives_last_30d            history positives in the same window
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .temporal_split import MS_PER_DAY


USER_SNAPSHOT_FEATURE_COLUMNS = [
    "user_id",
    "n_reviews_hist",
    "avg_rating_hist",
    "std_rating_hist",
    "n_unique_items_hist",
    "active_days_hist",
    "n_verified_hist",
    "verified_rate_hist",
    "avg_helpful_vote_hist",
    "total_helpful_votes_hist",
    "days_since_last_interaction",
    "interactions_last_30d",
    "positives_last_30d",
]

ITEM_SNAPSHOT_FEATURE_COLUMNS = [
    "parent_asin",
    "main_category",
    "store",
    "price",
    "n_features",
    "n_description",
    "n_categories",
    "has_bought_together",
    "missing_flag",
    "n_reviews_hist",
    "n_positives_hist",
    "avg_rating_hist",
    "n_unique_reviewers_hist",
    "in_history_catalog",
    "in_eligible_pool",
]

# Columns lifted verbatim from the Phase 0 item_features.parquet catalog part.
CATALOG_STATIC_COLUMNS = [
    "parent_asin",
    "main_category",
    "store",
    "price",
    "n_features",
    "n_description",
    "n_categories",
    "has_bought_together",
    "missing_flag",
]

_CATALOG_DEFAULTS = {
    "main_category": "Unknown",
    "store": "Unknown",
    "price": 0.0,
    "n_features": 0,
    "n_description": 0,
    "n_categories": 0,
    "has_bought_together": 0,
    "missing_flag": 1,
}


def build_snapshot_user_features(
    history_df: pd.DataFrame,
    as_of_ms: int,
    recent_days: int = 30,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """Per-user aggregates from snapshot history only.

    Only users WITH history get a row: cold users have nothing to aggregate.
    Downstream must treat missing users as cold (zero history), never drop them.
    """
    if (history_df[time_col] >= as_of_ms).any():
        raise ValueError(
            "history_df contains events at/after as_of_ms -- snapshot history "
            "must be strictly before the cutoff (temporal leakage guard)"
        )

    g = history_df.groupby(user_col)
    feats = pd.DataFrame({
        "n_reviews_hist": g.size(),
        "avg_rating_hist": g[rating_col].mean(),
        "std_rating_hist": g[rating_col].std().fillna(0.0),
        "n_unique_items_hist": g[item_col].nunique(),
    })
    ts_min = g[time_col].min()
    ts_max = g[time_col].max()
    feats["active_days_hist"] = (ts_max - ts_min) / MS_PER_DAY

    if "verified_purchase" in history_df.columns:
        vp = g["verified_purchase"]
        feats["n_verified_hist"] = vp.sum().astype(int)
        feats["verified_rate_hist"] = vp.mean().astype(float)
    else:
        feats["n_verified_hist"] = 0
        feats["verified_rate_hist"] = 0.0

    if "helpful_vote" in history_df.columns:
        hv = g["helpful_vote"]
        feats["avg_helpful_vote_hist"] = hv.mean().astype(float)
        feats["total_helpful_votes_hist"] = hv.sum().astype(int)
    else:
        feats["avg_helpful_vote_hist"] = 0.0
        feats["total_helpful_votes_hist"] = 0

    # -- basic temporal features, all relative to the snapshot cutoff ---------
    feats["days_since_last_interaction"] = (as_of_ms - ts_max) / MS_PER_DAY
    recent_start = as_of_ms - recent_days * MS_PER_DAY
    recent = history_df[history_df[time_col] >= recent_start]
    recent_counts = recent.groupby(user_col).size()
    recent_pos_counts = recent[recent["label"] == 1].groupby(user_col).size()
    feats[f"interactions_last_{recent_days}d"] = (
        recent_counts.reindex(feats.index).fillna(0).astype(int)
    )
    feats[f"positives_last_{recent_days}d"] = (
        recent_pos_counts.reindex(feats.index).fillna(0).astype(int)
    )

    feats.index.name = user_col
    feats = feats.reset_index()
    if recent_days == 30:
        feats = feats[USER_SNAPSHOT_FEATURE_COLUMNS]
    return feats


def build_snapshot_item_features(
    catalog_df: pd.DataFrame,
    history_df: pd.DataFrame,
    eligible_items: pd.Index,
    as_of_ms: int,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    rating_col: str = "rating",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Catalog statics + history-only aggregates + snapshot membership flags.

    `catalog_df` is the Phase 0 item_features.parquet (or any frame carrying
    the CATALOG_STATIC_COLUMNS subset); interaction-derived dump columns
    (average_rating, rating_number, *_train, in_*_catalog flags) are ignored.
    Rows cover the full catalog so cold items remain scoreable downstream.
    """
    if (history_df[time_col] >= as_of_ms).any():
        raise ValueError(
            "history_df contains events at/after as_of_ms -- snapshot history "
            "must be strictly before the cutoff (temporal leakage guard)"
        )
    if item_col not in catalog_df.columns:
        raise ValueError(f"catalog_df missing column {item_col!r}")

    items = catalog_df.copy()
    for col in CATALOG_STATIC_COLUMNS:
        if col == item_col:
            continue
        if col not in items.columns:
            items[col] = _CATALOG_DEFAULTS[col]
    items = items[CATALOG_STATIC_COLUMNS].drop_duplicates(subset=[item_col])
    items = items.set_index(item_col)

    g = history_df.groupby(item_col)
    agg = pd.DataFrame({
        "n_reviews_hist": g.size(),
        "n_positives_hist": history_df[history_df["label"] == 1]
        .groupby(item_col).size(),
        "avg_rating_hist": g[rating_col].mean(),
        "n_unique_reviewers_hist": g[user_col].nunique(),
    })
    items = items.join(agg, how="left")
    items["n_reviews_hist"] = items["n_reviews_hist"].fillna(0).astype(int)
    items["n_positives_hist"] = items["n_positives_hist"].fillna(0).astype(int)
    items["n_unique_reviewers_hist"] = (
        items["n_unique_reviewers_hist"].fillna(0).astype(int)
    )
    # No history -> no rating information at this point in time. 0.0 is an
    # explicit "unknown" sentinel; the dump's average_rating would be leakage.
    items["avg_rating_hist"] = items["avg_rating_hist"].fillna(0.0)

    history_catalog = set(history_df[item_col].unique())
    items["in_history_catalog"] = items.index.isin(history_catalog).astype(int)
    items["in_eligible_pool"] = items.index.isin(set(eligible_items)).astype(int)

    items.index.name = item_col
    items = items.reset_index()
    items = items[ITEM_SNAPSHOT_FEATURE_COLUMNS]
    # Alphabetical row order: the incoming catalog is sorted by descending
    # 2023 rating_number (a whole-timeline statistic), and downstream scorers
    # break ties positionally. Persisting a future-free order makes the
    # stored artifact safe for any consumer.
    items = items.sort_values(item_col, kind="mergesort").reset_index(drop=True)
    return items
