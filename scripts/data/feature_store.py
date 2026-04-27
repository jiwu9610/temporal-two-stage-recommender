"""Train-only feature stores with unified schema across categories.

Two outputs:

user_features.parquet
    columns:
        user_id                                        (explicit + index)
        n_reviews_train, avg_rating_train, std_rating_train,
        n_unique_items_train, active_days_train,
        n_verified_train, verified_rate_train,
        avg_helpful_vote_train, total_helpful_votes_train

item_features.parquet
    columns:
        parent_asin                                    (explicit + index)
        main_category, store, price                    catalog metadata
        average_rating, rating_number                  catalog metadata
        n_features, n_description, n_categories        list-length metadata features
        has_bought_together                            binary
        missing_flag                                   1 if any of {main_category, store, price} was filled
        n_reviews_train, avg_rating_train, n_unique_reviewers_train

Train-derived aggregates carry the `_train` suffix; metadata columns keep
their natural names. Schemas are identical across all 4 categories: any source
column missing from the raw data is filled with 0 / "Unknown" / 0-list and the
column still appears, so downstream code never has to branch on category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Canonical column orderings — enforced, so categories cannot diverge.
USER_FEATURE_COLUMNS = [
    "user_id",
    "n_reviews_train",
    "avg_rating_train",
    "std_rating_train",
    "n_unique_items_train",
    "active_days_train",
    "n_verified_train",
    "verified_rate_train",
    "avg_helpful_vote_train",
    "total_helpful_votes_train",
]

ITEM_FEATURE_COLUMNS = [
    "parent_asin",
    "main_category",
    "store",
    "price",
    "average_rating",
    "rating_number",
    "n_features",
    "n_description",
    "n_categories",
    "has_bought_together",
    "missing_flag",
    "n_reviews_train",
    "avg_rating_train",
    "n_unique_reviewers_train",
    # Catalog-membership flags. item_features is the FULL canonical catalog so
    # downstream code can score cold items if it wants; these flags let it
    # explicitly choose its candidate universe without re-deriving membership.
    "in_filtered_universe",          # item appears in interactions_filtered (post k-core)
    "in_train_catalog",              # item appears in train.parquet (any label)
    "in_train_positive_catalog",     # item appears in train.parquet with label==1
]


def build_user_features(
    train_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """Aggregate train interactions into per-user features.

    `train_df` must come from the train split (no val/test rows). All aggregates
    carry `_train` suffix to make this contract auditable downstream.
    """
    g = train_df.groupby(user_col)
    feats = pd.DataFrame({
        "n_reviews_train": g.size(),
        "avg_rating_train": g[rating_col].mean(),
        "std_rating_train": g[rating_col].std().fillna(0.0),
        "n_unique_items_train": g[item_col].nunique(),
    })
    # active_days: span between first and last review. Amazon ts is ms since epoch.
    ts_min = g[time_col].min()
    ts_max = g[time_col].max()
    feats["active_days_train"] = (ts_max - ts_min) / (1000.0 * 86400.0)

    # verified_purchase / helpful_vote may be missing from some categories'
    # raw schemas — fill with neutral defaults so the column set is invariant.
    if "verified_purchase" in train_df.columns:
        vp = g["verified_purchase"]
        feats["n_verified_train"] = vp.sum().astype(int)
        feats["verified_rate_train"] = vp.mean().astype(float)
    else:
        feats["n_verified_train"] = 0
        feats["verified_rate_train"] = 0.0

    if "helpful_vote" in train_df.columns:
        hv = g["helpful_vote"]
        feats["avg_helpful_vote_train"] = hv.mean().astype(float)
        feats["total_helpful_votes_train"] = hv.sum().astype(int)
    else:
        feats["avg_helpful_vote_train"] = 0.0
        feats["total_helpful_votes_train"] = 0

    feats.index.name = user_col
    feats = feats.reset_index()
    feats = feats[USER_FEATURE_COLUMNS]
    return feats


def _coerce_price(series: pd.Series) -> pd.Series:
    """Parse Amazon's price field. Handles strings like '$12.99' and pure numerics."""
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^\d.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _list_len(value) -> int:
    if isinstance(value, (list, np.ndarray)):
        return len(value)
    return 0


def build_item_features(
    canonical_meta: pd.DataFrame,
    train_df: pd.DataFrame,
    filtered_df: pd.DataFrame | None = None,
    item_col: str = "parent_asin",
    user_col: str = "user_id",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """Combine catalog metadata + train-only review aggregates into item features.

    `canonical_meta` should already be deduplicated to one row per canonical
    parent_asin (output of canonicalize.py). `train_df` is the train split.
    `filtered_df` is the post-k-core universe (output of filtering.py); if
    provided, sets the `in_filtered_universe` flag. If omitted, that flag is 0
    for every row.
    """
    if item_col not in canonical_meta.columns:
        raise ValueError(f"canonical_meta missing column {item_col!r}")

    meta = canonical_meta.copy()

    # ---- catalog scalar columns (with default fills, missing_flag tracking) ----
    for col, default in [("main_category", "Unknown"), ("store", "Unknown")]:
        if col not in meta.columns:
            meta[col] = default
        else:
            meta[col] = meta[col].fillna(default)

    if "price" not in meta.columns:
        meta["price"] = np.nan
    meta["price"] = _coerce_price(meta["price"])

    if "average_rating" not in meta.columns:
        meta["average_rating"] = np.nan
    meta["average_rating"] = pd.to_numeric(meta["average_rating"], errors="coerce")

    if "rating_number" not in meta.columns:
        meta["rating_number"] = 0
    meta["rating_number"] = pd.to_numeric(meta["rating_number"], errors="coerce").fillna(0).astype(int)

    # missing_flag *before* any numeric fill, so it reflects raw missingness.
    meta["missing_flag"] = (
        (meta["main_category"] == "Unknown").astype(int)
        | (meta["store"] == "Unknown").astype(int)
        | meta["price"].isna().astype(int)
    ).astype(int)

    # Now fill numeric defaults for downstream model use.
    price_fill = float(meta["price"].median()) if meta["price"].notna().any() else 0.0
    meta["price"] = meta["price"].fillna(price_fill)
    avg_rating_fill = float(meta["average_rating"].median()) if meta["average_rating"].notna().any() else 0.0
    meta["average_rating"] = meta["average_rating"].fillna(avg_rating_fill)

    # ---- list-length features --------------------------------------------------
    for src, out in [("features", "n_features"), ("description", "n_description"),
                     ("categories", "n_categories")]:
        if src in meta.columns:
            meta[out] = meta[src].apply(_list_len).astype(int)
        else:
            meta[out] = 0

    if "bought_together" in meta.columns:
        meta["has_bought_together"] = meta["bought_together"].apply(
            lambda x: 1 if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
        ).astype(int)
    else:
        meta["has_bought_together"] = 0

    # ---- train-only review aggregates -----------------------------------------
    g = train_df.groupby(item_col).agg(
        n_reviews_train=(rating_col, "count"),
        avg_rating_train=(rating_col, "mean"),
        n_unique_reviewers_train=(user_col, "nunique"),
    )
    items = meta.set_index(item_col).join(g, how="left")
    items["n_reviews_train"] = items["n_reviews_train"].fillna(0).astype(int)
    items["n_unique_reviewers_train"] = items["n_unique_reviewers_train"].fillna(0).astype(int)
    # Items with no train reviews fall back to catalog average_rating.
    items["avg_rating_train"] = items["avg_rating_train"].fillna(items["average_rating"])

    # ---- catalog membership flags ---------------------------------------------
    train_catalog = set(train_df[item_col].unique())
    train_pos_catalog = set(train_df.loc[train_df.get("label", 0) == 1, item_col].unique()) \
        if "label" in train_df.columns else set()
    filtered_catalog = (
        set(filtered_df[item_col].unique()) if filtered_df is not None else set()
    )
    idx = items.index
    items["in_filtered_universe"] = idx.isin(filtered_catalog).astype(int)
    items["in_train_catalog"] = idx.isin(train_catalog).astype(int)
    items["in_train_positive_catalog"] = idx.isin(train_pos_catalog).astype(int)

    items.index.name = item_col
    items = items.reset_index()
    items = items[ITEM_FEATURE_COLUMNS]
    return items
