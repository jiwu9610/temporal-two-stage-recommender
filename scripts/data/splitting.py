"""Per-user chronological leave-last-two split.

Spec: for each user with N >= 3 interactions, sorted by timestamp:
    train = first N-2 (chronological)
    val   = N-1 (second-to-last)
    test  = N   (last)

Filtering must enforce min_user_interactions >= 3 upstream so every user split
has a well-defined train history and one held-out row per eval split.

The split manifest reports per-split counts, label distribution, timestamp
ranges, and -- crucially for retrieval evaluation -- how many of the val/test
positive items also appear in the train catalog. Items missing from the train
catalog cannot be retrieved by popularity / item-id-based methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class SplitManifest:
    strategy: str
    n_users_total: int
    splits: Dict[str, dict]
    coverage: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_users_total": self.n_users_total,
            "splits": self.splits,
            "coverage": self.coverage,
        }


def _split_summary(
    df: pd.DataFrame, name: str, user_col: str, item_col: str, time_col: str
) -> dict:
    if len(df) == 0:
        return {
            "name": name,
            "n_rows": 0,
            "n_users": 0,
            "n_items": 0,
            "label_positive_rate": 0.0,
            "ts_min": None,
            "ts_max": None,
        }
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_users": int(df[user_col].nunique()),
        "n_items": int(df[item_col].nunique()),
        "label_positive_rate": float(df["label"].mean()),
        "ts_min": int(df[time_col].min()),
        "ts_max": int(df[time_col].max()),
    }


def leave_last_two_split(
    filtered_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Split filtered interactions per user into (train, val, test).

    Returns
    -------
    train, val, test : DataFrames with the same columns as `filtered_df`.
    manifest : SplitManifest with per-split + coverage stats.
    """
    # Stable sort with parent_asin as tertiary key makes the split deterministic
    # across runs even when a user has multiple interactions at the same
    # timestamp (Amazon timestamps occasionally tie when reviews are batch-posted).
    # The ts ordering invariant downstream allows `<=` because identical-timestamp
    # rows aren't future leakage; the parent_asin tiebreaker just removes
    # nondeterminism in *which* row lands in val vs test.
    df = filtered_df.sort_values(
        [user_col, time_col, item_col], kind="mergesort"
    ).reset_index(drop=True)

    # Per-user reverse rank: 0 = last interaction, 1 = second-to-last, ...
    rev_rank = df.groupby(user_col).cumcount(ascending=False)
    n_per_user = df.groupby(user_col)[user_col].transform("count")
    df = df.assign(_rev_rank=rev_rank, _n=n_per_user)

    # Defensive guard: every user should already have >= 3 interactions because of
    # the iterative filter. Drop any that slipped through (and warn via manifest).
    short_users = int((df["_n"] < 3).any() and df.loc[df["_n"] < 3, user_col].nunique())
    df = df[df["_n"] >= 3]

    test = df[df["_rev_rank"] == 0].drop(columns=["_rev_rank", "_n"]).reset_index(drop=True)
    val = df[df["_rev_rank"] == 1].drop(columns=["_rev_rank", "_n"]).reset_index(drop=True)
    train = df[df["_rev_rank"] >= 2].drop(columns=["_rev_rank", "_n"]).reset_index(drop=True)

    n_users_total = int(df[user_col].nunique())

    # Per-split summaries
    splits = {
        "train": _split_summary(train, "train", user_col, item_col, time_col),
        "val": _split_summary(val, "val", user_col, item_col, time_col),
        "test": _split_summary(test, "test", user_col, item_col, time_col),
    }
    splits["short_users_dropped"] = short_users  # 0 in normal runs

    # Positive-only counts on val / test
    val_pos = val[val["label"] == 1]
    test_pos = test[test["label"] == 1]
    splits["train"]["n_positive_users"] = int(train[train["label"] == 1][user_col].nunique())
    splits["train"]["n_positive_items"] = int(train[train["label"] == 1][item_col].nunique())
    splits["val"]["n_positive_users"] = int(val_pos[user_col].nunique())
    splits["val"]["n_positive_items"] = int(val_pos[item_col].nunique())
    splits["test"]["n_positive_users"] = int(test_pos[user_col].nunique())
    splits["test"]["n_positive_items"] = int(test_pos[item_col].nunique())

    # Train-catalog coverage of val / test positives
    train_catalog = set(train[item_col].unique())  # FULL train, not positives-only
    val_pos_items = set(val_pos[item_col].unique())
    test_pos_items = set(test_pos[item_col].unique())
    coverage = {
        "n_train_catalog_items": len(train_catalog),
        "val_positive_items_in_train_catalog_rate": (
            len(val_pos_items & train_catalog) / len(val_pos_items)
            if val_pos_items else 0.0
        ),
        "test_positive_items_in_train_catalog_rate": (
            len(test_pos_items & train_catalog) / len(test_pos_items)
            if test_pos_items else 0.0
        ),
        "val_positive_items_missing_from_train": int(len(val_pos_items - train_catalog)),
        "test_positive_items_missing_from_train": int(len(test_pos_items - train_catalog)),
    }

    manifest = SplitManifest(
        strategy="per_user_leave_last_two",
        n_users_total=n_users_total,
        splits=splits,
        coverage=coverage,
    )
    return train, val, test, manifest
