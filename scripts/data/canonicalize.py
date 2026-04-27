"""Canonicalize raw Amazon reviews + metadata to the Phase 0 interaction universe.

Produces three artifacts:

1. canonical_item_map: DataFrame with columns [raw_parent_asin, canonical_parent_asin].
   - For most items raw == canonical.
   - When two parent_asins share an identical title (Amazon catalog merge artifact),
     we collapse them to the parent_asin with the most reviews; raw -> canonical
     records the mapping.

2. canonical_metadata: deduplicated metadata, indexed by canonical parent_asin only.

3. interactions_clean: cleaned + canonicalized interaction rows.
   - drop rows missing user_id / parent_asin / rating / timestamp
   - remap parent_asin via canonical_item_map (so an interaction on a "raw" duplicate
     ends up on the canonical id)
   - drop interactions whose item has no metadata at all (orphans)
   - dedup (user_id, canonical_parent_asin) keeping the earliest timestamp
   - sort by (user_id, timestamp)
   - add `label` (1 if rating >= positive_threshold else 0) and `label_type`
     ("positive" or "hard_negative")

`label_type == "hard_negative"` means: user *did* interact, but rated < threshold.
Soft negatives (no interaction at all) are NOT created here -- downstream training
samples them when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class CanonicalizeStats:
    raw_reviews: int
    raw_meta: int
    meta_after_parent_dedup: int
    n_parent_asins_collapsed_by_title: int      # how many raw parent_asins were folded into another canonical id
    n_title_duplicate_groups: int                # how many distinct titles had >1 parent_asin sharing them
    canonical_meta: int
    reviews_after_dropna: int
    reviews_after_canonical_remap: int
    reviews_orphans_dropped: int
    reviews_after_dedup: int
    duplicates_removed: int
    n_unique_users: int
    n_unique_items: int
    positive_rate: float
    hard_negative_rate: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _build_canonical_map(meta: pd.DataFrame, item_col: str) -> Tuple[dict, int, int]:
    """Build raw -> canonical parent_asin mapping based on title duplicates.

    Assumes `meta` is already deduplicated on `item_col` and sorted with the
    "best" candidate (highest rating_number) first within each title group.

    Returns
    -------
    (mapping, n_parent_asins_collapsed_by_title, n_title_duplicate_groups)
        mapping : raw parent_asin -> canonical parent_asin
        n_parent_asins_collapsed_by_title : how many raw parent_asins ended up
            mapped to a different canonical parent_asin
        n_title_duplicate_groups : how many distinct titles had >1 parent_asin
    """
    if "title" not in meta.columns:
        return {pa: pa for pa in meta[item_col]}, 0, 0

    has_title = meta.dropna(subset=["title"])
    canonical_per_title = has_title.groupby("title")[item_col].first().to_dict()
    title_group_sizes = has_title.groupby("title").size()
    n_title_duplicate_groups = int((title_group_sizes > 1).sum())

    mapping: dict = {}
    n_collapsed = 0
    for _, row in meta.iterrows():
        pa = row[item_col]
        title = row.get("title")
        if pd.notna(title) and title in canonical_per_title:
            canon = canonical_per_title[title]
            mapping[pa] = canon
            if canon != pa:
                n_collapsed += 1
        else:
            mapping[pa] = pa
    return mapping, n_collapsed, n_title_duplicate_groups


def canonicalize(
    reviews_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    positive_threshold: float = 3.0,
    item_col: str = "parent_asin",
    user_col: str = "user_id",
    time_col: str = "timestamp",
    rating_col: str = "rating",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CanonicalizeStats]:
    """Run cleanup + canonicalization.

    Returns
    -------
    (interactions_clean, canonical_item_map, canonical_metadata, stats)
        interactions_clean : DataFrame of cleaned + canonicalized interactions with
            columns [user_id, parent_asin, rating, label, label_type, timestamp,
            verified_purchase, helpful_vote] (the last two passed through if present).
        canonical_item_map : DataFrame [raw_parent_asin, canonical_parent_asin].
        canonical_metadata : DataFrame indexed-on-`parent_asin` of catalog rows that
            survived dedup. parent_asin column is also kept explicitly.
        stats : CanonicalizeStats — reportable counts at each step.
    """
    raw_reviews = len(reviews_df)
    raw_meta = len(meta_df)

    # ---- 1. metadata: dedup by parent_asin (keep highest rating_number) ---------
    meta = meta_df.dropna(subset=[item_col]).copy()
    if "rating_number" in meta.columns:
        meta = meta.sort_values("rating_number", ascending=False)
    meta = meta.drop_duplicates(subset=[item_col], keep="first").reset_index(drop=True)
    meta_after_parent_dedup = len(meta)

    # ---- 2. metadata: build canonical map (title-duplicate collapse) ------------
    raw_to_canonical, n_collapsed, n_title_groups = _build_canonical_map(meta, item_col)
    canonical_winners = set(v for v in raw_to_canonical.values())
    canonical_meta = meta[meta[item_col].isin(canonical_winners)].reset_index(drop=True)

    canonical_map_df = pd.DataFrame(
        {
            "raw_parent_asin": list(raw_to_canonical.keys()),
            "canonical_parent_asin": list(raw_to_canonical.values()),
        }
    ).sort_values(["canonical_parent_asin", "raw_parent_asin"]).reset_index(drop=True)

    # ---- 3. reviews: drop nulls on critical columns ----------------------------
    critical = [user_col, item_col, rating_col, time_col]
    rev = reviews_df.dropna(subset=critical).copy()
    reviews_after_dropna = len(rev)

    # ---- 4. reviews: remap parent_asin via canonical map -----------------------
    # Items in reviews but not in our metadata stay as-is (we'll drop next step).
    rev[item_col] = rev[item_col].map(lambda pa: raw_to_canonical.get(pa, pa))
    reviews_after_canonical_remap = len(rev)

    # ---- 5. reviews: drop orphan interactions (item not in canonical metadata) -
    canonical_item_set = set(canonical_meta[item_col])
    n_before = len(rev)
    rev = rev[rev[item_col].isin(canonical_item_set)].copy()
    reviews_orphans_dropped = n_before - len(rev)

    # ---- 6. reviews: sort + dedup (user, item) keeping earliest ----------------
    rev = rev.sort_values([user_col, time_col]).reset_index(drop=True)
    n_before = len(rev)
    rev = rev.drop_duplicates(subset=[user_col, item_col], keep="first").reset_index(drop=True)
    duplicates_removed = n_before - len(rev)

    # ---- 7. reviews: add label + label_type ------------------------------------
    rev["label"] = (rev[rating_col] >= positive_threshold).astype(np.int8)
    rev["label_type"] = np.where(rev["label"] == 1, "positive", "hard_negative")

    # Project to a stable column set (carry helpful_vote / verified_purchase if present).
    keep = [user_col, item_col, rating_col, "label", "label_type", time_col]
    for opt in ["verified_purchase", "helpful_vote"]:
        if opt in rev.columns:
            keep.append(opt)
    interactions_clean = rev[keep].reset_index(drop=True)

    # Make canonical_meta carry parent_asin both as column and (re-built later) as index downstream.
    canonical_meta = canonical_meta.reset_index(drop=True)

    stats = CanonicalizeStats(
        raw_reviews=raw_reviews,
        raw_meta=raw_meta,
        meta_after_parent_dedup=meta_after_parent_dedup,
        n_parent_asins_collapsed_by_title=n_collapsed,
        n_title_duplicate_groups=n_title_groups,
        canonical_meta=len(canonical_meta),
        reviews_after_dropna=reviews_after_dropna,
        reviews_after_canonical_remap=reviews_after_canonical_remap,
        reviews_orphans_dropped=reviews_orphans_dropped,
        reviews_after_dedup=len(interactions_clean),
        duplicates_removed=duplicates_removed,
        n_unique_users=int(interactions_clean[user_col].nunique()),
        n_unique_items=int(interactions_clean[item_col].nunique()),
        positive_rate=float(interactions_clean["label"].mean()),
        hard_negative_rate=float(1.0 - interactions_clean["label"].mean()),
    )

    return interactions_clean, canonical_map_df, canonical_meta, stats
