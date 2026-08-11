"""Canonicalize raw Amazon reviews + metadata to the Phase 0 interaction universe.

Produces three artifacts:

1. canonical_item_map: DataFrame with columns [raw_parent_asin, canonical_parent_asin].
   - For most items raw == canonical.
   - Canon v2 merge key (approved 2026-08-11, MEMO D5): two parent_asins are the
     same product iff they share (normalized title, store). Title normalization
     is lowercase / strip / whitespace-collapse; a null or whitespace-only title
     OR a null / whitespace-only store NEVER merges. The v1 title-only rule
     produced Live2Pedal-style cross-store merges (same generic title, different
     sellers folded into one id -- the C1 poisoned-identity bug).
   - Within a merge group we collapse to the parent_asin with the highest
     rating_number (ties broken by parent_asin ascending, deterministic);
     raw -> canonical records the mapping.

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
    # Field names kept stable across canon v1 -> v2 (pipeline_run.json schema);
    # since v2 the unit of merging is the (norm_title, store) key, not the title.
    n_parent_asins_collapsed_by_title: int      # how many raw parent_asins were folded into another canonical id
    n_title_duplicate_groups: int                # how many distinct (norm_title, store) keys had >1 parent_asin
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


def normalize_title(title) -> str:
    """Canon v2 title leg of the merge key: lowercase, strip, collapse every
    internal whitespace run (any unicode whitespace) to a single space.

    Null / whitespace-only titles normalize to "" and NEVER merge -- each such
    parent_asin stays its own canonical id.

    FROZEN predicate: the S1 rebuild gate numbers are sensitive to this exact
    definition (spec section 1 Track D correction 2; observed +-0.5-1.5% swing
    between predicate variants). It is frozen by
    tests/test_preprocessing_pipeline.py::test_canon_v2_normalization_predicate_frozen;
    do not change without re-running the canon_v2_audit scripts and
    re-approving the recorded numbers.
    """
    if pd.isna(title):
        return ""
    return " ".join(str(title).lower().split())


def store_merge_key(store) -> str:
    """Canon v2 store leg of the merge key: the raw store string, verbatim
    (case-, punctuation- and inner-whitespace-sensitive).

    Null / whitespace-only stores yield "" and NEVER merge -- an unknown
    seller is not evidence of shared identity. Frozen alongside
    normalize_title (same test).
    """
    if pd.isna(store):
        return ""
    text = str(store)
    return text if text.strip() else ""


def _build_canonical_map(meta: pd.DataFrame, item_col: str) -> Tuple[dict, int, int]:
    """Build raw -> canonical parent_asin mapping on the canon v2 merge key
    (normalize_title(title), store_merge_key(store)).

    Assumes `meta` is already deduplicated on `item_col` and sorted with the
    "best" candidate (highest rating_number, ties by `item_col` ascending)
    first within each merge-key group.

    Returns
    -------
    (mapping, n_parent_asins_collapsed_by_title, n_title_duplicate_groups)
        mapping : raw parent_asin -> canonical parent_asin
        n_parent_asins_collapsed_by_title : how many raw parent_asins ended up
            mapped to a different canonical parent_asin
        n_title_duplicate_groups : how many distinct (norm_title, store) keys
            had >1 parent_asin
    """
    if "title" not in meta.columns:
        return {pa: pa for pa in meta[item_col]}, 0, 0
    if "store" not in meta.columns:
        # Guard symmetric to the title guard above: without a store column the
        # v2 merge key cannot be formed, and falling back to title-only would
        # silently reintroduce the cross-store (C1) merges. Merge nothing.
        return {pa: pa for pa in meta[item_col]}, 0, 0

    norm_titles = meta["title"].map(normalize_title)
    store_keys = meta["store"].map(store_merge_key)
    mergeable_mask = (norm_titles != "") & (store_keys != "")

    keyed = pd.DataFrame(
        {
            item_col: meta[item_col].to_numpy(),
            "_norm_title": norm_titles.to_numpy(),
            "_store_key": store_keys.to_numpy(),
        }
    )[mergeable_mask.to_numpy()]
    canonical_per_key = keyed.groupby(["_norm_title", "_store_key"])[item_col].first().to_dict()
    key_group_sizes = keyed.groupby(["_norm_title", "_store_key"]).size()
    n_duplicate_groups = int((key_group_sizes > 1).sum())

    mapping: dict = {}
    n_collapsed = 0
    for pa, nt, sk, ok in zip(meta[item_col], norm_titles, store_keys, mergeable_mask):
        if ok:
            canon = canonical_per_key[(nt, sk)]
        else:
            canon = pa
        mapping[pa] = canon
        if canon != pa:
            n_collapsed += 1
    return mapping, n_collapsed, n_duplicate_groups


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
        # Highest rating_number first; ties broken by item id ascending so the
        # surviving metadata row AND the merge-group winner are deterministic.
        meta = meta.sort_values(["rating_number", item_col], ascending=[False, True])
    meta = meta.drop_duplicates(subset=[item_col], keep="first").reset_index(drop=True)
    meta_after_parent_dedup = len(meta)

    # ---- 2. metadata: build canonical map ((norm_title, store) collapse) --------
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
