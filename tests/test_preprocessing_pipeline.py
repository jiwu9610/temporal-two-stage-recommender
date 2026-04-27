"""Unit tests for the Phase 0 preprocessing modules.

Covers (on small synthetic inputs, no disk I/O):
  - canonicalize.canonicalize
  - filtering.iterative_kcore
  - splitting.leave_last_two_split
  - feature_store.build_user_features / build_item_features
  - text_alignment.align_text_embeddings

Run from repo root:
    pytest tests/test_preprocessing_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `scripts.*` importable when running pytest from repo root or tests/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data.canonicalize import canonicalize
from scripts.data.feature_store import (
    ITEM_FEATURE_COLUMNS,
    USER_FEATURE_COLUMNS,
    build_item_features,
    build_user_features,
)
from scripts.data.filtering import iterative_kcore
from scripts.data.splitting import leave_last_two_split
from scripts.data.text_alignment import align_text_embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_meta() -> pd.DataFrame:
    # Two title-duplicate items (B1, B2 share "Sequel") -- B1 wins on rating_number.
    # B3 is its own canonical item. ZZ has null title.
    return pd.DataFrame({
        "parent_asin": ["B1", "B2", "B3", "ZZ"],
        "title": ["Sequel", "Sequel", "Original", None],
        "rating_number": [100, 30, 50, 5],
        "store": ["S1", "S1", "S2", "S3"],
        "main_category": ["Cat", "Cat", "Cat", None],
        "price": ["10.00", "10.00", None, "5"],
        "average_rating": [4.5, 4.5, 4.0, 3.0],
        "features": [["a", "b"], ["a"], [], ["x", "y", "z"]],
        "description": [["desc"], [], ["d1", "d2"], []],
        "categories": [["c1"], ["c1"], ["c1", "c2"], []],
        "bought_together": [["other"], [], [], None],
    })


@pytest.fixture
def synthetic_reviews() -> pd.DataFrame:
    # Users:
    #   U1: 4 reviews on B1, B3, B2 (duplicate), ZZ. After canonicalization (B2->B1)
    #       and dedup (U1,B1) becomes 3 unique items.
    #   U2: 3 reviews on B1, B3, ZZ — all canonical, distinct.
    #   U3: 2 reviews on B3, ZZ — too few for leave-last-two if min_user>=3.
    rows = []
    rows += [
        ("U1", "B1", 5, 1_000_000),
        ("U1", "B3", 4, 1_000_100),
        ("U1", "B2", 2, 1_000_200),  # B2 -> B1 by canonicalization, then dropped as dup
        ("U1", "ZZ", 1, 1_000_300),
        ("U2", "B1", 3, 2_000_000),
        ("U2", "B3", 5, 2_000_100),
        ("U2", "ZZ", 4, 2_000_200),
        ("U3", "B3", 2, 3_000_000),
        ("U3", "ZZ", 5, 3_000_100),
    ]
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------

def test_canonicalize_collapses_title_duplicates(synthetic_reviews, synthetic_meta):
    clean, canon_map, canon_meta, stats = canonicalize(
        synthetic_reviews, synthetic_meta, positive_threshold=3.0
    )
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict["B2"] == "B1"
    assert set(canon_meta["parent_asin"]) == {"B1", "B3", "ZZ"}
    u1_items = set(clean[clean["user_id"] == "U1"]["parent_asin"])
    assert u1_items == {"B1", "B3", "ZZ"}
    assert (clean.loc[clean["rating"] >= 3, "label"] == 1).all()
    assert (clean.loc[clean["rating"] < 3, "label"] == 0).all()
    assert set(clean["label_type"]) <= {"positive", "hard_negative"}
    assert not clean.duplicated(["user_id", "parent_asin"]).any()
    assert stats.duplicates_removed >= 1
    assert stats.canonical_meta == 3
    # Title "Sequel" had B1+B2 -> 1 group with >1 parent_asin; B2 collapsed.
    assert stats.n_parent_asins_collapsed_by_title == 1
    assert stats.n_title_duplicate_groups == 1


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def test_iterative_kcore_converges_and_reports():
    # 5 dense users x 5 dense items = 25 interactions: 5/item, 5/user -> all survive.
    rows = []
    for u in ["UA", "UB", "UC", "UD", "UE"]:
        for it in ["I1", "I2", "I3", "I4", "I5"]:
            rows.append((u, it, 5, 0))
    # noise: a lonely user with one interaction (must be pruned by min_user=3),
    # and a lonely item with one interaction (must be pruned by min_item=5).
    rows.append(("U_lonely", "I1", 5, 0))
    rows.append(("UA", "I_lonely", 5, 0))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])

    filtered, report = iterative_kcore(
        df, min_user_interactions=3, min_item_interactions=5, max_iterations=10
    )
    assert filtered["user_id"].value_counts().min() >= 3
    assert filtered["parent_asin"].value_counts().min() >= 5
    assert "U_lonely" not in set(filtered["user_id"])
    assert "I_lonely" not in set(filtered["parent_asin"])
    assert report.converged is True
    assert report.n_iterations >= 1
    assert report.initial["n_interactions"] == len(df)
    assert report.final["n_interactions"] == len(filtered)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def test_leave_last_two_split_per_user_chronological():
    rows = []
    for ts, it in enumerate(["I1", "I2", "I3", "I4", "I5"], start=1):
        rows.append(("UA", it, 5, ts))
    for ts, it in enumerate(["I9", "I8", "I7"], start=10):
        rows.append(("UB", it, 4, ts))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    df["label"] = (df["rating"] >= 3).astype(int)

    train, val, test, manifest = leave_last_two_split(df)
    assert sorted(train[train["user_id"] == "UA"]["parent_asin"]) == ["I1", "I2", "I3"]
    assert val[val["user_id"] == "UA"]["parent_asin"].iloc[0] == "I4"
    assert test[test["user_id"] == "UA"]["parent_asin"].iloc[0] == "I5"
    assert manifest.coverage["val_positive_items_in_train_catalog_rate"] == 0.0
    assert manifest.coverage["test_positive_items_in_train_catalog_rate"] == 0.0
    for uid in ["UA", "UB"]:
        tr_max = train[train["user_id"] == uid]["timestamp"].max()
        v = val[val["user_id"] == uid]["timestamp"].iloc[0]
        t = test[test["user_id"] == uid]["timestamp"].iloc[0]
        assert tr_max < v < t


# ---------------------------------------------------------------------------
# feature_store
# ---------------------------------------------------------------------------

def test_user_features_schema_and_train_suffix():
    train = pd.DataFrame({
        "user_id": ["U1", "U1", "U2", "U2", "U2"],
        "parent_asin": ["I1", "I2", "I1", "I3", "I4"],
        "rating": [5, 4, 3, 2, 5],
        "timestamp": [1, 2, 3, 4, 5],
        "verified_purchase": [True, False, True, True, False],
        "helpful_vote": [10, 0, 1, 2, 5],
    })
    feats = build_user_features(train)
    assert list(feats.columns) == USER_FEATURE_COLUMNS
    assert "user_id" in feats.columns
    assert all(c.endswith("_train") for c in feats.columns if c != "user_id")
    u1 = feats[feats["user_id"] == "U1"].iloc[0]
    assert u1["n_reviews_train"] == 2
    assert u1["n_unique_items_train"] == 2


def test_item_features_unified_schema_with_missing_metadata(synthetic_meta):
    train = pd.DataFrame({
        "user_id": ["U1", "U2", "U2"],
        "parent_asin": ["B1", "B1", "B3"],
        "rating": [5, 4, 1],                # B3 only has a hard-negative train row
        "label": [1, 1, 0],
        "timestamp": [1, 2, 3],
    })
    # Filtered universe excludes ZZ (e.g. it didn't survive k-core).
    filtered = pd.DataFrame({
        "user_id": ["U1", "U2", "U2", "U3"],
        "parent_asin": ["B1", "B1", "B3", "B3"],
        "rating": [5, 4, 1, 5],
        "label": [1, 1, 0, 1],
        "timestamp": [1, 2, 3, 4],
    })
    items = build_item_features(synthetic_meta, train, filtered_df=filtered)
    assert list(items.columns) == ITEM_FEATURE_COLUMNS
    assert "parent_asin" in items.columns
    b1 = items[items["parent_asin"] == "B1"].iloc[0]
    assert b1["n_reviews_train"] == 2
    assert b1["n_unique_reviewers_train"] == 2
    assert int(b1["in_filtered_universe"]) == 1
    assert int(b1["in_train_catalog"]) == 1
    assert int(b1["in_train_positive_catalog"]) == 1
    b3 = items[items["parent_asin"] == "B3"].iloc[0]
    assert int(b3["in_train_catalog"]) == 1
    # B3's only train row is label=0 -> not in positive catalog.
    assert int(b3["in_train_positive_catalog"]) == 0
    zz = items[items["parent_asin"] == "ZZ"].iloc[0]
    assert zz["n_reviews_train"] == 0
    assert int(zz["missing_flag"]) == 1
    assert int(zz["in_filtered_universe"]) == 0
    assert int(zz["in_train_catalog"]) == 0
    assert int(zz["in_train_positive_catalog"]) == 0


# ---------------------------------------------------------------------------
# text_alignment
# ---------------------------------------------------------------------------

def test_text_alignment_remaps_and_collapses(tmp_path):
    src_path = tmp_path / "title_bert.npz"
    np.savez(
        src_path,
        asins=np.array(["B1", "B2", "B3"], dtype=object),
        embs=np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    cmap = pd.DataFrame({
        "raw_parent_asin": ["B1", "B2", "B3"],
        "canonical_parent_asin": ["B1", "B1", "B3"],
    })
    item_features = pd.DataFrame({"parent_asin": ["B1", "B3"]})
    out_path = tmp_path / "aligned.npz"

    report = align_text_embeddings(src_path, item_features, cmap, out_path)
    assert report.status == "rebuilt"
    assert report.n_groups_collapsed == 1
    assert report.n_aligned_to_item_features == 2
    assert report.n_items_missing_embedding == 0

    arr = np.load(out_path, allow_pickle=True)
    asins = arr["asins"].astype(str)
    embs = arr["embs"]
    assert list(asins) == ["B1", "B3"]
    np.testing.assert_allclose(embs[0], [2.0, 0.0])
    np.testing.assert_allclose(embs[1], [0.0, 1.0])


def test_text_alignment_handles_missing_source(tmp_path):
    item_features = pd.DataFrame({"parent_asin": ["A", "B", "C"]})
    cmap = pd.DataFrame({"raw_parent_asin": ["A", "B", "C"], "canonical_parent_asin": ["A", "B", "C"]})
    out_path = tmp_path / "aligned.npz"
    report = align_text_embeddings(None, item_features, cmap, out_path)
    assert report.status == "not_present"
    assert report.n_items_missing_embedding == 3
    assert out_path.exists()
