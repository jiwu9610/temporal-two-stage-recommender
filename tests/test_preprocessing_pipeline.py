"""Unit tests for the Phase 0 preprocessing modules.

Covers (on small synthetic inputs; the only disk reads are repo-committed
config files):
  - canonicalize.canonicalize (canon v2 merge key = (norm_title, store))
  - the FROZEN canon-v2 normalization predicate
  - the pinned three_snapshot cutoffs in configs/preprocessing.yaml
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
import yaml

# Make `scripts.*` importable when running pytest from repo root or tests/ dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.canonicalize import canonicalize, normalize_title, store_merge_key
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
    # Canon v2 merge key = (normalized title, store):
    #   B1, B2 share ("sequel", "S1")            -> merge; B1 wins on rating_number.
    #   B5 title "  SEQUEL  " store "S1"          -> normalizes into the same key, merges into B1.
    #   B4 shares B3's title but store "S9" != "S2" -> cross-store, must NOT merge.
    #   ZZ has null title                          -> never merges.
    return pd.DataFrame({
        "parent_asin": ["B1", "B2", "B3", "ZZ", "B4", "B5"],
        "title": ["Sequel", "Sequel", "Original", None, "Original", "  SEQUEL  "],
        "rating_number": [100, 30, 50, 5, 40, 10],
        "store": ["S1", "S1", "S2", "S3", "S9", "S1"],
        "main_category": ["Cat", "Cat", "Cat", None, "Cat", "Cat"],
        "price": ["10.00", "10.00", None, "5", None, "9.50"],
        "average_rating": [4.5, 4.5, 4.0, 3.0, 3.5, 4.2],
        "features": [["a", "b"], ["a"], [], ["x", "y", "z"], [], ["a"]],
        "description": [["desc"], [], ["d1", "d2"], [], [], []],
        "categories": [["c1"], ["c1"], ["c1", "c2"], [], ["c1"], ["c1"]],
        "bought_together": [["other"], [], [], None, None, None],
    })


@pytest.fixture
def synthetic_reviews() -> pd.DataFrame:
    # Users:
    #   U1: 5 reviews on B1, B3, B2 (dup via merge), B5 (dup via merge), ZZ.
    #       After canonicalization (B2->B1, B5->B1) and dedup -> 3 unique items.
    #   U2: 4 reviews on B1, B3, ZZ, B4 — all canonical, distinct (B4 stays B4).
    #   U3: 2 reviews on B3, ZZ — too few for leave-last-two if min_user>=3.
    rows = []
    rows += [
        ("U1", "B1", 5, 1_000_000),
        ("U1", "B3", 4, 1_000_100),
        ("U1", "B2", 2, 1_000_200),  # B2 -> B1 by canonicalization, then dropped as dup
        ("U1", "B5", 4, 1_000_250),  # B5 -> B1 (normalized title), then dropped as dup
        ("U1", "ZZ", 1, 1_000_300),
        ("U2", "B1", 3, 2_000_000),
        ("U2", "B3", 5, 2_000_100),
        ("U2", "ZZ", 4, 2_000_200),
        ("U2", "B4", 5, 2_000_300),  # same title as B3 but different store: stays B4
        ("U3", "B3", 2, 3_000_000),
        ("U3", "ZZ", 5, 3_000_100),
    ]
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------

def test_canonicalize_collapses_same_store_title_duplicates(synthetic_reviews, synthetic_meta):
    clean, canon_map, canon_meta, stats = canonicalize(
        synthetic_reviews, synthetic_meta, positive_threshold=3.0
    )
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict["B2"] == "B1"
    assert map_dict["B5"] == "B1"   # "  SEQUEL  " normalizes to "sequel", same store S1
    assert map_dict["B4"] == "B4"   # same title as B3 but different store: no merge
    assert map_dict["B3"] == "B3"
    assert set(canon_meta["parent_asin"]) == {"B1", "B3", "ZZ", "B4"}
    u1_items = set(clean[clean["user_id"] == "U1"]["parent_asin"])
    assert u1_items == {"B1", "B3", "ZZ"}
    u2_items = set(clean[clean["user_id"] == "U2"]["parent_asin"])
    assert u2_items == {"B1", "B3", "ZZ", "B4"}
    assert (clean.loc[clean["rating"] >= 3, "label"] == 1).all()
    assert (clean.loc[clean["rating"] < 3, "label"] == 0).all()
    assert set(clean["label_type"]) <= {"positive", "hard_negative"}
    assert not clean.duplicated(["user_id", "parent_asin"]).any()
    assert stats.duplicates_removed >= 2
    assert stats.canonical_meta == 4
    # Key ("sequel", "S1") had B1+B2+B5 -> 1 group with >1 parent_asin; B2+B5 collapsed.
    # ("original", "S2") vs ("original", "S9") are DIFFERENT keys -> no group.
    assert stats.n_parent_asins_collapsed_by_title == 2
    assert stats.n_title_duplicate_groups == 1


def _tiny_reviews(asins) -> pd.DataFrame:
    # One reviewer per item, distinct timestamps -- enough to drive canonicalize.
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(len(asins))],
        "parent_asin": list(asins),
        "rating": [5] * len(asins),
        "timestamp": list(range(1_000, 1_000 + len(asins))),
    })


def test_canon_v2_cross_store_same_title_never_merges():
    # The Live2Pedal-style C1 bug: identical generic title sold by different
    # stores. v1 folded these into one id; v2 must keep them apart.
    meta = pd.DataFrame({
        "parent_asin": ["A1", "A2", "A3"],
        "title": ["Sustain Pedal", "Sustain Pedal", "Sustain Pedal"],
        "store": ["Live2Pedal", "OtherBrand", "ThirdBrand"],
        "rating_number": [500, 400, 300],
    })
    _, canon_map, canon_meta, stats = canonicalize(_tiny_reviews(meta["parent_asin"]), meta)
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict == {"A1": "A1", "A2": "A2", "A3": "A3"}
    assert set(canon_meta["parent_asin"]) == {"A1", "A2", "A3"}
    assert stats.n_parent_asins_collapsed_by_title == 0
    assert stats.n_title_duplicate_groups == 0


def test_canon_v2_null_or_empty_title_or_store_never_merges():
    meta = pd.DataFrame({
        "parent_asin": ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"],
        # N1/N2: same title, null store. N3/N4: null title, same store.
        # N5/N6: whitespace-only / empty title, same store.
        # N7/N8: same title, whitespace-only / empty store.
        "title": ["Journal", "Journal", None, None, "   ", "", "Diary", "Diary"],
        "store": [None, None, "S", "S", "S", "S", "  ", ""],
        "rating_number": [8, 7, 6, 5, 4, 3, 2, 1],
    })
    _, canon_map, _, stats = canonicalize(_tiny_reviews(meta["parent_asin"]), meta)
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict == {pa: pa for pa in meta["parent_asin"]}
    assert stats.n_parent_asins_collapsed_by_title == 0
    assert stats.n_title_duplicate_groups == 0


def test_canon_v2_missing_store_column_merges_nothing():
    # Guard symmetric to the missing-title guard: without a store column the
    # v2 key cannot be formed -- falling back to title-only would reintroduce
    # cross-store merges, so nothing merges at all.
    meta = pd.DataFrame({
        "parent_asin": ["A1", "A2"],
        "title": ["Same Title", "Same Title"],
        "rating_number": [10, 5],
    })
    _, canon_map, _, stats = canonicalize(_tiny_reviews(meta["parent_asin"]), meta)
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict == {"A1": "A1", "A2": "A2"}
    assert stats.n_parent_asins_collapsed_by_title == 0
    assert stats.n_title_duplicate_groups == 0


def test_canon_v2_missing_title_column_merges_nothing():
    meta = pd.DataFrame({
        "parent_asin": ["A1", "A2"],
        "store": ["S", "S"],
        "rating_number": [10, 5],
    })
    _, canon_map, _, stats = canonicalize(_tiny_reviews(meta["parent_asin"]), meta)
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict == {"A1": "A1", "A2": "A2"}
    assert stats.n_parent_asins_collapsed_by_title == 0


def test_canon_v2_normalization_predicate_frozen():
    # FROZEN: the S1
    # rebuild gate numbers are predicate-sensitive (+-0.5-1.5% observed between
    # variants, e.g. Books rows 12.55% vs 12.82%). Any change to this predicate
    # invalidates the recorded canon_v2_audit numbers and requires re-running
    # scripts/analysis/canon_v2_audit/ and regenerating the recorded numbers.
    #
    # Title leg: lowercase, strip, collapse ANY unicode-whitespace run to one space.
    assert normalize_title("  Foo   BAR ") == "foo bar"
    assert normalize_title("foo\t\n bar") == "foo bar"
    assert normalize_title("Foo\u00a0Bar") == "foo bar"    # NBSP is whitespace too
    assert normalize_title("FOO") == "foo"
    assert normalize_title("Straße") == "straße"   # str.lower, NOT casefold
    assert normalize_title(None) == ""
    assert normalize_title(float("nan")) == ""
    assert normalize_title("") == ""
    assert normalize_title("   ") == ""                       # whitespace-only never merges
    # Store leg: verbatim string -- case-, punctuation- and whitespace-sensitive.
    assert store_merge_key("Live2Pedal") == "Live2Pedal"
    assert store_merge_key(" Live2Pedal ") == " Live2Pedal "
    assert store_merge_key("STORE") == "STORE"                # no case-folding
    assert store_merge_key(None) == ""
    assert store_merge_key("") == ""
    assert store_merge_key("   ") == ""                       # whitespace-only never merges


def test_canon_v2_winner_highest_rating_number_ties_by_asin():
    meta = pd.DataFrame({
        # Group 1 ("dup", "S"): Q7 wins on rating_number outright.
        # Group 2 ("tie", "S"): all tie at 50 -> lexicographically smallest asin (A1) wins.
        "parent_asin": ["Z9", "Q7", "M5", "T2", "A1", "T9"],
        "title": ["Dup", "Dup", "Dup", "Tie", "Tie", "Tie"],
        "store": ["S", "S", "S", "S", "S", "S"],
        "rating_number": [50, 60, 40, 50, 50, 50],
    })
    _, canon_map, canon_meta, stats = canonicalize(_tiny_reviews(meta["parent_asin"]), meta)
    map_dict = dict(zip(canon_map["raw_parent_asin"], canon_map["canonical_parent_asin"]))
    assert map_dict["Z9"] == "Q7" and map_dict["M5"] == "Q7" and map_dict["Q7"] == "Q7"
    assert map_dict["T2"] == "A1" and map_dict["T9"] == "A1" and map_dict["A1"] == "A1"
    assert set(canon_meta["parent_asin"]) == {"Q7", "A1"}
    assert stats.n_parent_asins_collapsed_by_title == 4
    assert stats.n_title_duplicate_groups == 2


# ---------------------------------------------------------------------------
# pinned three_snapshot cutoffs
# ---------------------------------------------------------------------------

# Epoch-ms values from each category's pre-rebuild
# data/processed/{cat}/temporal_ranker/snapshot_manifest.json, hardcoded here so
# the pin survives the S1/S2 artifact rebuilds. All_Beauty t3 is the approved
# exception: manifest T3 (2023-09-09T00:39:36.667Z) - 166d, whole-second pin.
PINNED_THREE_SNAPSHOT_CUTOFFS_MS = {
    "All_Beauty":  {"t0": 1607218967372, "t1": 1621824434801, "t2": 1643166829879, "t3": 1679877576000},
    "Video_Games": {"t0": 1580172986322, "t1": 1610504438807, "t2": 1643626178086, "t3": 1680478398072},
    "Books":       {"t0": 1543330798937, "t1": 1578278676145, "t2": 1623073350040, "t3": 1679444512032},
    "Electronics": {"t0": 1588262994150, "t1": 1615924132589, "t2": 1646434378144, "t3": 1679345305320},
}


def test_three_snapshot_cutoffs_pinned_roundtrip():
    """The yaml pins must round-trip through resolve_four_cutoffs to EXACTLY
    the pre-rebuild manifest ms values (AB t3 to its approved T3-166d pin)."""
    from scripts.data.temporal_ranker_pipeline import resolve_four_cutoffs

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "preprocessing.yaml").read_text())
    three = cfg["temporal_snapshots"]["three_snapshot"]
    explicit_all = three.get("cutoffs") or {}
    # Timestamps far below every pin: if t3 ever fell back to max_ms+1 the
    # asserts below would fail loudly instead of silently floating.
    dummy_clean = pd.DataFrame({"timestamp": [0, 10_000]})

    for category, expected in PINNED_THREE_SNAPSHOT_CUTOFFS_MS.items():
        explicit = explicit_all.get(category)
        assert explicit, f"{category}: three_snapshot.cutoffs pin missing from preprocessing.yaml"
        for k in ("t0", "t1", "t2", "t3"):
            assert explicit.get(k), f"{category}: cutoff {k} not pinned explicitly"
            assert isinstance(explicit[k], str), (
                f"{category}: cutoff {k} must be a quoted ISO string, got {type(explicit[k])}"
            )
        resolved = resolve_four_cutoffs(
            dummy_clean,
            explicit=explicit,
            quantiles=three.get("cutoff_quantiles"),
        )
        assert resolved["source"] == "explicit"
        for k in ("t0", "t1", "t2", "t3"):
            assert resolved[f"{k}_ms"] == expected[k], (
                f"{category} {k}: yaml round-trips to {resolved[f'{k}_ms']}, "
                f"pinned manifest value is {expected[k]}"
            )


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
