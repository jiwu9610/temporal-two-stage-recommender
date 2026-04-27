"""Phase 0 data-layer invariants.

Run against the real outputs of `scripts.data.preprocessing_pipeline` for one or
more categories. Tests are auto-skipped if the artifacts aren't present yet, so
this can run safely in CI before any pipeline run.

The 8 invariants (spec):
  1. every val/test user has at least one train interaction
  2. per-user timestamp ordering: max(train.ts) <= val.ts <= test.ts (per user;
     equality allowed because identical-millisecond ties aren't future leakage)
  3. user_features were computed from train only
       (all user_features.user_id values must appear in train.user_id)
  4. item_features train aggregates were computed from train only
       (n_reviews_train per item == count in train.parquet)
  5. all split item ids use canonical parent_asin
       (every parent_asin in train/val/test must be a canonical id)
  6. no duplicate (user_id, parent_asin) anywhere within a single split
  7. feature schema identical across categories present
  8. train-catalog coverage of val/test positives is reported in split_manifest

Run from repo root:
    pytest tests/test_data_invariants.py
    pytest tests/test_data_invariants.py -k Video_Games   # one category
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.feature_store import ITEM_FEATURE_COLUMNS, USER_FEATURE_COLUMNS  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CATEGORIES = ["All_Beauty", "Video_Games", "Books", "Electronics"]

REQUIRED_FILES = [
    "interactions_clean.parquet",
    "interactions_filtered.parquet",
    "canonical_item_map.parquet",
    "train.parquet", "val.parquet", "test.parquet",
    "user_features.parquet", "item_features.parquet",
    "filtering_report.json", "split_manifest.json",
    "text_alignment_report.json", "pipeline_run.json",
]


def _has_outputs(category: str) -> bool:
    cat_dir = PROCESSED_DIR / category
    return cat_dir.exists() and all((cat_dir / f).exists() for f in REQUIRED_FILES)


def _present_categories() -> list[str]:
    return [c for c in CATEGORIES if _has_outputs(c)]


_LOAD_CACHE: dict[str, dict] = {}


def _load(category: str) -> dict:
    """Load all Phase 0 artifacts for one category. Memoized — Books'
    item_features is 3.86M rows so re-reading per test invocation would
    multiply runtime by an order of magnitude.
    """
    if category in _LOAD_CACHE:
        return _LOAD_CACHE[category]
    cat_dir = PROCESSED_DIR / category
    payload = {
        "train": pd.read_parquet(cat_dir / "train.parquet"),
        "val": pd.read_parquet(cat_dir / "val.parquet"),
        "test": pd.read_parquet(cat_dir / "test.parquet"),
        "user_features": pd.read_parquet(cat_dir / "user_features.parquet"),
        "item_features": pd.read_parquet(cat_dir / "item_features.parquet"),
        "canonical_map": pd.read_parquet(cat_dir / "canonical_item_map.parquet"),
        "split_manifest": json.loads((cat_dir / "split_manifest.json").read_text()),
        "filtering_report": json.loads((cat_dir / "filtering_report.json").read_text()),
        "text_alignment_report": json.loads((cat_dir / "text_alignment_report.json").read_text()),
        "pipeline_run": json.loads((cat_dir / "pipeline_run.json").read_text()),
    }
    _LOAD_CACHE[category] = payload
    return payload


def pytest_generate_tests(metafunc):
    # Defer category discovery to test-collection time so a missing data dir
    # produces clean per-test skips instead of a module-level collection error.
    if "category_data" in metafunc.fixturenames:
        cats = _present_categories()
        if not cats:
            metafunc.parametrize("category_data", [pytest.param(None, marks=pytest.mark.skip(
                reason="No Phase 0 outputs found under data/processed/."))])
        else:
            metafunc.parametrize("category_data", cats, indirect=True)


@pytest.fixture
def category_data(request) -> dict:
    return _load(request.param) | {"category": request.param}


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_inv1_eval_users_have_train_history(category_data):
    """Every user appearing in val or test must also appear in train."""
    train_users = set(category_data["train"]["user_id"].unique())
    for split in ["val", "test"]:
        eval_users = set(category_data[split]["user_id"].unique())
        missing = eval_users - train_users
        assert not missing, (
            f"{category_data['category']}: {len(missing)} {split} user(s) missing from train, "
            f"e.g. {sorted(missing)[:3]}"
        )


def test_inv2_per_user_timestamp_ordering(category_data):
    """For every user: max(train.ts) <= val.ts <= test.ts.

    Equality is allowed because Amazon timestamps occasionally tie when a user
    submits several reviews in the same batch -- that is not future leakage,
    both rows are equally informative about the user as of that moment.
    Strict-less-than would require an arbitrary tie-breaker baked into the
    split, which we deliberately don't do.
    """
    tr = category_data["train"][["user_id", "timestamp"]]
    va = category_data["val"][["user_id", "timestamp"]]
    te = category_data["test"][["user_id", "timestamp"]]
    tr_max = tr.groupby("user_id")["timestamp"].max()

    val_idx = va.set_index("user_id")["timestamp"]
    test_idx = te.set_index("user_id")["timestamp"]
    common_val = tr_max.index.intersection(val_idx.index)
    bad_val = (tr_max.loc[common_val] > val_idx.loc[common_val]).sum()
    assert bad_val == 0, f"{bad_val} users have train.ts_max > val.ts (future leakage)"

    common_test = val_idx.index.intersection(test_idx.index)
    bad_test = (val_idx.loc[common_test] > test_idx.loc[common_test]).sum()
    assert bad_test == 0, f"{bad_test} users have val.ts > test.ts (future leakage)"


def test_inv3_user_features_train_only(category_data):
    """user_features must only cover users present in train (train-only aggregation)."""
    uf = category_data["user_features"]
    assert "user_id" in uf.columns, "user_features must carry explicit user_id column"
    train_users = set(category_data["train"]["user_id"].unique())
    feat_users = set(uf["user_id"].unique())
    extra = feat_users - train_users
    assert not extra, f"{len(extra)} user_features rows correspond to non-train users"
    assert list(uf.columns) == USER_FEATURE_COLUMNS


def test_inv4_item_features_train_aggregates_match_train(category_data):
    """For each item, n_reviews_train in item_features must equal train count."""
    train = category_data["train"]
    feats = category_data["item_features"]
    assert "parent_asin" in feats.columns, "item_features must carry explicit parent_asin column"
    expected = train.groupby("parent_asin").size().rename("expected").to_frame()
    merged = feats[["parent_asin", "n_reviews_train"]].merge(
        expected, left_on="parent_asin", right_index=True, how="left"
    )
    merged["expected"] = merged["expected"].fillna(0).astype(int)
    mismatch = (merged["n_reviews_train"].astype(int) != merged["expected"]).sum()
    assert mismatch == 0, f"{mismatch} items have n_reviews_train != train count"


def test_inv5_split_ids_use_canonical_parent_asin(category_data):
    """Every parent_asin in train/val/test must be a canonical id from the map."""
    canonical = set(category_data["canonical_map"]["canonical_parent_asin"].unique())
    for split in ["train", "val", "test"]:
        ids = set(category_data[split]["parent_asin"].unique())
        non_canonical = ids - canonical
        assert not non_canonical, (
            f"{split}: {len(non_canonical)} parent_asin not in canonical map, "
            f"e.g. {sorted(non_canonical)[:3]}"
        )


def test_inv6_no_duplicate_user_item_pairs(category_data):
    for split in ["train", "val", "test"]:
        df = category_data[split]
        n_dup = df.duplicated(["user_id", "parent_asin"]).sum()
        assert n_dup == 0, f"{split}: {n_dup} duplicate (user_id, parent_asin) rows"


def test_inv7_schema_identical_across_categories():
    """user_features and item_features schemas must be identical across all categories present."""
    cats = _present_categories()
    if len(cats) < 2:
        pytest.skip(f"Only {len(cats)} category present; need >=2 for cross-category schema check.")
    user_cols, item_cols = None, None
    for c in cats:
        d = _load(c)
        u = list(d["user_features"].columns)
        i = list(d["item_features"].columns)
        if user_cols is None:
            user_cols, item_cols = u, i
            continue
        assert u == user_cols, f"{c}: user_features schema differs from {cats[0]}"
        assert i == item_cols, f"{c}: item_features schema differs from {cats[0]}"
    assert user_cols == USER_FEATURE_COLUMNS
    assert item_cols == ITEM_FEATURE_COLUMNS


def test_inv8_split_manifest_reports_train_catalog_coverage(category_data):
    """split_manifest.coverage must include the train-catalog coverage rates."""
    cov = category_data["split_manifest"]["coverage"]
    for key in [
        "n_train_catalog_items",
        "val_positive_items_in_train_catalog_rate",
        "test_positive_items_in_train_catalog_rate",
        "val_positive_items_missing_from_train",
        "test_positive_items_missing_from_train",
    ]:
        assert key in cov, f"split_manifest.coverage missing {key!r}"
    for k in ["val_positive_items_in_train_catalog_rate",
              "test_positive_items_in_train_catalog_rate"]:
        assert 0.0 <= float(cov[k]) <= 1.0, f"{k} out of [0,1]: {cov[k]}"
    for split in ["val", "test"]:
        s = category_data["split_manifest"]["splits"][split]
        for key in ["n_positive_users", "n_positive_items"]:
            assert key in s, f"split_manifest.splits.{split} missing {key!r}"
