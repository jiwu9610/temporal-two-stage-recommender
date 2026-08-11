"""Shared helpers for the canon-v2 audit scripts (Track D, REBUILD_WAVE_SPEC P2.1).

Everything here reads RAW inputs only (data/raw/{category}/) and never touches
data/processed frozen artifacts or results/. The merge predicate is imported
from the SHIPPED scripts.data.canonicalize -- the audit numbers must come from
the exact predicate that S1 will run with (it is frozen in
tests/test_preprocessing_pipeline.py::test_canon_v2_normalization_predicate_frozen).
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
AUDIT_DIR = Path(__file__).resolve().parent

CATEGORIES = ("All_Beauty", "Video_Games", "Books", "Electronics")

META_AUDIT_COLUMNS = ["parent_asin", "title", "store", "rating_number"]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_meta_dedup(category: str) -> pd.DataFrame:
    """Raw metadata restricted to the audit columns, deduplicated on
    parent_asin EXACTLY like scripts.data.canonicalize step 1: highest
    rating_number first, ties by parent_asin ascending, keep first."""
    path = RAW_DIR / category / "metadata.parquet"
    assert path.exists(), f"{path} not found -- raw metadata is a hard requirement"
    available = set(pq.read_schema(path).names)
    cols = [c for c in META_AUDIT_COLUMNS if c in available]
    assert "parent_asin" in cols and "title" in cols and "store" in cols, (
        f"{category} metadata is missing audit columns: have {sorted(cols)} -- "
        f"refresh_metadata.py should have produced the full 16-col schema"
    )
    meta = pq.read_table(path, columns=cols).to_pandas()
    meta = meta.dropna(subset=["parent_asin"])
    if "rating_number" in meta.columns:
        meta = meta.sort_values(["rating_number", "parent_asin"], ascending=[False, True])
    meta = meta.drop_duplicates(subset=["parent_asin"], keep="first").reset_index(drop=True)
    return meta


def review_row_counts(category: str) -> Tuple[pd.Series, int]:
    """(review rows per raw parent_asin, total review rows). Reads the single
    parent_asin column via pyarrow so Books stays within a 64G sbatch node."""
    path = RAW_DIR / category / "reviews.parquet"
    assert path.exists(), f"{path} not found -- raw reviews is a hard requirement"
    col = pq.read_table(path, columns=["parent_asin"]).column("parent_asin")
    vc = col.value_counts()
    counts = pd.Series(
        vc.field("counts").to_numpy(zero_copy_only=False),
        index=pd.Index(vc.field("values").to_pandas(), name="parent_asin"),
    )
    total = int(len(col))
    assert int(counts.sum()) <= total  # nulls (if any) are not counted
    return counts, total


def mapping_from_key(
    meta_dedup: pd.DataFrame,
    key_values,
    mergeable_mask,
    item_col: str = "parent_asin",
) -> Tuple[pd.Series, int]:
    """Raw -> canonical mapping for an arbitrary merge key.

    Winner = first row in `meta_dedup` order within each key group, i.e.
    highest rating_number with parent_asin-ascending tie-break (load
    meta_dedup via load_meta_dedup). Non-mergeable rows map to themselves.

    Returns (canonical Series indexed by raw parent_asin, n_multi_asin_groups).
    """
    asins = meta_dedup[item_col].to_numpy()
    keys = np.asarray(key_values, dtype=object)
    mask = np.asarray(mergeable_mask, dtype=bool)
    assert len(asins) == len(keys) == len(mask)

    keyed = pd.DataFrame({item_col: asins[mask], "_key": keys[mask]})
    winner_per_key = keyed.groupby("_key", sort=False)[item_col].first()
    group_sizes = keyed.groupby("_key", sort=False).size()
    n_multi_groups = int((group_sizes > 1).sum())

    canonical = asins.copy()
    canonical[mask] = keyed["_key"].map(winner_per_key).to_numpy()
    return pd.Series(canonical, index=pd.Index(asins, name=item_col)), n_multi_groups


def summarize_mapping(
    canonical: pd.Series,
    counts: pd.Series,
    total_review_rows: int,
    n_multi_groups: int,
) -> dict:
    """Option-comparison numbers for one raw -> canonical mapping."""
    raw = canonical.index.to_numpy()
    remapped_mask = canonical.to_numpy() != raw
    rows_per_asin = counts.reindex(canonical.index).fillna(0).to_numpy()
    rows_on_catalog = int(rows_per_asin.sum())
    rows_remapped = int(rows_per_asin[remapped_mask].sum())
    return {
        "n_multi_asin_groups": n_multi_groups,
        "n_parent_asins_folded": int(remapped_mask.sum()),
        "n_canonical_items": int(canonical.nunique()),
        "n_review_rows_remapped": rows_remapped,
        "pct_review_rows_remapped_of_catalog_rows": round(100.0 * rows_remapped / rows_on_catalog, 4)
        if rows_on_catalog else 0.0,
        "pct_review_rows_remapped_of_all_rows": round(100.0 * rows_remapped / total_review_rows, 4)
        if total_review_rows else 0.0,
        "n_review_rows_on_catalog": rows_on_catalog,
    }
