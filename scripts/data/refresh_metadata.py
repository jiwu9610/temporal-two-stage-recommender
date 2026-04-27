"""Re-pull full metadata from HF and rebuild item_features without rerunning splits.

Motivated by Phase 1 finding: `data/raw/{Video_Games,Books,Electronics}/metadata.parquet`
was saved with a truncated 8-column schema (no `features`, `description`,
`categories`, ...). All list-length columns in `item_features.parquet`
(`n_features`, `n_description`, `n_categories`, `has_bought_together`) are
therefore uniformly zero for those three categories, which silently degrades
any rule-based / two-tower model that consumes them.

This script:

  1. Re-pulls the full HF parquet shards (`raw_meta_{category}/full-*.parquet`)
     for the three affected categories with the complete column set.
  2. Backs up the existing 8-column file as `metadata.parquet.8col_backup`
     (skipped if a backup already exists -- idempotent).
  3. Rebuilds `data/processed/{category}/item_features.parquet` using the new
     full metadata, restricted to the canonical parent_asin set already
     committed in `canonical_item_map.parquet`. Splits, the filtered
     universe, and canonical keys are NOT changed.
  4. Writes `data/processed/{category}/metadata_schema_report.json` for ALL
     four categories, documenting available columns, null rates, and which
     expected metadata features end up nonzero.

CLI:
    python -m scripts.data.refresh_metadata
        --categories Video_Games Books Electronics
        [--report-only]                    # skip re-pull + rebuild, only write reports
        [--skip-repull]                    # rebuild item_features from existing local meta
        [--no-backup]                      # don't keep the 8-column backup
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from scripts.data.feature_store import build_item_features


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

ALL_CATEGORIES = ["All_Beauty", "Video_Games", "Books", "Electronics"]
DEFAULT_REPULL_CATEGORIES = ["Video_Games", "Books", "Electronics"]

# Columns we expect to populate downstream features. Anything else from HF is
# kept too -- HF's full schema is 16 cols and disk is cheap on this HPC.
EXPECTED_LIST_COLUMNS = ["features", "description", "categories", "bought_together"]


def _list_hf_metadata_shards(category: str) -> List[str]:
    """Return parquet shard paths under ``raw_meta_{category}/`` on HF, or [] if
    that directory doesn't exist (only ~9 categories have it; Books and
    Video_Games only ship as JSONL under ``raw/meta_categories/``)."""
    api = HfApi()
    try:
        files = list(api.list_repo_tree(
            HF_REPO_ID, repo_type="dataset", path_in_repo=f"raw_meta_{category}",
        ))
    except Exception:
        return []
    return sorted(f.path for f in files if f.path.endswith(".parquet"))


def _concat_local_parquet_shards_to_file(
    shard_paths: List[Path],
    out_path: Path,
    parent_asin_filter: Optional[Set[str]] = None,
) -> int:
    """Stream-concat a list of local parquet shards into a single output parquet
    via pyarrow ``ParquetWriter``, never materializing the full dataset in RAM.

    Returns the number of rows written. Optionally filters rows where
    `parent_asin` is not in `parent_asin_filter`.
    """
    if not shard_paths:
        raise RuntimeError("no shards to write")

    schemas = [pq.read_schema(sp) for sp in shard_paths]
    unified = pa.unify_schemas(schemas, promote_options="permissive")

    n_written = 0
    with pq.ParquetWriter(out_path, unified, compression="snappy") as writer:
        for sp in shard_paths:
            t = pq.read_table(sp)
            # Add any missing columns as nulls so the table conforms to `unified`.
            for f in unified:
                if f.name not in t.column_names:
                    t = t.append_column(
                        f.name, pa.nulls(t.num_rows, type=f.type)
                    )
            t = t.select([f.name for f in unified])
            try:
                t = t.cast(unified, safe=False)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # Schema unify failed to produce castable types -- fall back to
                # via-pandas cast (slower, more permissive).
                t = pa.Table.from_pandas(t.to_pandas(), schema=unified, safe=False)
            if parent_asin_filter is not None and "parent_asin" in t.column_names:
                pa_col = t.column("parent_asin")
                # Build a boolean mask without round-tripping to Python.
                mask = pa.compute.is_in(
                    pa_col, value_set=pa.array(list(parent_asin_filter))
                )
                t = t.filter(mask)
            writer.write_table(t)
            n_written += t.num_rows
    return n_written


def _read_meta_from_parquet_shards_to_file(
    category: str,
    out_path: Path,
    parent_asin_filter: Optional[Set[str]] = None,
) -> int:
    shards = _list_hf_metadata_shards(category)
    print(f"[repull] {category}: {len(shards)} HF parquet shard(s) -> downloading",
          flush=True)
    local_shards: List[Path] = []
    for pf in shards:
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=pf, repo_type="dataset")
        local_shards.append(Path(local))
    return _concat_local_parquet_shards_to_file(
        local_shards, out_path, parent_asin_filter=parent_asin_filter,
    )


def _stream_jsonl_to_parquet(
    category: str,
    out_path: Path,
    parent_asin_filter: Optional[Set[str]] = None,
    chunk_size: int = 250_000,
) -> int:
    """Stream the HF meta JSONL into a single parquet file at `out_path` via
    intermediate parquet shards, never materializing the full dataset in RAM.

    Returns the number of rows written. With Books at 14.7 GB / 4.4M rows,
    the previous pandas concat+to_parquet path peaked >96 GB and got killed;
    this path peaks at ~one shard worth (~5 GB).
    """
    import tempfile
    print(f"[repull] {category}: streaming JSONL from HF "
          f"(filter={'on' if parent_asin_filter else 'off'})...", flush=True)
    jsonl_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        repo_type="dataset",
    )

    tmpdir = Path(tempfile.mkdtemp(prefix=f"refresh_meta_{category}_"))
    shard_paths: List[Path] = []

    def _flush(rows_buf: List[dict]) -> None:
        if not rows_buf:
            return
        sp = tmpdir / f"shard_{len(shard_paths):05d}.parquet"
        df_chunk = pd.DataFrame(rows_buf)
        # Amazon's `price` is a free-text field ("$12.99", "from 14.99",
        # numeric, None). Force it to string so the schema is stable across
        # shards.
        if "price" in df_chunk.columns:
            df_chunk["price"] = df_chunk["price"].astype("string")
        df_chunk.to_parquet(sp, index=False)
        shard_paths.append(sp)

    rows_buf: List[dict] = []
    n_seen = 0
    n_kept = 0
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=f"meta_{category}"):
            n_seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parent_asin_filter is not None:
                pa_id = row.get("parent_asin")
                if pa_id is None or pa_id not in parent_asin_filter:
                    continue
            rows_buf.append(row)
            n_kept += 1
            if len(rows_buf) >= chunk_size:
                _flush(rows_buf)
                rows_buf = []
    _flush(rows_buf)
    rows_buf = []

    print(f"[repull] {category}: parsed {n_seen:,} lines, kept {n_kept:,} "
          f"in {len(shard_paths)} shard(s); concatenating to final parquet...",
          flush=True)
    if not shard_paths:
        # Empty result -- write an empty parquet so callers don't see a missing file.
        pd.DataFrame().to_parquet(out_path, index=False)
        return 0

    n_written = _concat_local_parquet_shards_to_file(
        shard_paths, out_path, parent_asin_filter=None,
    )

    for sp in shard_paths:
        sp.unlink()
    tmpdir.rmdir()
    return n_written


def _download_full_metadata_to_file(
    category: str,
    out_path: Path,
    parent_asin_filter: Optional[Set[str]] = None,
) -> int:
    """Download the full-schema metadata for `category` and write a single
    parquet file at `out_path` without materializing the full DataFrame.

    Uses HF parquet shards when available; otherwise streams the meta JSONL.
    Returns the number of rows written.
    """
    shards = _list_hf_metadata_shards(category)
    if shards:
        return _read_meta_from_parquet_shards_to_file(
            category, out_path, parent_asin_filter=parent_asin_filter,
        )
    return _stream_jsonl_to_parquet(
        category, out_path, parent_asin_filter=parent_asin_filter,
    )


def _canonical_raw_asin_filter(category: str) -> Set[str]:
    """Set of every parent_asin recorded in this category's existing
    canonical_item_map.parquet. Used as a JSONL streaming filter so we don't
    materialize 14 GB of Books metadata when the canonical universe is what we
    care about."""
    canon = pd.read_parquet(
        PROCESSED_DIR / category / "canonical_item_map.parquet"
    )
    return set(canon["raw_parent_asin"].astype(str))


def _repull_one(category: str, keep_backup: bool = True) -> Path:
    """Re-pull full metadata for one category. Returns the path to the new file.

    Streams source shards/JSONL directly to the destination parquet via
    pyarrow ``ParquetWriter`` -- never holds the full table in RAM. Books
    (4.4M rows, 14.7 GB JSONL) used to OOM the 96 GB sbatch on the previous
    pandas-concat path.
    """
    out_path = RAW_DIR / category / "metadata.parquet"
    backup_path = RAW_DIR / category / "metadata.parquet.8col_backup"

    asin_filter = _canonical_raw_asin_filter(category)
    print(f"[repull] {category}: canonical raw_parent_asin universe = "
          f"{len(asin_filter):,}", flush=True)

    # Back up BEFORE writing so we don't write to the same path twice.
    if keep_backup and out_path.exists() and not backup_path.exists():
        out_path.rename(backup_path)
        print(f"[repull] {category}: backed up old meta -> {backup_path.name}",
              flush=True)
    elif not keep_backup and out_path.exists():
        out_path.unlink()

    # If a previous run left a partial file, clear it.
    if out_path.exists():
        out_path.unlink()

    n_rows = _download_full_metadata_to_file(
        category, out_path, parent_asin_filter=asin_filter,
    )
    n_cols = len(pq.read_schema(out_path).names)
    print(f"[repull] {category}: wrote new full metadata -> {out_path}  "
          f"({n_rows:,} rows, {n_cols} cols)", flush=True)
    return out_path


def _restrict_to_canonical(meta_df: pd.DataFrame,
                           canonical_map_df: pd.DataFrame,
                           item_col: str = "parent_asin") -> pd.DataFrame:
    """Reproduce the canonicalize.py dedup, then restrict to the existing
    canonical winners so canonical keys stay identical to what splits saw.

    canonicalize.py logic (replayed):
      - dropna(parent_asin)
      - sort by rating_number desc (tie-broken by current order)
      - drop_duplicates(parent_asin)

    Then we restrict to the set of canonical_parent_asin values from the
    saved canonical_item_map, which is the contract train/val/test were
    built against. If a canonical key is missing from the new metadata
    (shouldn't happen for the same HF dataset), we surface it loudly.
    """
    meta = meta_df.dropna(subset=[item_col]).copy()
    if "rating_number" in meta.columns:
        meta = meta.sort_values("rating_number", ascending=False)
    meta = meta.drop_duplicates(subset=[item_col], keep="first").reset_index(drop=True)

    canon_set = set(canonical_map_df["canonical_parent_asin"].astype(str))
    meta = meta[meta[item_col].astype(str).isin(canon_set)].reset_index(drop=True)

    missing = canon_set - set(meta[item_col].astype(str))
    if missing:
        # Should be empty in practice; fail loudly if not so we don't silently
        # drop items that train/val/test reference.
        raise RuntimeError(
            f"{len(missing)} canonical parent_asins not found in new metadata "
            f"(first 5: {sorted(list(missing))[:5]})"
        )
    return meta


def _rebuild_item_features(category: str) -> dict:
    """Rebuild item_features.parquet from the (now-full) raw metadata,
    preserving canonical keys. Returns a small summary dict."""
    cat_proc = PROCESSED_DIR / category
    canonical_map = pd.read_parquet(cat_proc / "canonical_item_map.parquet")
    train = pd.read_parquet(cat_proc / "train.parquet")
    filtered = pd.read_parquet(cat_proc / "interactions_filtered.parquet")

    new_meta_path = RAW_DIR / category / "metadata.parquet"
    # Only load columns build_item_features actually consumes. For Books at 4.4M
    # rows, skipping `images`/`videos`/`details`/`subtitle`/`author`/`title`
    # keeps peak pandas memory bounded (those columns are the heaviest blobs).
    available = pq.read_schema(new_meta_path).names
    needed = [c for c in (
        "parent_asin", "main_category", "store", "price",
        "average_rating", "rating_number",
        "features", "description", "categories", "bought_together",
    ) if c in available]
    new_meta = pd.read_parquet(new_meta_path, columns=needed)

    # Snapshot the existing item_features so we can verify keys are unchanged.
    old_item_features = pd.read_parquet(cat_proc / "item_features.parquet")
    old_keys = set(old_item_features["parent_asin"].astype(str))

    canonical_meta = _restrict_to_canonical(new_meta, canonical_map)

    item_features = build_item_features(canonical_meta, train, filtered_df=filtered)
    new_keys = set(item_features["parent_asin"].astype(str))

    if new_keys != old_keys:
        only_old = old_keys - new_keys
        only_new = new_keys - old_keys
        raise RuntimeError(
            f"[{category}] item_features parent_asin set changed: "
            f"+{len(only_new)} / -{len(only_old)}"
        )

    out_path = cat_proc / "item_features.parquet"
    item_features.to_parquet(out_path, index=False)

    summary = {
        "category": category,
        "new_meta_path": str(new_meta_path),
        "new_meta_columns": list(new_meta.columns),
        "n_rows_new_meta": int(len(new_meta)),
        "n_rows_after_dedup_to_canonical": int(len(canonical_meta)),
        "n_rows_old_item_features": int(len(old_item_features)),
        "n_rows_new_item_features": int(len(item_features)),
        "keys_unchanged": True,
        "feature_populated": {
            col: int((item_features[col] > 0).sum())
            for col in ("n_features", "n_description", "n_categories",
                        "has_bought_together", "missing_flag")
        },
    }
    print(f"[rebuild] {category}: item_features rebuilt -> "
          f"n_features>{summary['feature_populated']['n_features']:,}  "
          f"n_description>{summary['feature_populated']['n_description']:,}  "
          f"n_categories>{summary['feature_populated']['n_categories']:,}",
          flush=True)
    return summary


def _list_or_array(x) -> bool:
    return isinstance(x, (list, np.ndarray))


def _column_summary(series: pd.Series) -> dict:
    """Per-column null rate and (for list columns) length distribution."""
    n = len(series)
    null_count = int(series.isna().sum())
    summary = {
        "dtype": str(series.dtype),
        "n": n,
        "null_count": null_count,
        "null_rate": (null_count / n) if n else 0.0,
    }
    if series.dtype == object and n > 0:
        head = series.dropna().head(50)
        if any(_list_or_array(v) for v in head):
            lens = series.apply(lambda v: len(v) if _list_or_array(v) else None)
            valid_lens = lens.dropna()
            empty = int((valid_lens == 0).sum())
            nonempty = int((valid_lens > 0).sum())
            summary.update({
                "kind": "list",
                "non_list_count": int(lens.isna().sum() - null_count),
                "empty_list_count": empty,
                "nonempty_list_count": nonempty,
                "nonempty_list_rate": (nonempty / n) if n else 0.0,
                "mean_len_when_present": float(valid_lens.mean()) if len(valid_lens) else 0.0,
                "max_len": int(valid_lens.max()) if len(valid_lens) else 0,
            })
            return summary
        summary["kind"] = "scalar_object"
    else:
        summary["kind"] = "scalar"
    return summary


def _schema_report(category: str) -> dict:
    """Build the metadata_schema_report for one category. Reads:
        - data/raw/{cat}/metadata.parquet
        - data/processed/{cat}/item_features.parquet
        - data/processed/{cat}/canonical_item_map.parquet
    """
    raw_path = RAW_DIR / category / "metadata.parquet"
    if_path = PROCESSED_DIR / category / "item_features.parquet"
    canon_path = PROCESSED_DIR / category / "canonical_item_map.parquet"

    item_features = pd.read_parquet(if_path)
    canon = pd.read_parquet(canon_path)

    # Read columns one-at-a-time so we don't materialize Books (4.4M x 16 cols
    # of long text) all at once.
    schema = pq.read_schema(raw_path)
    n_total = pq.read_metadata(raw_path).num_rows
    columns_summary = {}
    for col in schema.names:
        s = pd.read_parquet(raw_path, columns=[col])[col]
        columns_summary[col] = _column_summary(s)
        del s

    # Restricted-to-canonical view: report list-feature populated rates ONLY
    # over rows that actually feed item_features (post canonical dedup +
    # restriction). For the canonical view we need parent_asin + the list
    # columns + rating_number for the dedup tie-break; no need for title etc.
    canon_set = set(canon["canonical_parent_asin"].astype(str))
    list_cols_avail = [c for c in EXPECTED_LIST_COLUMNS if c in schema.names]
    canon_cols = ["parent_asin"] + list_cols_avail
    if "rating_number" in schema.names:
        canon_cols.append("rating_number")
    meta_canon_slim = pd.read_parquet(raw_path, columns=canon_cols)
    if "parent_asin" in meta_canon_slim.columns:
        meta_in_canon = meta_canon_slim[
            meta_canon_slim["parent_asin"].astype(str).isin(canon_set)
        ].copy()
        meta_in_canon = meta_in_canon.dropna(subset=["parent_asin"])
        if "rating_number" in meta_in_canon.columns:
            meta_in_canon = meta_in_canon.sort_values("rating_number", ascending=False)
        meta_in_canon = meta_in_canon.drop_duplicates(
            subset=["parent_asin"], keep="first",
        )
    else:
        meta_in_canon = meta_canon_slim.iloc[0:0]

    canonical_view = {col: _column_summary(meta_in_canon[col])
                      for col in EXPECTED_LIST_COLUMNS if col in meta_in_canon.columns}
    canonical_view_meta = {
        "n_canonical_rows": int(len(meta_in_canon)),
        "expected_n_canonical_rows": int(len(item_features)),
        "match": int(len(meta_in_canon)) == int(len(item_features)),
    }
    del meta_canon_slim, meta_in_canon

    feature_status = {}
    for col in ("n_features", "n_description", "n_categories",
                "has_bought_together", "missing_flag"):
        if col in item_features.columns:
            n_nz = int((item_features[col] > 0).sum())
            feature_status[col] = {
                "n_nonzero": n_nz,
                "n_total": int(len(item_features)),
                "nonzero_rate": (n_nz / len(item_features)) if len(item_features) else 0.0,
                "is_populated": bool(n_nz > 0),
            }

    return {
        "category": category,
        "raw_metadata_path": str(raw_path),
        "raw_metadata_n_rows": int(n_total),
        "raw_metadata_n_columns": int(len(schema.names)),
        "raw_metadata_columns": list(schema.names),
        "columns_summary": columns_summary,
        "canonical_view": {
            **canonical_view_meta,
            "list_columns_in_canonical_rows": canonical_view,
        },
        "item_features_summary": {
            "path": str(if_path),
            "n_rows": int(len(item_features)),
            "feature_status": feature_status,
        },
    }


def _write_schema_reports(categories: Iterable[str]) -> None:
    for cat in categories:
        report = _schema_report(cat)
        out = PROCESSED_DIR / cat / "metadata_schema_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        # Friendly summary
        fs = report["item_features_summary"]["feature_status"]
        flags = " ".join(
            f"{k.split('_',1)[1] if k.startswith('n_') else k}="
            f"{'OK' if fs[k]['is_populated'] else '0'}"
            for k in fs
        )
        print(f"[schema] {cat}: cols={report['raw_metadata_n_columns']}  "
              f"rows={report['raw_metadata_n_rows']:,}  {flags}  -> {out.name}",
              flush=True)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--categories", nargs="+", default=DEFAULT_REPULL_CATEGORIES,
                   choices=ALL_CATEGORIES,
                   help="Categories to re-pull + rebuild item_features for.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip re-pull + rebuild; only write metadata_schema_report.json "
                        "for all four categories using whatever is already on disk.")
    p.add_argument("--skip-repull", action="store_true",
                   help="Don't re-pull from HF; rebuild item_features from existing local "
                        "metadata.parquet (assumes it's already full).")
    p.add_argument("--no-backup", action="store_true",
                   help="Don't keep the 8-column metadata as .8col_backup")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    t0 = time.time()

    if args.report_only:
        print("[refresh] report-only mode; writing schema reports for all 4 categories",
              flush=True)
        _write_schema_reports(ALL_CATEGORIES)
        print(f"[refresh] done in {time.time()-t0:.1f}s", flush=True)
        return

    if not args.skip_repull:
        for cat in args.categories:
            _repull_one(cat, keep_backup=not args.no_backup)

    rebuild_summaries = []
    for cat in args.categories:
        rebuild_summaries.append(_rebuild_item_features(cat))

    _write_schema_reports(ALL_CATEGORIES)
    print(f"[refresh] all done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
