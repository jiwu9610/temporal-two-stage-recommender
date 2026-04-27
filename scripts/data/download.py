"""
Download Amazon Reviews 2023 datasets from HuggingFace Hub.

The Amazon Reviews 2023 dataset (McAuley-Lab) is hosted on HuggingFace but uses
a custom loading script incompatible with newer `datasets` versions. This module
bypasses that by downloading raw files directly via `huggingface_hub`:

  - Reviews: raw/review_categories/{Category}.jsonl  (one JSON object per line)
  - Metadata: raw_meta_{Category}/full-*.parquet  OR  raw/meta_categories/meta_{Category}.jsonl

Downloaded files are converted to Parquet and cached locally under data/raw/{Category}/
so subsequent loads are fast (see loader.py).

Usage:
    from scripts.data.download import download_category
    download_category("All_Beauty")                      # full download
    download_category("Electronics", max_rows=1_000_000) # cap large categories
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# HuggingFace dataset identifier — all categories live under this repo
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

# Default root for data storage: <project_root>/data/
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _download_metadata(category: str) -> pd.DataFrame:
    """Download item metadata for a category, handling multiple HF repo layouts.

    The HuggingFace repo stores metadata in different formats depending on
    when the category was uploaded. We try them in order of preference:
      1. Parquet shards in raw_meta_{category}/ — fastest, no parsing needed
      2. Single JSONL file in raw/meta_categories/ — slower, needs line-by-line parse

    Args:
        category: Amazon category name, e.g. "All_Beauty", "Electronics"

    Returns:
        DataFrame with one row per product (columns: parent_asin, title,
        average_rating, price, features, description, bought_together, etc.)
    """
    api = HfApi()

    # --- Strategy 1: Parquet shards (preferred) ---
    # List files in the raw_meta_{category}/ directory on HuggingFace
    try:
        files = list(api.list_repo_tree(
            REPO_ID, repo_type="dataset",
            path_in_repo=f"raw_meta_{category}",
        ))
        parquet_files = [f.path for f in files if f.path.endswith(".parquet")]
        if parquet_files:
            print(f"[download] Found {len(parquet_files)} parquet shard(s) for {category} metadata")
            dfs = []
            for pf in sorted(parquet_files):
                # hf_hub_download returns a local cache path
                local = hf_hub_download(repo_id=REPO_ID, filename=pf, repo_type="dataset")
                dfs.append(pd.read_parquet(local))
            return pd.concat(dfs, ignore_index=True)
    except Exception:
        pass  # Directory doesn't exist or access error — fall through

    # --- Strategy 2: JSONL fallback ---
    print(f"[download] Falling back to JSONL metadata for {category}")
    jsonl_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        repo_type="dataset",
    )
    rows = []
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="meta"):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def download_category(
    category: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_rows: int | None = None,
    overwrite: bool = False,
):
    """Download review and metadata files for one Amazon category, save as Parquet.

    This is the main entry point for data acquisition. It downloads two files:
      1. reviews.parquet  — user reviews (user_id, parent_asin, rating, text, timestamp, ...)
      2. metadata.parquet — item catalog  (parent_asin, title, price, features, bought_together, ...)

    Files are cached under data/raw/{category}/. If they already exist, the
    download is skipped (unless overwrite=True).

    For very large categories (Electronics ~44M, Books ~30M reviews), use max_rows
    to cap the download size and keep iteration fast during development.

    Args:
        category: Amazon category name, e.g. "All_Beauty", "Electronics", "Books"
        data_dir: root data directory (creates raw/{category}/ under it)
        max_rows: if set, only keep the first N review rows (useful for large categories)
        overwrite: re-download even if parquet files already exist on disk

    Returns:
        (reviews_path, meta_path): Paths to the saved Parquet files
    """
    data_dir = Path(data_dir)
    out_dir = data_dir / "raw" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = out_dir / "reviews.parquet"
    meta_path = out_dir / "metadata.parquet"

    # --- Reviews: download JSONL from HF, parse line-by-line, save as Parquet ---
    if not reviews_path.exists() or overwrite:
        print(f"[download] Downloading reviews JSONL for {category}...")
        jsonl_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"raw/review_categories/{category}.jsonl",
            repo_type="dataset",
        )
        print(f"[download] Parsing JSONL -> Parquet...")
        rows = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc="reviews")):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        df.to_parquet(reviews_path, index=False)
        print(f"[download] Saved {len(df):,} reviews -> {reviews_path}")
    else:
        print(f"[download] Reviews already exist: {reviews_path}")

    # --- Metadata: uses _download_metadata() which handles multiple HF layouts ---
    if not meta_path.exists() or overwrite:
        print(f"[download] Downloading metadata for {category}...")
        df_meta = _download_metadata(category)
        df_meta.to_parquet(meta_path, index=False)
        print(f"[download] Saved {len(df_meta):,} items -> {meta_path}")
    else:
        print(f"[download] Metadata already exists: {meta_path}")

    return reviews_path, meta_path
