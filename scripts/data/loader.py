"""
Data loading utilities — streaming stats and sampled DataFrame loading.

Provides two approaches to accessing Amazon review data:
  1. **Streaming** (stream_stats, stream_user_ids) — reads the full JSONL line-by-line
     without loading it all into memory. Good for computing population-level statistics
     on large categories (Electronics: 44M reviews) where the full DataFrame won't fit.
  2. **Sampled loading** (load_sampled, load_metadata) — reads from cached Parquet
     (created by download.py) into a pandas DataFrame, with optional random sampling.

The streaming functions access the raw JSONL on HuggingFace (cached locally after
first download). The sampled loading functions prefer the local Parquet cache.

Usage:
    from scripts.data.loader import stream_stats, load_sampled

    # Streaming: O(unique_users + unique_items) memory, scans full JSONL
    stats = stream_stats("All_Beauty")

    # Sampled: loads into DataFrame, optionally subsampled
    df = load_sampled("All_Beauty", n=500_000)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _get_jsonl_path(category: str) -> str:
    """Download (or retrieve from HF cache) the raw JSONL review file for a category.

    HuggingFace Hub caches files locally after first download, so subsequent
    calls return the cached path without re-downloading.
    """
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"raw/review_categories/{category}.jsonl",
        repo_type="dataset",
    )


def stream_stats(
    category: str,
    save_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Stream through a category's reviews JSONL and collect statistics.

    Never materializes the full dataset in memory. Memory usage is
    O(unique_items + unique_users).

    Args:
        category: e.g. "All_Beauty", "Electronics"
        save_path: if set, save summary as JSON

    Returns:
        dict with keys:
            total_reviews, item_counts, user_counts, rating_hist,
            helpful_votes_hist, n_unique_items, n_unique_users
    """
    parquet_path = DEFAULT_DATA_DIR / "raw" / category / "reviews.parquet"

    item_counts: Counter = Counter()
    user_counts: Counter = Counter()
    rating_hist = [0] * 5
    helpful_bins = Counter()

    if parquet_path.exists():
        print(f"[stream_stats] Reading parquet for {category} -> {parquet_path}")
        df = pd.read_parquet(
            parquet_path, columns=["parent_asin", "user_id", "rating", "helpful_vote"]
        )
        total = len(df)

        item_counts = Counter(df["parent_asin"].value_counts().to_dict())
        user_counts = Counter(df["user_id"].value_counts().to_dict())

        ratings = df["rating"].dropna().astype(int).clip(1, 5) - 1
        for idx, cnt in ratings.value_counts().items():
            rating_hist[int(idx)] = int(cnt)

        hv = df["helpful_vote"].fillna(0).astype(int)
        helpful_bins["0"] = int((hv == 0).sum())
        helpful_bins["1-5"] = int(((hv >= 1) & (hv <= 5)).sum())
        helpful_bins["6-20"] = int(((hv >= 6) & (hv <= 20)).sum())
        helpful_bins["21-100"] = int(((hv >= 21) & (hv <= 100)).sum())
        helpful_bins["100+"] = int((hv > 100).sum())
    else:
        jsonl_path = _get_jsonl_path(category)
        total = 0
        with open(jsonl_path, "r") as f:
            for line in tqdm(f, desc=f"Streaming {category}"):
                row = json.loads(line)
                total += 1
                item_counts[row["parent_asin"]] += 1
                user_counts[row["user_id"]] += 1
                rating = row.get("rating")
                if rating is not None:
                    idx = max(0, min(4, int(rating) - 1))
                    rating_hist[idx] += 1
                hv_v = row.get("helpful_vote", 0) or 0
                if hv_v == 0:
                    helpful_bins["0"] += 1
                elif hv_v <= 5:
                    helpful_bins["1-5"] += 1
                elif hv_v <= 20:
                    helpful_bins["6-20"] += 1
                elif hv_v <= 100:
                    helpful_bins["21-100"] += 1
                else:
                    helpful_bins["100+"] += 1

    stats = {
        "category": category,
        "total_reviews": total,
        "n_unique_items": len(item_counts),
        "n_unique_users": len(user_counts),
        "rating_hist": rating_hist,                # [count_1star, ..., count_5star]
        "helpful_votes_hist": dict(helpful_bins),   # {"0": N, "1-5": N, ...}
        "item_counts": item_counts,                 # Counter: parent_asin -> count
        "user_counts": user_counts,                 # Counter: user_id -> count
    }

    # Optionally save a JSON summary (excluding large Counter objects)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip out the raw Counters (too large for JSON) and add percentile summaries
        summary = {k: v for k, v in stats.items() if k not in ("item_counts", "user_counts")}
        for name, counter in [("item", item_counts), ("user", user_counts)]:
            vals = np.array(list(counter.values()))
            summary[f"{name}_review_percentiles"] = {
                f"P{p}": float(np.percentile(vals, p))
                for p in [20, 50, 90, 99]
            }
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[stream_stats] Summary saved -> {save_path}")

    return stats


def stream_user_ids(category: str) -> set:
    """Stream through the full JSONL and collect the set of unique user_ids.

    Used by cross_category_users() in preprocessing.py to compute user overlap
    across categories without loading full DataFrames.

    Memory: O(unique_users) — only stores user_id strings, not full rows.
    """
    parquet_path = DEFAULT_DATA_DIR / "raw" / category / "reviews.parquet"
    if parquet_path.exists():
        print(f"[stream_user_ids] Reading parquet for {category}")
        return set(pd.read_parquet(parquet_path, columns=["user_id"])["user_id"].unique())
    jsonl_path = _get_jsonl_path(category)
    users = set()
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=f"User IDs {category}"):
            row = json.loads(line)
            users.add(row["user_id"])
    return users


def load_sampled(
    category: str,
    n: int | None = None,
    seed: int = 42,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load reviews as a pandas DataFrame, optionally sampled.

    Prefers reading from cached parquet (from download_category),
    otherwise reads from HuggingFace JSONL directly.

    Args:
        category: e.g. "All_Beauty"
        n: number of rows to sample (None = all)
        seed: random seed for sampling
        data_dir: root data directory
        columns: subset of columns to load (None = all)

    Returns:
        pandas DataFrame of review records
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / "raw" / category / "reviews.parquet"

    # Prefer the local Parquet cache (created by download.py) for speed;
    # fall back to parsing the raw JSONL from the HuggingFace cache
    if parquet_path.exists():
        print(f"[load_sampled] Reading from {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=columns)
    else:
        print(f"[load_sampled] Reading JSONL for {category} from HuggingFace cache...")
        jsonl_path = _get_jsonl_path(category)
        rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        if columns:
            df = df[columns]

    # Random subsample if requested (deterministic via seed)
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        print(f"[load_sampled] Sampled {n:,} rows from {category}")
    else:
        print(f"[load_sampled] Loaded {len(df):,} rows from {category}")

    return df


def load_metadata(
    category: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load item metadata (product catalog) as a pandas DataFrame.

    Metadata contains product-level info: parent_asin, title, average_rating,
    rating_number, price, main_category, store, features, description,
    categories, bought_together, etc.

    Prefers cached parquet (from download.py), otherwise downloads a single
    parquet shard directly from HuggingFace.
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / "raw" / category / "metadata.parquet"

    if parquet_path.exists():
        print(f"[load_metadata] Reading from {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=columns)
    else:
        print(f"[load_metadata] Downloading {category} metadata from HuggingFace...")
        pq_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"raw_meta_{category}/full-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(pq_path, columns=columns)

    print(f"[load_metadata] Loaded {len(df):,} items for {category}")
    return df
