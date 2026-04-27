"""Train/val-only feature diagnostics for the rule-based retrieval signal.

Motivated by the Phase 1 result that Electronics is the only category where
rule-based loses to popularity. The hypothesis is one of:

- main_category is dominated by a single value -> low entropy -> the per-user
  category histogram carries no discriminative signal.
- store has a long tail of one-off sellers -> per-user store affinity is noisy
  and rarely overlaps with the val/test stores.

This script measures both, on train + val only (no peeking at test), so the
we can decide whether to swap `main_category` for the deeper `categories`
hierarchy or to drop store entirely on a category-by-category basis.

CLI:
    python -m scripts.retrieval.diagnose_features --category Electronics

Writes results/phase1/diagnose_{category}.json.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase1"


def _shannon_entropy_bits(counts: pd.Series) -> float:
    """Shannon entropy in bits over a distribution of counts."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.to_numpy(dtype=np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _facet_distribution(item_features: pd.DataFrame, facet: str) -> Dict:
    """Top-20 value counts + tail summary for a categorical item facet."""
    s = item_features[facet].astype(str).fillna("Unknown")
    vc = s.value_counts()
    top20 = vc.head(20).to_dict()
    tail = vc.iloc[20:]
    return {
        "n_unique": int(vc.shape[0]),
        "max_share": float(vc.iloc[0] / vc.sum()) if len(vc) else 0.0,
        "entropy_bits": _shannon_entropy_bits(vc),
        "top_20": {str(k): int(v) for k, v in top20.items()},
        "tail_n_categories": int(tail.shape[0]),
        "tail_n_items": int(tail.sum()),
    }


def _user_facet_match_rate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    item_features: pd.DataFrame,
    facet: str,
) -> Dict:
    """Among val-positive (user, item) pairs, what fraction have the item's
    facet value already in the user's train-positive facet histogram?

    Reports both:
      - per_pair_match_rate     -- micro avg over (user, val_positive_item) rows
      - per_user_match_rate     -- macro avg, mean over users with >=1 val positive
    """
    train_pos = train_df[train_df["label"] == 1]
    val_pos = val_df[val_df["label"] == 1]
    item_facet = item_features.set_index("parent_asin")[facet].astype(str)

    train_user_facets: Dict[str, set] = {
        u: set(g) for u, g in train_pos.assign(
            facet_val=train_pos["parent_asin"].map(item_facet)
        ).dropna(subset=["facet_val"]).groupby("user_id")["facet_val"]
    }

    n_pair = 0
    n_pair_hit = 0
    per_user_rates = []
    for u, g in val_pos.groupby("user_id"):
        seen = train_user_facets.get(u, set())
        items = g["parent_asin"].tolist()
        hits = 0
        for it in items:
            facet_val = item_facet.get(it)
            if pd.isna(facet_val):
                continue
            n_pair += 1
            if facet_val in seen:
                hits += 1
                n_pair_hit += 1
        if items:
            per_user_rates.append(hits / len(items))

    return {
        "n_val_positive_pairs": n_pair,
        "per_pair_match_rate": (n_pair_hit / n_pair) if n_pair else 0.0,
        "per_user_match_rate_mean": float(np.mean(per_user_rates)) if per_user_rates else 0.0,
        "per_user_match_rate_median": float(np.median(per_user_rates)) if per_user_rates else 0.0,
        "n_users_with_val_positive": len(per_user_rates),
    }


def diagnose_category(category: str) -> dict:
    cat_dir = PROCESSED_DIR / category
    train = pd.read_parquet(cat_dir / "train.parquet")
    val = pd.read_parquet(cat_dir / "val.parquet")
    item_features = pd.read_parquet(cat_dir / "item_features.parquet")

    # Restrict the facet stats to items that are actually in the train catalog
    # -- diagnostics for the full canonical catalog would be misleading because
    # most items there are cold and never affect retrieval scoring.
    in_train = item_features[item_features["in_train_catalog"] == 1]

    return {
        "category": category,
        "n_items_in_train_catalog": int(len(in_train)),
        "n_train_positive_users": int(train.loc[train["label"] == 1, "user_id"].nunique()),
        "main_category": _facet_distribution(in_train, "main_category"),
        "store": _facet_distribution(in_train, "store"),
        "user_train_to_val_overlap": {
            "main_category": _user_facet_match_rate(train, val, item_features, "main_category"),
            "store": _user_facet_match_rate(train, val, item_features, "store"),
        },
    }


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   choices=["All_Beauty", "Video_Games", "Books", "Electronics"])
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out = diagnose_category(args.category)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"diagnose_{args.category}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[diagnose] wrote {path}")
    # Friendly stdout summary
    mc = out["main_category"]
    st = out["store"]
    ov = out["user_train_to_val_overlap"]
    print(f"\n  main_category : n_unique={mc['n_unique']:>5}  "
          f"max_share={mc['max_share']:.4f}  entropy={mc['entropy_bits']:.3f} bits")
    print(f"  store         : n_unique={st['n_unique']:>5}  "
          f"max_share={st['max_share']:.4f}  entropy={st['entropy_bits']:.3f} bits")
    print(f"\n  per-pair train->val overlap:")
    print(f"    main_category : {ov['main_category']['per_pair_match_rate']:.4f}")
    print(f"    store         : {ov['store']['per_pair_match_rate']:.4f}")


if __name__ == "__main__":
    main()
