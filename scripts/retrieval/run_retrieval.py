"""Phase 1 retrieval CLI: run popularity + rule-based on one category, write report.

CLI:
    python -m scripts.retrieval.run_retrieval --category Video_Games

Loads Phase 0 artifacts from data/processed/{category}/, runs the two
non-learned baselines, and writes a JSON report to results/phase1/{category}.json.

The report carries everything needed for downstream review: candidate pool type +
size + held-out coverage, n_total/n_positive eval users, Recall@K and
Precision@K at K in {10, 50, 100}, plus the rule-based weights chosen on val.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .evaluator import build_groundtruth, build_split_report
from .popularity import compute_popularity, recommend_popularity, user_seen_from_train
from .random_baseline import recommend_random
from .rule_based import (
    build_user_facet_affinity,
    normalized_popularity_prior,
    recommend_rule_based,
    tune_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase1"
DEFAULT_KS = (10, 50, 100)
DEFAULT_K_RETRIEVE = 100   # fixed since all reported Ks are <= 100
DEFAULT_RANDOM_SEEDS = 5   # multi-seed random so small-pool numbers come with std


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_phase0(cat_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(cat_dir / "train.parquet"),
        "val": pd.read_parquet(cat_dir / "val.parquet"),
        "test": pd.read_parquet(cat_dir / "test.parquet"),
        "item_features": pd.read_parquet(cat_dir / "item_features.parquet"),
    }


def _build_candidate_pool(item_features: pd.DataFrame, pool_type: str) -> set[str]:
    if pool_type == "in_train_catalog":
        flag = item_features["in_train_catalog"] == 1
    elif pool_type == "in_train_positive_catalog":
        flag = item_features["in_train_positive_catalog"] == 1
    elif pool_type == "in_filtered_universe":
        flag = item_features["in_filtered_universe"] == 1
    elif pool_type == "all":
        flag = pd.Series(True, index=item_features.index)
    else:
        raise ValueError(f"unknown candidate_pool type: {pool_type!r}")
    return set(item_features.loc[flag, "parent_asin"].astype(str))


def _eval_users(*splits: pd.DataFrame, user_col: str = "user_id") -> Sequence[str]:
    users = set()
    for s in splits:
        users.update(s[user_col].unique())
    return sorted(users)


def run_category(
    category: str,
    candidate_pool_type: str = "in_train_catalog",
    k_retrieve: int = DEFAULT_K_RETRIEVE,
    ks: Sequence[int] = DEFAULT_KS,
    n_tuning_users: int = 5000,
    seed: int = 42,
    n_random_seeds: int = DEFAULT_RANDOM_SEEDS,
) -> Dict[str, Any]:
    """Run Phase 1 baselines for one category. Returns + writes the report dict."""
    t_start = time.time()
    cat_dir = PROCESSED_DIR / category
    if not cat_dir.exists():
        raise FileNotFoundError(f"no Phase 0 outputs at {cat_dir}")
    data = _load_phase0(cat_dir)
    train, val, test, item_features = (
        data["train"], data["val"], data["test"], data["item_features"]
    )

    # Candidate pool, aligned numpy arrays for fast scoring.
    pool = _build_candidate_pool(item_features, candidate_pool_type)
    pool_df = item_features[item_features["parent_asin"].isin(pool)].reset_index(drop=True)
    candidate_items = pool_df["parent_asin"].astype(str).to_numpy()
    item_store_arr = pool_df["store"].astype(str).to_numpy()
    item_cat_arr = pool_df["main_category"].astype(str).to_numpy()
    item_pop_prior = normalized_popularity_prior(train, candidate_items)

    print(f"[run] {category} pool_type={candidate_pool_type} pool_size={len(pool)}", flush=True)

    # Eval users = union of users in val and test (we score every user once and
    # measure both splits against the same predictions; train-seen masking is
    # the same for both so this is correct).
    eval_users = _eval_users(val, test)
    user_seen = user_seen_from_train(train, users=eval_users)

    # ---- random baseline (multi-seed, sanity floor) --------------------------
    print(f"[run] random baseline x {n_random_seeds} seeds ...", flush=True)
    random_seeds = [seed + 100 * i for i in range(n_random_seeds)]
    random_topks: list[Dict[str, list[str]]] = [
        recommend_random(
            candidate_pool=pool,
            user_ids=eval_users,
            user_seen=user_seen,
            k=k_retrieve,
            seed=s,
        )
        for s in random_seeds
    ]

    # ---- popularity baseline -------------------------------------------------
    print("[run] popularity ...", flush=True)
    pop = compute_popularity(train)
    pop_topk = recommend_popularity(
        popularity=pop,
        candidate_pool=pool,
        user_ids=eval_users,
        user_seen=user_seen,
        k=k_retrieve,
    )

    # ---- rule-based: build train-only affinities, tune on val, score --------
    print("[run] rule-based affinities ...", flush=True)
    user_store_aff = build_user_facet_affinity(train, item_features, "store")
    user_cat_aff = build_user_facet_affinity(train, item_features, "main_category")

    rng = np.random.RandomState(seed)
    val_pos_users = sorted(set(val.loc[val["label"] == 1, "user_id"].astype(str)))
    if len(val_pos_users) > n_tuning_users:
        tuning_users = list(rng.choice(val_pos_users, size=n_tuning_users, replace=False))
    else:
        tuning_users = val_pos_users

    print(f"[run] rule-based tuning on {len(tuning_users)} val-positive users ...", flush=True)
    tune = tune_weights(
        val_df=val,
        user_ids_for_tuning=tuning_users,
        candidate_items=candidate_items,
        item_store_arr=item_store_arr,
        item_cat_arr=item_cat_arr,
        item_pop_prior=item_pop_prior,
        user_store_aff=user_store_aff,
        user_cat_aff=user_cat_aff,
        user_seen=user_seen,
        k_for_tuning=max(ks),
    )
    print(f"[run] rule-based best weights={tune.best_weights} "
          f"val_recall@{max(ks)}={tune.best_recall:.4f}", flush=True)

    print("[run] rule-based scoring all eval users with best weights ...", flush=True)
    rule_topk = recommend_rule_based(
        user_ids=eval_users,
        candidate_items=candidate_items,
        item_store_arr=item_store_arr,
        item_cat_arr=item_cat_arr,
        item_pop_prior=item_pop_prior,
        user_store_aff=user_store_aff,
        user_cat_aff=user_cat_aff,
        user_seen=user_seen,
        weights=tune.best_weights,
        k=k_retrieve,
    )

    # ---- evaluate all three models on val and test ---------------------------
    print("[run] evaluating ...", flush=True)
    splits_payload: Dict[str, dict] = {}
    for split_name, split_df in [("val", val), ("test", test)]:
        gt = build_groundtruth(split_df)
        # Multi-seed random: per-seed report -> mean / std per metric.
        per_seed_metrics = [
            build_split_report(split_name, t, gt, pool, candidate_pool_type, ks).metrics
            for t in random_topks
        ]
        random_summary: Dict[str, Any] = {
            "n_seeds": n_random_seeds,
            "seeds": random_seeds,
        }
        for metric_key in per_seed_metrics[0].keys():
            vals = [m[metric_key] for m in per_seed_metrics]
            random_summary[metric_key] = float(np.mean(vals))
            random_summary[f"{metric_key}_std"] = float(np.std(vals))
            random_summary[f"{metric_key}_per_seed"] = [float(v) for v in vals]

        pop_rep = build_split_report(
            split_name, pop_topk, gt, pool, candidate_pool_type, ks
        )
        rule_rep = build_split_report(
            split_name, rule_topk, gt, pool, candidate_pool_type, ks
        )
        splits_payload[split_name] = {
            "n_total_eval_users": pop_rep.n_total_eval_users,
            "n_positive_eval_users": pop_rep.n_positive_eval_users,
            "positive_eval_user_rate": pop_rep.positive_eval_user_rate,
            "heldout_positive_coverage_by_candidate_pool":
                pop_rep.heldout_positive_coverage_by_candidate_pool,
            "models": {
                "random": random_summary,
                "popularity": pop_rep.metrics,
                "rule_based": {
                    "best_weights": {
                        "w_store": tune.best_weights[0],
                        "w_cat": tune.best_weights[1],
                        "w_pop": tune.best_weights[2],
                    },
                    **rule_rep.metrics,
                },
            },
        }

    report = {
        "category": category,
        "started_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "candidate_pool": {
            "type": candidate_pool_type,
            "size": len(pool),
        },
        "k_retrieve": k_retrieve,
        "ks_reported": list(ks),
        "tuning": {
            "n_tuning_users": len(tuning_users),
            "k_for_tuning": max(ks),
            "best_weights": {
                "w_store": tune.best_weights[0],
                "w_cat": tune.best_weights[1],
                "w_pop": tune.best_weights[2],
            },
            "best_val_recall": tune.best_recall,
            "grid": tune.grid,
        },
        "splits": splits_payload,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{category}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[run] wrote {out} (elapsed {report['elapsed_seconds']}s)", flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   choices=["All_Beauty", "Video_Games", "Books", "Electronics"])
    p.add_argument("--candidate-pool", default="in_train_catalog",
                   choices=["in_train_catalog", "in_train_positive_catalog",
                            "in_filtered_universe", "all"])
    p.add_argument("--k-retrieve", type=int, default=DEFAULT_K_RETRIEVE)
    p.add_argument("--n-tuning-users", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seeds", type=int, default=DEFAULT_RANDOM_SEEDS,
                   help="Number of seeds for the random baseline; metrics reported as mean/std.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_category(
        category=args.category,
        candidate_pool_type=args.candidate_pool,
        k_retrieve=args.k_retrieve,
        n_tuning_users=args.n_tuning_users,
        seed=args.seed,
        n_random_seeds=args.random_seeds,
    )


if __name__ == "__main__":
    main()
