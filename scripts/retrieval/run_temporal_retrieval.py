"""Temporal-snapshot retrieval CLI: random + popularity + rule-based baselines.

CLI:
    python -m scripts.retrieval.run_temporal_retrieval --category Video_Games

Consumes the artifacts written by scripts.data.temporal_pipeline under
data/processed/{category}/temporal/ and writes a JSON report to
results/phase1_temporal/{category}.json. The leave-last-two runner
(run_retrieval.py) is untouched; both protocols coexist.

Protocol differences vs run_retrieval.py -- each snapshot is served
INDEPENDENTLY, exactly as production would at that calendar instant:

    val snapshot   history < train_end   pool/popularity/affinity/seen from
                                         val history; rule weights TUNED here
    test snapshot  history < val_end     everything recomputed from test
                                         history (which includes the val
                                         window); rule weights REUSED from val

No single frozen prediction set is shared across snapshots. Ground truth is
each snapshot's first-positive-in-window labels: one positive item per user.
Every label user is evaluated -- cold users (no history) receive
popularity-fallback style recommendations and out-of-pool positives stay in
the denominator (they bound recall by the reported pool coverage).

Report adds per-snapshot warm/cold user breakdowns and cold-item statistics
on top of the usual Recall@K / Precision@K / coverage fields.
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
RESULTS_DIR = REPO_ROOT / "results" / "phase1_temporal"
DEFAULT_KS = (10, 50, 100)
DEFAULT_K_RETRIEVE = 100
DEFAULT_RANDOM_SEEDS = 5


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_snapshot(tdir: Path, name: str) -> Dict[str, pd.DataFrame]:
    return {
        "history": pd.read_parquet(tdir / f"{name}_history.parquet"),
        "labels": pd.read_parquet(tdir / f"{name}_labels.parquet"),
        "item_features": pd.read_parquet(tdir / f"{name}_item_features.parquet"),
    }


def _build_pool(item_features: pd.DataFrame, pool_type: str) -> set[str]:
    if pool_type == "eligible":
        flag = item_features["in_eligible_pool"] == 1
    elif pool_type == "history":
        flag = item_features["in_history_catalog"] == 1
    elif pool_type == "catalog":
        flag = pd.Series(True, index=item_features.index)
    else:
        raise ValueError(f"unknown candidate_pool type: {pool_type!r}")
    return set(item_features.loc[flag, "parent_asin"].astype(str))


def _subset_report(split_name, topk, gt, users, pool, pool_type, ks):
    """build_split_report over a user subset (warm-only / cold-only views)."""
    users = set(users)
    sub_topk = {u: v for u, v in topk.items() if u in users}
    sub_gt = {u: v for u, v in gt.items() if u in users}
    return build_split_report(split_name, sub_topk, sub_gt, pool, pool_type, ks)


def _score_snapshot(
    snap_name: str,
    snapshot: Dict[str, pd.DataFrame],
    pool_type: str,
    k_retrieve: int,
    ks: Sequence[int],
    seed: int,
    n_random_seeds: int,
    rule_weights: tuple | None,
    n_tuning_users: int,
) -> Dict[str, Any]:
    """Run all baselines on ONE snapshot from its own history. Returns the
    per-snapshot report payload (plus tuned weights when snap_name == 'val')."""
    history, labels, item_features = (
        snapshot["history"], snapshot["labels"], snapshot["item_features"]
    )

    pool = _build_pool(item_features, pool_type)
    # Alphabetical pool order so rule-based score TIES break on a future-free
    # key. The Phase 0 catalog arrives sorted by descending 2023 rating_number
    # (whole-timeline review counts); recommend_rule_based breaks ties by
    # array position, so inheriting that order would rank tied items -- the
    # entire pool, for cold users at w_pop=0 -- by post-cutoff popularity.
    pool_df = (
        item_features[item_features["parent_asin"].isin(pool)]
        .sort_values("parent_asin", kind="mergesort")
        .reset_index(drop=True)
    )
    candidate_items = pool_df["parent_asin"].astype(str).to_numpy()
    item_store_arr = pool_df["store"].astype(str).to_numpy()
    item_cat_arr = pool_df["main_category"].astype(str).to_numpy()
    item_pop_prior = normalized_popularity_prior(history, candidate_items)

    eval_users = sorted(labels["user_id"].astype(str).unique())
    user_seen = user_seen_from_train(history, users=eval_users)
    gt = build_groundtruth(labels)

    print(f"[{snap_name}] pool={len(pool)} eval_users={len(eval_users)}", flush=True)

    warm_users = set(labels.loc[labels["is_warm_user"] == 1, "user_id"].astype(str))
    cold_users = set(labels.loc[labels["is_warm_user"] == 0, "user_id"].astype(str))

    def _all_views(topk):
        overall = build_split_report(snap_name, topk, gt, pool, pool_type, ks)
        warm = _subset_report(f"{snap_name}_warm", topk, gt, warm_users,
                              pool, pool_type, ks)
        cold = _subset_report(f"{snap_name}_cold", topk, gt, cold_users,
                              pool, pool_type, ks)
        return overall, warm, cold

    models: Dict[str, Any] = {}

    # ---- random (multi-seed; each seed is evaluated immediately and its
    # predictions discarded -- holding 5 x n_users x K string lists at once
    # OOMs the 6GB login-node cgroup on full categories) ----------------------
    random_seeds = [seed + 100 * i for i in range(n_random_seeds)]
    seed_views: Dict[str, list] = {"overall": [], "warm": [], "cold": []}
    for s in random_seeds:
        topk = recommend_random(pool, eval_users, user_seen, k=k_retrieve, seed=s)
        overall, warm, cold = _all_views(topk)
        seed_views["overall"].append(overall.metrics)
        seed_views["warm"].append(warm.metrics)
        seed_views["cold"].append(cold.metrics)
        del topk
    random_summary: Dict[str, Any] = {"n_seeds": n_random_seeds, "seeds": random_seeds}
    for view_name, view_metrics in seed_views.items():
        agg = {}
        for key in view_metrics[0]:
            vals = [m[key] for m in view_metrics]
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        random_summary[view_name] = agg
    models["random"] = random_summary

    # ---- popularity (evaluate, then free predictions) -------------------------
    pop = compute_popularity(history)
    pop_topk = recommend_popularity(pop, pool, eval_users, user_seen, k=k_retrieve)
    overall_pop_report, warm_rep, cold_rep = _all_views(pop_topk)
    models["popularity"] = {
        "overall": overall_pop_report.metrics,
        "warm": warm_rep.metrics,
        "cold": cold_rep.metrics,
    }
    del pop_topk

    # ---- rule-based -------------------------------------------------------------
    user_store_aff = build_user_facet_affinity(history, item_features, "store")
    user_cat_aff = build_user_facet_affinity(history, item_features, "main_category")

    tuning_payload = None
    if rule_weights is None:
        # Tune on THIS snapshot (only ever the validation snapshot).
        rng = np.random.RandomState(seed)
        candidates_for_tuning = sorted(
            set(labels.loc[labels["label"] == 1, "user_id"].astype(str))
        )
        if len(candidates_for_tuning) > n_tuning_users:
            tuning_users = list(rng.choice(
                candidates_for_tuning, size=n_tuning_users, replace=False,
            ))
        else:
            tuning_users = candidates_for_tuning
        print(f"[{snap_name}] tuning rule weights on {len(tuning_users)} users...",
              flush=True)
        tune = tune_weights(
            val_df=labels,
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
        rule_weights = tune.best_weights
        tuning_payload = {
            "n_tuning_users": len(tuning_users),
            "k_for_tuning": max(ks),
            "best_weights": {
                "w_store": rule_weights[0],
                "w_cat": rule_weights[1],
                "w_pop": rule_weights[2],
            },
            "best_recall": tune.best_recall,
            "grid": tune.grid,
        }
        print(f"[{snap_name}] best weights={rule_weights} "
              f"recall@{max(ks)}={tune.best_recall:.4f}", flush=True)

    rule_topk = recommend_rule_based(
        user_ids=eval_users,
        candidate_items=candidate_items,
        item_store_arr=item_store_arr,
        item_cat_arr=item_cat_arr,
        item_pop_prior=item_pop_prior,
        user_store_aff=user_store_aff,
        user_cat_aff=user_cat_aff,
        user_seen=user_seen,
        weights=rule_weights,
        k=k_retrieve,
    )
    overall, warm_rep, cold_rep = _all_views(rule_topk)
    models["rule_based"] = {
        "overall": overall.metrics,
        "warm": warm_rep.metrics,
        "cold": cold_rep.metrics,
        "weights": {
            "w_store": rule_weights[0],
            "w_cat": rule_weights[1],
            "w_pop": rule_weights[2],
        },
    }
    del rule_topk

    n_labels = len(labels)
    payload = {
        "n_eval_users": n_labels,
        "n_warm_users": len(warm_users),
        "n_cold_users": len(cold_users),
        "cold_user_rate": (len(cold_users) / n_labels) if n_labels else 0.0,
        "cold_item_stats": {
            "n_label_items_not_in_history": int(
                (labels["item_in_history"] == 0).sum()
            ),
            "n_label_items_not_in_eligible_pool": int(
                (labels["item_in_eligible_pool"] == 0).sum()
            ),
            "label_item_in_history_rate": float(labels["item_in_history"].mean())
            if n_labels else 0.0,
            "label_item_in_eligible_pool_rate":
                float(labels["item_in_eligible_pool"].mean()) if n_labels else 0.0,
        },
        "candidate_pool": {"type": pool_type, "size": len(pool)},
        "heldout_positive_coverage_by_candidate_pool":
            overall_pop_report.heldout_positive_coverage_by_candidate_pool,
        "models": models,
    }
    return {"payload": payload, "rule_weights": rule_weights, "tuning": tuning_payload}


def run_category(
    category: str,
    candidate_pool_type: str = "eligible",
    k_retrieve: int = DEFAULT_K_RETRIEVE,
    ks: Sequence[int] = DEFAULT_KS,
    n_tuning_users: int = 5000,
    seed: int = 42,
    n_random_seeds: int = DEFAULT_RANDOM_SEEDS,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Run temporal-snapshot baselines for one category; write + return report."""
    t_start = time.time()
    tdir = Path(processed_dir) / category / "temporal"
    manifest_path = tdir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run "
            f"python -m scripts.data.temporal_pipeline --category {category} first"
        )
    manifest = json.loads(manifest_path.read_text())

    # Validation snapshot first: it owns weight tuning. Test reuses the weights
    # but recomputes every history-derived structure from its own snapshot.
    val_out = _score_snapshot(
        "val", _load_snapshot(tdir, "val"), candidate_pool_type,
        k_retrieve, ks, seed, n_random_seeds,
        rule_weights=None, n_tuning_users=n_tuning_users,
    )
    test_out = _score_snapshot(
        "test", _load_snapshot(tdir, "test"), candidate_pool_type,
        k_retrieve, ks, seed, n_random_seeds,
        rule_weights=val_out["rule_weights"], n_tuning_users=n_tuning_users,
    )

    report = {
        "category": category,
        "protocol": "global_temporal_snapshots",
        "started_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "cutoffs": manifest["cutoffs"],
        "k_retrieve": k_retrieve,
        "ks_reported": list(ks),
        "tuning": val_out["tuning"],
        "snapshots": {
            "val": val_out["payload"],
            "test": test_out["payload"],
        },
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{category}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[run] wrote {out} (elapsed {report['elapsed_seconds']}s)", flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   choices=["All_Beauty", "Video_Games", "Books", "Electronics"])
    p.add_argument("--candidate-pool", default="eligible",
                   choices=["eligible", "history", "catalog"])
    p.add_argument("--k-retrieve", type=int, default=DEFAULT_K_RETRIEVE)
    p.add_argument("--n-tuning-users", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seeds", type=int, default=DEFAULT_RANDOM_SEEDS)
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
