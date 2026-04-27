"""Phase 2 candidate builder — produces the unified ranker input table.

Both the MLP and the Deep+Cross rankers consume a single parquet:

    data/processed/{category}/candidates.parquet

so they cannot diverge on what counts as a candidate or how features are
computed. Every per-user / per-item aggregate is **train-only**; labels come
from val.parquet / test.parquet held-out positives only. Train labels are
never written into this table.

Schema (one row per (user_id, parent_asin, split) candidate triple):

    user_id, parent_asin, split (val|test), label (0|1)
    source_popularity, source_rule, source_two_tower         (binary)
    popularity_score, rule_score, two_tower_score             (float)
    popularity_rank,  rule_rank,  two_tower_rank              (int, 0 = not in top-K, else 1..K)
    best_rank, num_sources
    user_features: n_reviews_train, avg_rating_train, std_rating_train,
                   n_unique_items_train, active_days_train, verified_rate_train
    item_features: main_category, store, price, average_rating, rating_number,
                   n_features, n_description, n_categories
    cross_features: user_store_affinity, user_category_affinity,
                    same_top_store, same_top_category

Retrieval sources:
  - popularity   : recomputed from train.parquet positives.
  - rule_based   : recomputed using best weights from results/phase1/{category}.json.
  - two_tower    : loaded from results/phase1/{category}_two_tower_{variant}_predictions.parquet
                   (written by train_two_tower.py at the end of training).

Implementation note: built with vectorized pandas merges (NOT a per-row loop)
because VG generates ~31M rows; the previous Python-list approach OOM'd on
the login node above ~10 min in.

CLI:
    python -m scripts.ranker.candidate_builder --category Video_Games
        [--top-k 100]
        [--two-tower-variant metadata_and_ids]
        [--out data/processed/{category}/candidates.parquet]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from scripts.retrieval.popularity import (
    compute_popularity,
    recommend_popularity,
    user_seen_from_train,
)
from scripts.retrieval.rule_based import (
    build_user_facet_affinity,
    normalized_popularity_prior,
    recommend_rule_based,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
PHASE1_RESULTS = REPO_ROOT / "results" / "phase1"
PHASE2_RESULTS = REPO_ROOT / "results" / "phase2"


# ---------------------------------------------------------------------------
# Retrieval source materialization (per source -> long DataFrame)
# ---------------------------------------------------------------------------

def _topk_dict_to_long_df(
    topk_items: Dict[str, List[str]],
    topk_scores: Dict[str, List[float]],
    source_name: str,
) -> pd.DataFrame:
    """Flatten {user: [item, ...]} + {user: [score, ...]} into a long DataFrame
    [user_id, parent_asin, {source}_rank, {source}_score]."""
    rows = []
    for u, items in topk_items.items():
        scores = topk_scores.get(u, [])
        for r, (it, sc) in enumerate(zip(items, scores)):
            rows.append((u, it, r + 1, float(sc)))
    df = pd.DataFrame(
        rows,
        columns=["user_id", "parent_asin", f"{source_name}_rank", f"{source_name}_score"],
    )
    return df


def _popularity_source(
    train_df: pd.DataFrame,
    candidate_pool: Set[str],
    eval_users: Sequence[str],
    user_seen: Dict[str, Set[str]],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pop = compute_popularity(train_df)
    pool_pop = {pa: float(pop.get(pa, 0)) for pa in candidate_pool}
    topk_items = recommend_popularity(
        popularity=pop, candidate_pool=candidate_pool,
        user_ids=eval_users, user_seen=user_seen, k=k,
    )
    topk_scores = {u: [pool_pop.get(it, 0.0) for it in items]
                   for u, items in topk_items.items()}
    return _topk_dict_to_long_df(topk_items, topk_scores, "popularity"), pool_pop


def _rule_source(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    candidate_pool: Set[str],
    eval_users: Sequence[str],
    user_seen: Dict[str, Set[str]],
    weights: Tuple[float, float, float],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    pool_df = item_features[item_features["parent_asin"].isin(candidate_pool)].reset_index(drop=True)
    candidate_items = pool_df["parent_asin"].astype(str).to_numpy()
    item_store_arr = pool_df["store"].astype(str).to_numpy()
    item_cat_arr = pool_df["main_category"].astype(str).to_numpy()
    item_pop_prior = normalized_popularity_prior(train_df, candidate_items)
    pop_prior_map = dict(zip(candidate_items, item_pop_prior.astype(float)))

    user_store_aff = build_user_facet_affinity(train_df, item_features, "store")
    user_cat_aff = build_user_facet_affinity(train_df, item_features, "main_category")

    topk_items = recommend_rule_based(
        user_ids=eval_users,
        candidate_items=candidate_items,
        item_store_arr=item_store_arr,
        item_cat_arr=item_cat_arr,
        item_pop_prior=item_pop_prior,
        user_store_aff=user_store_aff,
        user_cat_aff=user_cat_aff,
        user_seen=user_seen,
        weights=weights,
        k=k,
    )

    # Compute scores for top-K rows.
    item_meta_lookup = dict(zip(candidate_items, zip(item_store_arr, item_cat_arr)))
    w_store, w_cat, w_pop = weights
    topk_scores: Dict[str, List[float]] = {}
    for u, items in topk_items.items():
        s_aff = user_store_aff.get(u, {})
        c_aff = user_cat_aff.get(u, {})
        scores = []
        for it in items:
            store, cat = item_meta_lookup.get(it, ("Unknown", "Unknown"))
            scores.append(
                w_store * float(s_aff.get(store, 0.0))
                + w_cat * float(c_aff.get(cat, 0.0))
                + w_pop * float(pop_prior_map.get(it, 0.0))
            )
        topk_scores[u] = scores

    return (
        _topk_dict_to_long_df(topk_items, topk_scores, "rule"),
        pop_prior_map,
        user_store_aff,
        user_cat_aff,
    )


def _two_tower_source(category: str, variant: str, k: int) -> pd.DataFrame:
    pred_path = PHASE1_RESULTS / f"{category}_two_tower_{variant}_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"two-tower predictions not found at {pred_path}. "
            f"Run train_two_tower.py for {category}/{variant} first."
        )
    df = pd.read_parquet(pred_path)
    df = df.sort_values(["user_id", "rank"], kind="stable")
    df = df[df["rank"] <= k].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df = df.rename(columns={"rank": "two_tower_rank", "score": "two_tower_score"})
    return df[["user_id", "parent_asin", "two_tower_rank", "two_tower_score"]]


# ---------------------------------------------------------------------------
# Cross feature helpers
# ---------------------------------------------------------------------------

def _user_facet_top_value(train_df: pd.DataFrame, item_features: pd.DataFrame,
                          facet: str) -> pd.DataFrame:
    """For each user, the most-frequent value of `facet` over their train positives.
    Returns DataFrame [user_id, top_{facet}].
    """
    facet_map = item_features.set_index("parent_asin")[facet].astype(str).to_dict()
    pos = train_df[train_df["label"] == 1].copy()
    pos["_facet"] = pos["parent_asin"].astype(str).map(facet_map)
    pos = pos.dropna(subset=["_facet"])
    rows: List[Tuple[str, str]] = []
    for u, g in pos.groupby("user_id"):
        c = Counter(g["_facet"].tolist())
        if c:
            rows.append((u, c.most_common(1)[0][0]))
    return pd.DataFrame(rows, columns=["user_id", f"top_{facet}"])


def _affinity_to_long(aff: Dict[str, Dict[str, float]], facet: str) -> pd.DataFrame:
    """Affinity nested dict -> long [user_id, {facet}, user_{facet}_affinity]."""
    rows = [
        (u, fv, float(score))
        for u, m in aff.items()
        for fv, score in m.items()
    ]
    return pd.DataFrame(rows, columns=["user_id", facet, f"user_{facet}_affinity"])


# ---------------------------------------------------------------------------
# Main builder (vectorized)
# ---------------------------------------------------------------------------

def build_candidates(
    category: str,
    top_k: int = 100,
    two_tower_variant: str = "metadata_and_ids",
    out_path: Optional[Path] = None,
) -> Dict:
    t0 = time.time()
    cat_dir = PROCESSED_DIR / category
    train = pd.read_parquet(cat_dir / "train.parquet")
    val = pd.read_parquet(cat_dir / "val.parquet")
    test = pd.read_parquet(cat_dir / "test.parquet")
    item_features = pd.read_parquet(cat_dir / "item_features.parquet")
    user_features = pd.read_parquet(cat_dir / "user_features.parquet")

    for df in (train, val, test):
        df["user_id"] = df["user_id"].astype(str)
        df["parent_asin"] = df["parent_asin"].astype(str)
    item_features["parent_asin"] = item_features["parent_asin"].astype(str)
    user_features["user_id"] = user_features["user_id"].astype(str)

    candidate_pool = set(
        item_features.loc[item_features["in_train_catalog"] == 1, "parent_asin"]
    )
    print(f"[candidates] {category}: pool size = {len(candidate_pool):,}", flush=True)

    eval_users = sorted(set(val["user_id"]) | set(test["user_id"]))
    user_seen = user_seen_from_train(train, users=eval_users)
    print(f"[candidates] eval_users = {len(eval_users):,}", flush=True)

    # ---- 1. popularity ------------------------------------------------------
    print("[candidates] popularity ...", flush=True)
    pop_long, pop_global = _popularity_source(
        train, candidate_pool, eval_users, user_seen, top_k,
    )
    print(f"[candidates]   pop rows = {len(pop_long):,}", flush=True)

    # ---- 2. rule_based ------------------------------------------------------
    phase1_path = PHASE1_RESULTS / f"{category}.json"
    with open(phase1_path) as f:
        phase1 = json.load(f)
    bw = phase1["tuning"]["best_weights"]
    weights = (float(bw["w_store"]), float(bw["w_cat"]), float(bw["w_pop"]))
    print(f"[candidates] rule_based weights w_store={weights[0]} w_cat={weights[1]} "
          f"w_pop={weights[2]} (from {phase1_path.name})", flush=True)
    rule_long, pop_prior_map, store_aff, cat_aff = _rule_source(
        train, item_features, candidate_pool, eval_users, user_seen,
        weights, top_k,
    )
    print(f"[candidates]   rule rows = {len(rule_long):,}", flush=True)

    # ---- 3. two_tower -------------------------------------------------------
    print("[candidates] two_tower predictions ...", flush=True)
    tt_long = _two_tower_source(category, two_tower_variant, top_k)
    print(f"[candidates]   tt rows = {len(tt_long):,}", flush=True)

    # ---- 4. union: outer-join long-form sources on (user_id, parent_asin) --
    print("[candidates] union ...", flush=True)
    merged = pop_long.merge(rule_long, on=["user_id", "parent_asin"], how="outer")
    merged = merged.merge(tt_long, on=["user_id", "parent_asin"], how="outer")

    # Source flags from rank-presence.
    for s in ("popularity", "rule", "two_tower"):
        merged[f"source_{s}"] = merged[f"{s}_rank"].notna().astype(np.int8)
        merged[f"{s}_rank"] = merged[f"{s}_rank"].fillna(0).astype(np.int32)
    merged["num_sources"] = (
        merged["source_popularity"] + merged["source_rule"] + merged["source_two_tower"]
    ).astype(np.int8)

    # best_rank = min over present-source ranks (0s ignored).
    rank_arr = merged[["popularity_rank", "rule_rank", "two_tower_rank"]].to_numpy()
    rank_arr_masked = np.where(rank_arr > 0, rank_arr, np.iinfo(np.int32).max)
    best_rank = rank_arr_masked.min(axis=1)
    best_rank[best_rank == np.iinfo(np.int32).max] = 0
    merged["best_rank"] = best_rank.astype(np.int32)

    # Fill missing per-source scores. popularity_score: global per-item count;
    # rule_score: recompute from affinities + pop_prior; two_tower_score: a
    # below-min sentinel (model's training scores were higher than this).
    print("[candidates] fill missing per-source scores ...", flush=True)
    item_features_idx = item_features.set_index("parent_asin")[
        ["main_category", "store"]
    ].rename(columns={"main_category": "_main_cat", "store": "_store"})

    # Bring item store / main_category onto merged for rule recompute + cross.
    merged = merged.merge(item_features_idx, left_on="parent_asin",
                          right_index=True, how="left")
    merged["_main_cat"] = merged["_main_cat"].astype(str).fillna("Unknown")
    merged["_store"] = merged["_store"].astype(str).fillna("Unknown")

    # popularity_score fill: pop_global lookup
    pop_global_series = pd.Series(pop_global, name="_pop_global")
    pop_global_series.index.name = "parent_asin"
    merged = merged.merge(pop_global_series, left_on="parent_asin",
                          right_index=True, how="left")
    merged["popularity_score"] = merged["popularity_score"].fillna(merged["_pop_global"]).fillna(0.0)
    merged = merged.drop(columns=["_pop_global"])

    # Affinities -> long DFs for vectorized join on (user_id, store) /
    # (user_id, main_category).
    store_aff_df = _affinity_to_long(store_aff, "store").rename(
        columns={"user_store_affinity": "user_store_affinity"}
    )
    cat_aff_df = _affinity_to_long(cat_aff, "main_category").rename(
        columns={"user_main_category_affinity": "user_category_affinity"}
    )
    merged = merged.merge(
        store_aff_df.rename(columns={"store": "_store"}),
        on=["user_id", "_store"], how="left",
    )
    merged = merged.merge(
        cat_aff_df.rename(columns={"main_category": "_main_cat"}),
        on=["user_id", "_main_cat"], how="left",
    )
    merged["user_store_affinity"] = merged["user_store_affinity"].fillna(0.0).astype(np.float32)
    merged["user_category_affinity"] = merged["user_category_affinity"].fillna(0.0).astype(np.float32)

    # popularity prior per item -> for rule_score fill.
    pop_prior_series = pd.Series(pop_prior_map, name="_pop_prior")
    pop_prior_series.index.name = "parent_asin"
    merged = merged.merge(pop_prior_series, left_on="parent_asin",
                          right_index=True, how="left")
    merged["_pop_prior"] = merged["_pop_prior"].fillna(0.0)
    rule_fill = (
        weights[0] * merged["user_store_affinity"]
        + weights[1] * merged["user_category_affinity"]
        + weights[2] * merged["_pop_prior"]
    )
    merged["rule_score"] = merged["rule_score"].fillna(rule_fill).astype(np.float32)
    merged = merged.drop(columns=["_pop_prior"])

    # two_tower_score fill: below the observed minimum.
    tt_min = merged["two_tower_score"].min(skipna=True)
    tt_fill = float(tt_min) - 1.0 if pd.notna(tt_min) else 0.0
    merged["two_tower_score"] = merged["two_tower_score"].fillna(tt_fill).astype(np.float32)

    # ---- 5. user dense + item dense joins ----------------------------------
    print("[candidates] joins (user / item dense) ...", flush=True)
    user_dense_cols = [
        "n_reviews_train", "avg_rating_train", "std_rating_train",
        "n_unique_items_train", "active_days_train", "verified_rate_train",
    ]
    user_dense = user_features.set_index("user_id")[user_dense_cols]
    merged = merged.merge(user_dense, left_on="user_id", right_index=True, how="left")
    for c in user_dense_cols:
        merged[c] = merged[c].fillna(0.0).astype(np.float32)

    item_dense_cols = [
        "main_category", "store", "price", "average_rating", "rating_number",
        "n_features", "n_description", "n_categories",
    ]
    item_dense = item_features.set_index("parent_asin")[item_dense_cols]
    merged = merged.merge(item_dense, left_on="parent_asin",
                          right_index=True, how="left", suffixes=("", "_idense"))
    # The merge produces *_idense for any name collision; we don't have any
    # because the merged-side names are scoped (_main_cat / _store), but the
    # dense item columns are real-named main_category / store / price / etc.
    # Defensive cast / fill:
    merged["main_category"] = merged["main_category"].astype(str).fillna("Unknown")
    merged["store"] = merged["store"].astype(str).fillna("Unknown")
    for c in ("price", "average_rating", "rating_number",
              "n_features", "n_description", "n_categories"):
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # ---- 6. cross flags: same_top_store / same_top_category ----------------
    print("[candidates] cross flags ...", flush=True)
    top_store = _user_facet_top_value(train, item_features, "store")
    top_cat = _user_facet_top_value(train, item_features, "main_category")
    merged = merged.merge(top_store, on="user_id", how="left")
    merged = merged.merge(top_cat, on="user_id", how="left")
    merged["same_top_store"] = (
        merged["top_store"].astype(str) == merged["_store"]
    ).astype(np.int8)
    merged["same_top_category"] = (
        merged["top_main_category"].astype(str) == merged["_main_cat"]
    ).astype(np.int8)

    merged = merged.drop(columns=["_store", "_main_cat", "top_store", "top_main_category"])

    # ---- 7. duplicate per-split with split-specific labels -----------------
    print("[candidates] split + label assignment ...", flush=True)
    val_pos = val[val["label"] == 1][["user_id", "parent_asin"]].copy()
    val_pos["label_val"] = 1
    test_pos = test[test["label"] == 1][["user_id", "parent_asin"]].copy()
    test_pos["label_test"] = 1
    merged = merged.merge(val_pos, on=["user_id", "parent_asin"], how="left")
    merged = merged.merge(test_pos, on=["user_id", "parent_asin"], how="left")
    merged["label_val"] = merged["label_val"].fillna(0).astype(np.int8)
    merged["label_test"] = merged["label_test"].fillna(0).astype(np.int8)

    # Stack into split-rows.
    val_rows = merged.copy()
    val_rows["split"] = "val"
    val_rows["label"] = val_rows["label_val"]
    test_rows = merged.copy()
    test_rows["split"] = "test"
    test_rows["label"] = test_rows["label_test"]
    out_df = pd.concat([val_rows, test_rows], ignore_index=True)
    out_df = out_df.drop(columns=["label_val", "label_test"])

    # Final column ordering for stability.
    final_cols = [
        "user_id", "parent_asin", "split", "label",
        "source_popularity", "source_rule", "source_two_tower",
        "popularity_score", "rule_score", "two_tower_score",
        "popularity_rank", "rule_rank", "two_tower_rank",
        "best_rank", "num_sources",
        "n_reviews_train", "avg_rating_train", "std_rating_train",
        "n_unique_items_train", "active_days_train", "verified_rate_train",
        "main_category", "store", "price", "average_rating", "rating_number",
        "n_features", "n_description", "n_categories",
        "user_store_affinity", "user_category_affinity",
        "same_top_store", "same_top_category",
    ]
    out_df = out_df[final_cols]
    print(f"[candidates] final rows = {len(out_df):,}", flush=True)

    # ---- 8. persist + report ------------------------------------------------
    if out_path is None:
        out_path = cat_dir / "candidates.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[candidates] wrote {out_path}", flush=True)

    val_df_only = out_df[out_df["split"] == "val"]
    test_df_only = out_df[out_df["split"] == "test"]
    report = {
        "category": category,
        "built_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "top_k": top_k,
        "two_tower_variant": two_tower_variant,
        "rule_weights": {"w_store": weights[0], "w_cat": weights[1], "w_pop": weights[2]},
        "candidate_pool_size": len(candidate_pool),
        "n_eval_users": len(eval_users),
        "n_rows_total": int(len(out_df)),
        "n_rows_val": int(len(val_df_only)),
        "n_rows_test": int(len(test_df_only)),
        "positive_rate_val": float(val_df_only["label"].mean()) if len(val_df_only) else 0.0,
        "positive_rate_test": float(test_df_only["label"].mean()) if len(test_df_only) else 0.0,
        "per_user_candidates_mean": float(
            out_df.groupby(["user_id", "split"]).size().mean()
        ),
        "per_user_candidates_median": float(
            out_df.groupby(["user_id", "split"]).size().median()
        ),
        "source_overlap": {
            "n_only_pop": int(((val_df_only["source_popularity"] == 1) &
                               (val_df_only["source_rule"] == 0) &
                               (val_df_only["source_two_tower"] == 0)).sum()),
            "n_only_rule": int(((val_df_only["source_popularity"] == 0) &
                                (val_df_only["source_rule"] == 1) &
                                (val_df_only["source_two_tower"] == 0)).sum()),
            "n_only_tt": int(((val_df_only["source_popularity"] == 0) &
                              (val_df_only["source_rule"] == 0) &
                              (val_df_only["source_two_tower"] == 1)).sum()),
            "n_all_three": int(((val_df_only["source_popularity"] == 1) &
                                (val_df_only["source_rule"] == 1) &
                                (val_df_only["source_two_tower"] == 1)).sum()),
        },
    }
    PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)
    report_path = PHASE2_RESULTS / f"{category}_candidate_build_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[candidates] wrote report -> {report_path}", flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   choices=["All_Beauty", "Video_Games", "Books", "Electronics"])
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--two-tower-variant", default="metadata_and_ids")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out = Path(args.out) if args.out else None
    build_candidates(
        category=args.category,
        top_k=args.top_k,
        two_tower_variant=args.two_tower_variant,
        out_path=out,
    )


if __name__ == "__main__":
    main()
