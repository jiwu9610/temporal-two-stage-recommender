"""Electronics rule-based feature ablation -- val ONLY (no test peek).

Diagnostics on Electronics showed `main_category` is the most diverse facet
(31 values, max share 27.8%) yet the per-pair train->val match rate is only
46.9%; users frequently buy from categories they have never bought from
before. Hypothesis: a deeper category path captures finer specialization
("Electronics | Computers | Laptops | Gaming Laptops") and might help.

This script tunes 3 rule-based variants on val and reports each one's
val-only Recall@K. We do not touch test in this run -- the protocol explicitly
asked for val-only ablation. If a variant clearly wins on val, the next step
(separate run) will do a single test evaluation with the chosen variant's
weights.

Variants:
    A.  store + main_category + popularity     (current main run setup)
    B.  main_category + popularity             (no store)
    C.  deeper_category + popularity           (deeper, no store, no main_category)

Output: results/phase1/electronics_ablation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download

from scripts.data.loader import load_metadata
from .evaluator import build_groundtruth, recall_precision_at_k
from .popularity import user_seen_from_train
from .rule_based import (
    build_user_facet_affinity,
    normalized_popularity_prior,
    recommend_weighted_facets,
)


import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase1"
HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY = "Electronics"
KS = (10, 50, 100)
K_RETRIEVE = 100
K_FOR_TUNING = 100
N_TUNING_USERS = 5000
SEED = 42


def _categories_path(value) -> str:
    """Join a metadata `categories` list (e.g. ['Electronics','Computers',...])
    into a single string; missing/empty -> 'Unknown'."""
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        return " | ".join(str(x) for x in value)
    return "Unknown"


def _ensure_categories_side_file(category: str) -> pd.DataFrame:
    """Return a DataFrame [parent_asin, categories] for `category`.

    Our local `metadata.parquet` for Video_Games / Books / Electronics was
    saved with a truncated 8-column schema (no `categories`). The HF parquet
    shards for the same category actually carry the full 16-column schema
    including `categories`. To avoid silently re-downloading the full 2 GB
    metadata each time, we cache just [parent_asin, categories] to a side
    file under data/raw/{category}/metadata_categories.parquet.
    """
    side_path = REPO_ROOT / "data" / "raw" / category / "metadata_categories.parquet"
    if side_path.exists():
        return pd.read_parquet(side_path)

    # Try local first -- All_Beauty does carry categories locally.
    local_meta = load_metadata(category)
    if "categories" in local_meta.columns:
        df = local_meta[["parent_asin", "categories"]].copy()
        side_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(side_path, index=False)
        return df

    print(f"[ablation] {category}: local meta has no 'categories'; "
          f"fetching from HF parquet shards (only the 2 needed columns)...", flush=True)
    api = HfApi()
    files = list(api.list_repo_tree(
        HF_REPO_ID, repo_type="dataset", path_in_repo=f"raw_meta_{category}"
    ))
    parquet_files = sorted(f.path for f in files if f.path.endswith(".parquet"))
    dfs = []
    for pf in parquet_files:
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=pf, repo_type="dataset")
        dfs.append(pd.read_parquet(local, columns=["parent_asin", "categories"]))
    df = pd.concat(dfs, ignore_index=True)
    side_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(side_path, index=False)
    print(f"[ablation] cached {len(df):,} rows -> {side_path}", flush=True)
    return df


def _add_deeper_category(item_features: pd.DataFrame, category: str) -> pd.DataFrame:
    """Augment item_features (in-memory) with a deeper-category column built
    from raw metadata's `categories` list, restricted to canonical parent_asins."""
    cats_df = _ensure_categories_side_file(category)
    cats_df = cats_df.dropna(subset=["parent_asin"])
    # The original parquet may have multiple rows per parent_asin (no dedup);
    # keep the row with the longest categories list (most informative).
    cats_df = cats_df.assign(_n=cats_df["categories"].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    ))
    cats_df = cats_df.sort_values("_n", ascending=False).drop_duplicates(
        subset=["parent_asin"], keep="first"
    )
    deeper_map = dict(
        zip(
            cats_df["parent_asin"].astype(str),
            cats_df["categories"].apply(_categories_path),
        )
    )
    out = item_features.copy()
    out["deeper_category"] = (
        out["parent_asin"].astype(str).map(deeper_map).fillna("Unknown")
    )
    return out


def _grid_2_param() -> List[Tuple[float, float]]:
    """Small grid over (w_facet, w_pop). 9 combos."""
    return [
        (ws, wp)
        for ws in (0.5, 1.0, 2.0)
        for wp in (0.0, 0.5, 1.0)
    ]


def _grid_3_param() -> List[Tuple[float, float, float]]:
    """Same shape as the main run for comparability with the 'A' variant."""
    return [
        (ws, wc, wp)
        for ws in (0.5, 1.0, 2.0)
        for wc in (0.5, 1.0, 2.0)
        for wp in (0.0, 0.5, 1.0)
    ]


def _tune_variant_on_val(
    variant_name: str,
    facet_weights_grid: Sequence[Dict[str, float]],
    user_ids: Sequence[str],
    candidate_items: np.ndarray,
    facet_arrays: Dict[str, np.ndarray],
    user_affinities: Dict[str, Dict[str, Dict[str, float]]],
    item_pop_prior: np.ndarray,
    user_seen,
    val_gt_for_tuning: Dict[str, set],
    pop_weights: Sequence[float],
    k: int = K_FOR_TUNING,
) -> Dict:
    """Tune one variant: enumerate (facet weights, pop_weight) on val tuning users."""
    rows: List[dict] = []
    best = (None, -1.0)
    for facet_weights in facet_weights_grid:
        for pop_w in pop_weights:
            topk = recommend_weighted_facets(
                user_ids=user_ids,
                candidate_items=candidate_items,
                facet_arrays=facet_arrays,
                user_affinities=user_affinities,
                facet_weights=facet_weights,
                item_pop_prior=item_pop_prior,
                pop_weight=pop_w,
                user_seen=user_seen,
                k=k,
            )
            recall, _, _ = recall_precision_at_k(topk, val_gt_for_tuning, k)
            row = {**facet_weights, "w_pop": pop_w, f"recall@{k}": recall}
            rows.append(row)
            if recall > best[1]:
                best = ({**facet_weights, "w_pop": pop_w}, recall)
    return {
        "variant": variant_name,
        "best_weights": best[0],
        f"best_val_recall@{k}": best[1],
        "grid": rows,
    }


def _full_val_recall(
    user_ids: Sequence[str],
    candidate_items: np.ndarray,
    facet_arrays: Dict[str, np.ndarray],
    user_affinities: Dict[str, Dict[str, Dict[str, float]]],
    item_pop_prior: np.ndarray,
    user_seen,
    val_gt: Dict[str, set],
    facet_weights: Dict[str, float],
    pop_weight: float,
    ks: Sequence[int] = KS,
) -> Dict[str, float]:
    """Evaluate one (facet_weights, pop_weight) at all reported Ks on FULL val."""
    topk = recommend_weighted_facets(
        user_ids=user_ids,
        candidate_items=candidate_items,
        facet_arrays=facet_arrays,
        user_affinities=user_affinities,
        facet_weights=facet_weights,
        item_pop_prior=item_pop_prior,
        pop_weight=pop_weight,
        user_seen=user_seen,
        k=max(ks),
    )
    out = {}
    for k in ks:
        r, p, _ = recall_precision_at_k(topk, val_gt, k)
        out[f"Recall@{k}"] = r
        out[f"Precision@{k}"] = p
    return out


def _full_recall(
    user_ids,
    candidate_items,
    facet_arrays,
    user_affinities,
    item_pop_prior,
    user_seen,
    gt,
    facet_weights,
    pop_weight,
    ks=KS,
):
    """Same as _full_val_recall but split-agnostic (used by test-eval)."""
    topk = recommend_weighted_facets(
        user_ids=user_ids,
        candidate_items=candidate_items,
        facet_arrays=facet_arrays,
        user_affinities=user_affinities,
        facet_weights=facet_weights,
        item_pop_prior=item_pop_prior,
        pop_weight=pop_weight,
        user_seen=user_seen,
        k=max(ks),
    )
    out = {}
    for k in ks:
        r, p, _ = recall_precision_at_k(topk, gt, k)
        out[f"Recall@{k}"] = r
        out[f"Precision@{k}"] = p
    return out


def _test_eval_only():
    """Load best weights from the existing val ablation JSON, run a single
    test evaluation for variants B and C, append results to the same JSON.

    Rule: only touch test once a variant clearly wins on val.
    Variant C beat A on val Recall@10 (~24% lift); B beat A across all K.
    Reporting B + C on test for completeness; A is unchanged from main run.
    """
    t0 = time.time()
    cat_dir = PROCESSED_DIR / CATEGORY
    json_path = RESULTS_DIR / "electronics_ablation.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} missing. Run `python -m scripts.retrieval.electronics_ablation` "
            f"(default mode) first to produce the val report and chosen weights."
        )
    report = json.loads(json_path.read_text())

    train = pd.read_parquet(cat_dir / "train.parquet")
    test = pd.read_parquet(cat_dir / "test.parquet")
    item_features = pd.read_parquet(cat_dir / "item_features.parquet")
    item_features = _add_deeper_category(item_features, CATEGORY)

    pool_df = item_features[item_features["in_train_catalog"] == 1].reset_index(drop=True)
    candidate_items = pool_df["parent_asin"].astype(str).to_numpy()
    facet_arrays = {
        "main_category": pool_df["main_category"].astype(str).to_numpy(),
        "deeper_category": pool_df["deeper_category"].astype(str).to_numpy(),
    }
    item_pop_prior = normalized_popularity_prior(train, candidate_items)
    print("[test-eval] building B + C affinities (no store needed) ...", flush=True)
    user_affinities = {
        "main_category": build_user_facet_affinity(train, item_features, "main_category"),
        "deeper_category": build_user_facet_affinity(train, item_features, "deeper_category"),
    }
    test_users = sorted(set(test["user_id"].astype(str)))
    user_seen = user_seen_from_train(train, users=test_users)
    test_gt = build_groundtruth(test)

    test_payload = {}
    for key in ("B_maincat_pop", "C_deepercat_pop"):
        bw = report["variants"][key]["best_weights"]
        weights = {k: v for k, v in bw.items() if k != "w_pop"}
        pop_w = bw["w_pop"]
        print(f"[test-eval] scoring {key} on test with weights {bw} ...", flush=True)
        m = _full_recall(
            user_ids=test_users,
            candidate_items=candidate_items,
            facet_arrays=facet_arrays,
            user_affinities=user_affinities,
            item_pop_prior=item_pop_prior,
            user_seen=user_seen,
            gt=test_gt,
            facet_weights=weights,
            pop_weight=pop_w,
        )
        test_payload[key] = {"best_weights_from_val": bw, "test_metrics": m}

    report.setdefault("test_eval", {}).update(test_payload)
    report["test_eval"]["note"] = (
        "Only B and C are evaluated on test, with weights chosen on val. "
        "Variant A's test number is the rule_based result in results/phase1/Electronics.json."
    )
    report["test_eval"]["elapsed_seconds"] = round(time.time() - t0, 2)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"[test-eval] updated {json_path}", flush=True)
    for key, payload in test_payload.items():
        m = payload["test_metrics"]
        print(f"  {key:<22} test  R@10={m['Recall@10']:.4f}  R@50={m['Recall@50']:.4f}  R@100={m['Recall@100']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", default="val-tune", choices=["val-tune", "test-only"],
        help="val-tune (default): tune all 3 variants on val and write the val report. "
             "test-only: read the existing val report and add a single test evaluation "
             "for variants B and C using their val-chosen weights.",
    )
    args = parser.parse_args()
    if args.mode == "test-only":
        _test_eval_only()
        return
    _val_tune()


def _val_tune():
    t0 = time.time()
    cat_dir = PROCESSED_DIR / CATEGORY
    train = pd.read_parquet(cat_dir / "train.parquet")
    val = pd.read_parquet(cat_dir / "val.parquet")
    item_features = pd.read_parquet(cat_dir / "item_features.parquet")

    # Add the new `deeper_category` column.
    item_features = _add_deeper_category(item_features, CATEGORY)

    # Candidate pool = in_train_catalog (same as main run).
    pool_df = item_features[item_features["in_train_catalog"] == 1].reset_index(drop=True)
    candidate_items = pool_df["parent_asin"].astype(str).to_numpy()
    print(f"[ablation] {CATEGORY} pool=in_train_catalog size={len(candidate_items)}", flush=True)

    facet_arrays = {
        "store": pool_df["store"].astype(str).to_numpy(),
        "main_category": pool_df["main_category"].astype(str).to_numpy(),
        "deeper_category": pool_df["deeper_category"].astype(str).to_numpy(),
    }
    item_pop_prior = normalized_popularity_prior(train, candidate_items)

    # Train-only affinities for each facet.
    print("[ablation] building affinities ...", flush=True)
    user_affinities = {
        "store": build_user_facet_affinity(train, item_features, "store"),
        "main_category": build_user_facet_affinity(train, item_features, "main_category"),
        "deeper_category": build_user_facet_affinity(train, item_features, "deeper_category"),
    }

    # User sets and tuning sample.
    eval_users = sorted(set(val["user_id"].astype(str)))
    user_seen = user_seen_from_train(train, users=eval_users)
    val_gt = build_groundtruth(val)
    rng = np.random.RandomState(SEED)
    val_pos_users = sorted([u for u, items in val_gt.items() if items])
    if len(val_pos_users) > N_TUNING_USERS:
        tuning_users = list(rng.choice(val_pos_users, size=N_TUNING_USERS, replace=False))
    else:
        tuning_users = val_pos_users
    val_gt_tune = {u: val_gt[u] for u in tuning_users if u in val_gt}
    print(f"[ablation] tuning on {len(tuning_users)} val-positive users ...", flush=True)

    # ---- Variant A: store + main_category + popularity (3-param grid) -------
    print("[ablation] variant A (store + main_cat + pop) ...", flush=True)
    A_grid_facets = [
        {"store": ws, "main_category": wc} for ws in (0.5, 1.0, 2.0) for wc in (0.5, 1.0, 2.0)
    ]
    A_tune = _tune_variant_on_val(
        "A",
        facet_weights_grid=A_grid_facets,
        user_ids=tuning_users,
        candidate_items=candidate_items,
        facet_arrays=facet_arrays,
        user_affinities=user_affinities,
        item_pop_prior=item_pop_prior,
        user_seen=user_seen,
        val_gt_for_tuning=val_gt_tune,
        pop_weights=(0.0, 0.5, 1.0),
    )

    # ---- Variant B: main_category + popularity (2-param grid) ---------------
    print("[ablation] variant B (main_cat + pop) ...", flush=True)
    B_grid_facets = [{"main_category": ws} for ws in (0.5, 1.0, 2.0)]
    B_tune = _tune_variant_on_val(
        "B",
        facet_weights_grid=B_grid_facets,
        user_ids=tuning_users,
        candidate_items=candidate_items,
        facet_arrays=facet_arrays,
        user_affinities=user_affinities,
        item_pop_prior=item_pop_prior,
        user_seen=user_seen,
        val_gt_for_tuning=val_gt_tune,
        pop_weights=(0.0, 0.5, 1.0),
    )

    # ---- Variant C: deeper_category + popularity (2-param grid) -------------
    print("[ablation] variant C (deeper_cat + pop) ...", flush=True)
    C_grid_facets = [{"deeper_category": ws} for ws in (0.5, 1.0, 2.0)]
    C_tune = _tune_variant_on_val(
        "C",
        facet_weights_grid=C_grid_facets,
        user_ids=tuning_users,
        candidate_items=candidate_items,
        facet_arrays=facet_arrays,
        user_affinities=user_affinities,
        item_pop_prior=item_pop_prior,
        user_seen=user_seen,
        val_gt_for_tuning=val_gt_tune,
        pop_weights=(0.0, 0.5, 1.0),
    )

    # ---- Full-val Recall@K for each variant's chosen weights ----------------
    print("[ablation] scoring full val with chosen weights for each variant ...", flush=True)
    def _full_for(tune):
        weights = {k: v for k, v in tune["best_weights"].items() if k != "w_pop"}
        pop_w = tune["best_weights"]["w_pop"]
        return _full_val_recall(
            user_ids=eval_users,
            candidate_items=candidate_items,
            facet_arrays=facet_arrays,
            user_affinities=user_affinities,
            item_pop_prior=item_pop_prior,
            user_seen=user_seen,
            val_gt=val_gt,
            facet_weights=weights,
            pop_weight=pop_w,
        )

    A_full = _full_for(A_tune)
    B_full = _full_for(B_tune)
    C_full = _full_for(C_tune)

    report = {
        "category": CATEGORY,
        "split": "val",
        "candidate_pool": {"type": "in_train_catalog", "size": int(len(candidate_items))},
        "n_tuning_users": len(tuning_users),
        "k_for_tuning": K_FOR_TUNING,
        "variants": {
            "A_store_maincat_pop": {
                "best_weights": A_tune["best_weights"],
                "best_val_recall@100_tuning_subset": A_tune[f"best_val_recall@{K_FOR_TUNING}"],
                "full_val_metrics": A_full,
            },
            "B_maincat_pop": {
                "best_weights": B_tune["best_weights"],
                "best_val_recall@100_tuning_subset": B_tune[f"best_val_recall@{K_FOR_TUNING}"],
                "full_val_metrics": B_full,
            },
            "C_deepercat_pop": {
                "best_weights": C_tune["best_weights"],
                "best_val_recall@100_tuning_subset": C_tune[f"best_val_recall@{K_FOR_TUNING}"],
                "full_val_metrics": C_full,
            },
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "electronics_ablation.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ablation] wrote {out}", flush=True)
    print()
    print(f"  variant                       best_weights                  Recall@10  Recall@50  Recall@100")
    for name, key in [("A store+maincat+pop", "A_store_maincat_pop"),
                      ("B maincat+pop", "B_maincat_pop"),
                      ("C deepercat+pop", "C_deepercat_pop")]:
        v = report["variants"][key]
        bw = v["best_weights"]
        full = v["full_val_metrics"]
        bw_str = " ".join(f"{k}={vv}" for k, vv in bw.items())
        print(f"  {name:<28} {bw_str:<38} "
              f"{full['Recall@10']:<10.4f} {full['Recall@50']:<10.4f} {full['Recall@100']:<10.4f}")


if __name__ == "__main__":
    main()
