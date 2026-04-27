"""Shared end-to-end runner for Phase 2 rankers.

Both train_mlp_ranker.py and train_complex_ranker.py call into this module so
the candidate dataset, feature spec, training loop, and evaluation contract
are identical -- only the model constructor differs.

Pipeline (spec, post the val-as-train fix):

    1. Load data/processed/{cat}/candidates.parquet.
    2. Split val users 80/20 -> ranker_train / ranker_val (deterministic seed).
    3. Build RankerFeatureSpec from ranker_train rows ONLY (no test, no
       ranker_val rows feed normalization or vocab).
    4. Build ranker_train / ranker_val / test tensors.
    5. Compute pos_weight = n_neg / n_pos on ranker_train labels.
    6. Train pointwise BCE-with-logits (with pos_weight); early-stop on
       ranker_val Recall@100.
    7. Compute retrieval-only baselines (rank by raw two_tower / popularity /
       rule scores) over ranker_val + test for context.
    8. Final eval on test.
    9. Write report JSON to results/phase2/{cat}_{name}.json.

Reporting contract:
    - candidate_pool_type     = "candidate_union_top{K}"   (per-user union)
    - heldout_positive_in_candidate_union_rate is computed per user and
      reported on every split: that is the Recall upper-bound the ranker
      could theoretically achieve.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.retrieval.evaluator import build_split_report

from scripts.ranker.ranker_features import (
    RankerFeatureSpec,
    build_feature_spec,
    build_tensors,
    evaluate_split,
    heldout_positive_coverage_in_candidate_union,
    split_val_users_for_ranker,
    train_pointwise_bce,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PHASE2_RESULTS = REPO_ROOT / "results" / "phase2"


def _retrieval_only_topk(df: pd.DataFrame, score_col: str, k: int = 100) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for u, g in df.groupby("user_id", sort=False):
        sorted_g = g.sort_values(score_col, ascending=False, kind="stable")
        out[u] = sorted_g["parent_asin"].astype(str).tolist()[:k]
    return out


def _gt_from_parquet(category: str, split: str) -> Dict[str, Set[str]]:
    """Canonical held-out positives for a split (NOT from the candidate
    table). Building gt from candidate rows would silently drop users whose
    held-out positive wasn't reached by any retriever, trivially making
    coverage == 1.0 and inflating Recall."""
    p = PROCESSED_DIR / category / f"{split}.parquet"
    df = pd.read_parquet(p)
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    pos = df[df["label"] == 1]
    return (
        pos.groupby("user_id")["parent_asin"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )


def run(
    category: str,
    ranker_name: str,
    build_model: Callable[[RankerFeatureSpec], nn.Module],
    *,
    epochs: int = 100,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    seed: int = 42,
    ranker_val_frac: float = 0.2,
    device: Optional[str] = None,
    extra_config: Optional[Dict] = None,
    candidates_path: Optional[Path] = None,
    candidate_pool_type: str = "candidate_union_top100",
) -> Dict:
    """End-to-end ranker run for one (category, model) combination."""
    t0 = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cat_dir = PROCESSED_DIR / category
    cands_path = candidates_path or (cat_dir / "candidates.parquet")
    if not cands_path.exists():
        raise FileNotFoundError(
            f"candidates parquet not found at {cands_path}. "
            f"Run scripts/ranker/candidate_builder.py first."
        )
    candidates = pd.read_parquet(cands_path)
    candidates["user_id"] = candidates["user_id"].astype(str)
    candidates["parent_asin"] = candidates["parent_asin"].astype(str)

    # ---- 1. internal user-wise val split ------------------------------------
    ranker_train_users, ranker_val_users = split_val_users_for_ranker(
        candidates, ranker_val_frac=ranker_val_frac, seed=seed,
    )
    val_mask = candidates["split"] == "val"
    test_mask = candidates["split"] == "test"
    train_df = candidates[val_mask & candidates["user_id"].isin(ranker_train_users)].reset_index(drop=True)
    rval_df = candidates[val_mask & candidates["user_id"].isin(ranker_val_users)].reset_index(drop=True)
    test_df = candidates[test_mask].reset_index(drop=True)
    print(f"[{ranker_name}] {category}: ranker_train_users={len(ranker_train_users):,}  "
          f"ranker_val_users={len(ranker_val_users):,}", flush=True)
    print(f"[{ranker_name}] rows  ranker_train={len(train_df):,}  "
          f"ranker_val={len(rval_df):,}  test={len(test_df):,}", flush=True)

    # ---- 2. feature spec from ranker_train only -----------------------------
    spec = build_feature_spec(train_df)
    print(f"[{ranker_name}] dense dim = {spec.n_dense}  cat vocab sizes = "
          f"{[(c, spec.cat_vocab_size(c)) for c in spec.cat_vocabs]}", flush=True)

    # ---- 3. tensors + pos_weight + groundtruth -----------------------------
    train_inputs = build_tensors(train_df, spec)
    rval_inputs = build_tensors(rval_df, spec)
    test_inputs = build_tensors(test_df, spec)

    n_pos_train = int(train_df["label"].sum())
    n_neg_train = int((train_df["label"] == 0).sum())
    pos_weight = (n_neg_train / max(1, n_pos_train)) if n_pos_train > 0 else 1.0
    print(f"[{ranker_name}] pos/neg train: {n_pos_train:,}/{n_neg_train:,}  "
          f"pos_weight={pos_weight:.2f}", flush=True)

    train_user_ids = train_df["user_id"].to_numpy()
    train_pa = train_df["parent_asin"].to_numpy()
    rval_user_ids = rval_df["user_id"].to_numpy()
    rval_pa = rval_df["parent_asin"].to_numpy()
    test_user_ids = test_df["user_id"].to_numpy()
    test_pa = test_df["parent_asin"].to_numpy()

    # Canonical held-out positives from val.parquet / test.parquet. NOT from
    # candidate rows: a user whose positive wasn't returned by any retriever
    # would be silently filtered (label==1 wouldn't appear), trivially
    # making coverage == 1.0 and inflating Recall numbers.
    val_gt_canonical = _gt_from_parquet(category, "val")
    test_gt_canonical = _gt_from_parquet(category, "test")
    train_gt = {u: g for u, g in val_gt_canonical.items() if u in ranker_train_users}
    rval_gt = {u: g for u, g in val_gt_canonical.items() if u in ranker_val_users}
    test_gt = test_gt_canonical

    # Per-split candidate-union pool (a flat union, used as the "pool"
    # argument to build_split_report). Recall computation is per-user via
    # _scores_to_topk; the pool here only affects coverage diagnostics.
    train_pool = set(train_df["parent_asin"].astype(str))
    rval_pool = set(rval_df["parent_asin"].astype(str))
    test_pool = set(test_df["parent_asin"].astype(str))

    # Per-user upper-bound coverage: does each user's held-out positive sit
    # in the user's candidate-union rows? This is the true Recall ceiling.
    cov_train = heldout_positive_coverage_in_candidate_union(train_df, train_gt)
    cov_rval = heldout_positive_coverage_in_candidate_union(rval_df, rval_gt)
    cov_test = heldout_positive_coverage_in_candidate_union(test_df, test_gt)
    print(f"[{ranker_name}] candidate-union held-out coverage "
          f"train={cov_train:.4f}  ranker_val={cov_rval:.4f}  test={cov_test:.4f}",
          flush=True)

    model = build_model(spec).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{ranker_name}] model params = {n_params:,}", flush=True)

    # ---- 4. train + early stop on ranker_val -------------------------------
    train_summary = train_pointwise_bce(
        model,
        train_inputs=train_inputs,
        val_inputs=rval_inputs,
        val_user_ids=rval_user_ids,
        val_pa=rval_pa,
        val_groundtruth=rval_gt,
        candidate_pool=rval_pool,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
        seed=seed,
        pos_weight=pos_weight,
    )

    # ---- 5. retrieval-only baselines (over ranker_val and test) ------------
    retrieval_only_reports: Dict = {}
    for src_col, name in [
        ("two_tower_score", "retrieval_only_two_tower"),
        ("popularity_score", "retrieval_only_popularity"),
        ("rule_score", "retrieval_only_rule_based"),
    ]:
        for split_name, df_split, gt, pool in (
            ("ranker_val", rval_df, rval_gt, rval_pool),
            ("test", test_df, test_gt, test_pool),
        ):
            topk = _retrieval_only_topk(df_split, src_col, k=100)
            rep = build_split_report(
                split_name, topk, gt, pool, candidate_pool_type, ks=(10, 50, 100),
            )
            retrieval_only_reports.setdefault(name, {})[split_name] = rep.to_dict()

    # ---- 6. final eval (ranker on ranker_val and on test) -------------------
    rval_rep, _ = evaluate_split(
        model, rval_inputs, rval_user_ids, rval_pa, rval_gt,
        rval_pool, "ranker_val", candidate_pool_type=candidate_pool_type, device=device,
    )
    test_rep, _ = evaluate_split(
        model, test_inputs, test_user_ids, test_pa, test_gt,
        test_pool, "test", candidate_pool_type=candidate_pool_type, device=device,
    )

    elapsed = time.time() - t0
    report = {
        "category": category,
        "ranker": ranker_name,
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "device": device,
        "n_params": n_params,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            "ranker_val_frac": ranker_val_frac,
            "model": extra_config or {},
        },
        "candidate_pool_type": candidate_pool_type,
        "splits_setup": {
            "n_ranker_train_users": len(ranker_train_users),
            "n_ranker_val_users": len(ranker_val_users),
            "n_ranker_train_rows": int(len(train_df)),
            "n_ranker_val_rows": int(len(rval_df)),
            "n_test_rows": int(len(test_df)),
        },
        "class_imbalance": {
            "n_pos_ranker_train": n_pos_train,
            "n_neg_ranker_train": n_neg_train,
            "positive_rate_ranker_train": (n_pos_train / max(1, n_pos_train + n_neg_train)),
            "pos_weight": pos_weight,
            "positive_rate_ranker_val": float(rval_df["label"].mean()) if len(rval_df) else 0.0,
            "positive_rate_test": float(test_df["label"].mean()) if len(test_df) else 0.0,
        },
        "heldout_positive_in_candidate_union_rate": {
            "ranker_train": cov_train,
            "ranker_val": cov_rval,
            "test": cov_test,
        },
        "feature_spec": {
            "n_dense": spec.n_dense,
            "cat_vocab_sizes": {c: spec.cat_vocab_size(c) for c in spec.cat_vocabs},
        },
        "history": train_summary["history"],
        "best_ranker_val_recall@100": train_summary["best_ranker_val_recall@100"],
        "retrieval_only_baselines": retrieval_only_reports,
        "splits": {
            "ranker_val": rval_rep.to_dict(),
            "test": test_rep.to_dict(),
        },
    }

    PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)
    out = PHASE2_RESULTS / f"{category}_{ranker_name}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{ranker_name}] {category}: wrote {out}  (elapsed {elapsed:.1f}s)", flush=True)
    return report
