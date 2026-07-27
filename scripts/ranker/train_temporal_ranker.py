"""Clean temporal ranker: time-separated training, model selection, refit, test.

CLI:
    python -m scripts.ranker.train_temporal_ranker --category All_Beauty [--smoke]

Protocol (walk-forward, no within-snapshot user split):

    TRAIN       candidates_ranker_train.parquet    labels from [T0, T1)
    SELECTION   candidates_model_selection.parquet labels from [T1, T2)
                -- early stopping, architecture comparison (MLP vs Deep&Cross),
                   learning-rate selection, epoch selection happen HERE ONLY.
    REFIT       chosen config retrained on ranker_train + model_selection rows
                for the exact epoch count chosen during selection.
                NO early stopping, NO metric peeking at test.
    FINAL TEST  the fixed refit model applied once to candidates_test.parquet,
                scored against groundtruth_test ([T2, T3) labels).

Test labels influence nothing upstream: not fitting, not early stopping, not
architecture/lr/epoch choice, not normalization or vocabularies (refit stats
come from train+selection rows only).

Reported on final test: overall Recall/Precision@10/50/100 over ALL ground
truth users (missed users stay in the denominator), conditional Recall@K
given the target was retrieved, the candidate ceiling (retrieved coverage),
0/1/2/3+ history cohorts, and warm/cold target-item cohorts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.retrieval.evaluator import build_split_report, recall_precision_at_k
from scripts.ranker.mlp_ranker import MLPRanker, MLPRankerConfig
from scripts.ranker.complex_ranker import DeepCrossConfig, DeepCrossRanker
from scripts.ranker.ranker_features import (
    RankerFeatureSpec,
    _build_cat_vocab,
    _scores_to_topk,
    train_pointwise_bce,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase2_temporal"
KS = (10, 50, 100)

# Dense features available in candidates_{snapshot}.parquet. Snapshot-scoped
# `_hist` aggregates + the three temporal features; NO price / average_rating /
# rating_number (dump-time whole-timeline statistics).
TEMPORAL_DENSE_FEATURES = (
    "popularity_score", "rule_score", "two_tower_score",
    "popularity_rank", "rule_rank", "two_tower_rank",
    "best_rank", "num_sources",
    "source_popularity", "source_rule", "source_two_tower",
    "user_n_reviews_hist", "user_avg_rating_hist", "user_std_rating_hist",
    "user_n_unique_items_hist", "user_active_days_hist",
    "user_verified_rate_hist",
    "days_since_last_interaction", "interactions_last_30d",
    "positives_last_30d", "n_history_events", "user_has_history",
    "item_n_reviews_hist", "item_n_positives_hist", "item_avg_rating_hist",
    "item_n_unique_reviewers_hist",
    "n_features", "n_description", "n_categories",
    "user_store_affinity", "user_category_affinity",
    "same_top_store", "same_top_category",
)
TEMPORAL_LOG1P = {
    "popularity_score", "rule_score",
    "popularity_rank", "rule_rank", "two_tower_rank", "best_rank",
    "user_n_reviews_hist", "user_n_unique_items_hist", "user_active_days_hist",
    "days_since_last_interaction", "interactions_last_30d", "positives_last_30d",
    "n_history_events",
    "item_n_reviews_hist", "item_n_positives_hist",
    "item_n_unique_reviewers_hist",
    "n_features", "n_description", "n_categories",
}
TEMPORAL_CATEGORICALS = ("main_category", "store")


def resolve_feature_list(df: pd.DataFrame):
    """Dense feature list for a candidates table: the frozen base tuple plus
    any Phase 3A extra-source columns (source_*/\*_rank/\*_score triplets),
    sorted for a stable order. Extra ranks/scores get log1p (non-negative,
    long-tailed)."""
    base = [f for f in TEMPORAL_DENSE_FEATURES if f in df.columns]
    known = set(TEMPORAL_DENSE_FEATURES) | {
        "snapshot", "user_id", "parent_asin", "label",
        *TEMPORAL_CATEGORICALS,
    }
    extras = sorted(
        c for c in df.columns
        if c not in known and (
            c.startswith("source_") or c.endswith("_rank")
            or c.endswith("_score"))
    )
    features = tuple(base + extras)
    log1p = set(TEMPORAL_LOG1P) | {
        c for c in extras if c.endswith("_rank") or c.endswith("_score")
    }
    return features, log1p


def build_temporal_feature_spec(train_df: pd.DataFrame,
                                features=None, log1p=None) -> RankerFeatureSpec:
    """Norm stats + categorical vocabs from the given TRAINING rows only."""
    if features is None:
        features, log1p = TEMPORAL_DENSE_FEATURES, TEMPORAL_LOG1P
    dense = np.zeros((len(train_df), len(features)), dtype=np.float64)
    for j, col in enumerate(features):
        x = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        if col in log1p:
            x = np.log1p(np.maximum(x, 0.0))
        dense[:, j] = x
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    cat_vocabs = {c: _build_cat_vocab(train_df[c]) for c in TEMPORAL_CATEGORICALS}
    return RankerFeatureSpec(
        n_dense=len(features),
        dense_mean=mean, dense_std=std, cat_vocabs=cat_vocabs,
    )


def build_temporal_tensors(df: pd.DataFrame, spec: RankerFeatureSpec,
                           features=None, log1p=None) -> Dict[str, torch.Tensor]:
    if features is None:
        features, log1p = TEMPORAL_DENSE_FEATURES, TEMPORAL_LOG1P
    dense = np.zeros((len(df), len(features)), dtype=np.float32)
    for j, col in enumerate(features):
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        if col in log1p:
            x = np.log1p(np.maximum(x, 0.0))
        dense[:, j] = ((x - spec.dense_mean[j]) / spec.dense_std[j]).astype(np.float32)
    out = {
        "dense": torch.from_numpy(dense),
        "label": torch.from_numpy(df["label"].to_numpy(np.float32)),
    }
    for c, vocab in spec.cat_vocabs.items():
        out[f"cat__{c}"] = torch.from_numpy(
            df[c].astype(str).map(vocab).fillna(0).to_numpy(np.int64)
        )
    return out


def _make_model(arch: str, spec: RankerFeatureSpec) -> nn.Module:
    if arch == "mlp":
        return MLPRanker(spec, MLPRankerConfig())
    if arch == "deep_cross":
        return DeepCrossRanker(spec, DeepCrossConfig())
    raise ValueError(f"unknown arch {arch!r}")


def _score_df(model: nn.Module, inputs: Dict[str, torch.Tensor],
              device: str, batch_size: int = 65536) -> np.ndarray:
    model.eval()
    feats = {k: v for k, v in inputs.items() if k != "label"}
    n = next(iter(feats.values())).shape[0]
    chunks = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            sub = {k: v[s:e].to(device) for k, v in feats.items()}
            chunks.append(model(**sub).cpu())
    return torch.cat(chunks).numpy() if chunks else np.empty(0)


def train_fixed_epochs(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    n_epochs: int,
    *,
    batch_size: int, lr: float, weight_decay: float,
    grad_clip: float = 5.0, device: str = "cpu", seed: int = 42,
    pos_weight: Optional[float] = None,
) -> List[Dict]:
    """Deterministic-schedule training: exactly n_epochs, NO evaluation, NO
    early stopping, NO best-state selection. Used for the final refit so test
    labels cannot influence when training stops."""
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    dev = {k: v.to(device) for k, v in inputs.items()}
    labels = dev["label"]
    n = labels.numel()
    pw = torch.tensor(float(pos_weight), device=device) if pos_weight else None
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    log = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=rng).to(device)
        sum_loss, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:min(start + batch_size, n)]
            feats = {k: v.index_select(0, idx) for k, v in dev.items() if k != "label"}
            y = labels.index_select(0, idx)
            logits = model(**feats)
            loss = bce(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = y.numel()
            seen += bs
            sum_loss += float(loss.detach()) * bs
        log.append({"epoch": epoch, "train_loss": sum_loss / max(1, seen)})
        print(f"  refit ep{epoch:02d}/{n_epochs} loss={sum_loss/max(1,seen):.4f}",
              flush=True)
    return log


def _cohort_of(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"


def _eval_test(
    model: nn.Module,
    test_df: pd.DataFrame,
    test_inputs: Dict[str, torch.Tensor],
    gt_df: pd.DataFrame,
    device: str,
) -> Dict:
    """Final test evaluation with all spec-required breakdowns."""
    logits = _score_df(model, test_inputs, device)
    topk = _scores_to_topk(
        logits, test_df["user_id"].to_numpy(), test_df["parent_asin"].to_numpy(), k=100,
    )
    gt = {u: {pa} for u, pa in zip(gt_df["user_id"].astype(str),
                                   gt_df["parent_asin"].astype(str))}
    pool = set(test_df["parent_asin"].astype(str))
    overall = build_split_report("test", topk, gt, pool,
                                 "candidate_union_top100", KS)

    cand_per_user: Dict[str, Set[str]] = {
        u: set(g) for u, g in
        test_df.groupby("user_id")["parent_asin"].agg(set).items()
    }
    retrieved_users = {u for u, g in gt.items()
                       if next(iter(g)) in cand_per_user.get(u, set())}
    ceiling = len(retrieved_users) / len(gt) if gt else 0.0
    gt_retrieved = {u: g for u, g in gt.items() if u in retrieved_users}
    conditional = {}
    for k in KS:
        r, p, n = recall_precision_at_k(topk, gt_retrieved, k)
        conditional[f"Recall@{k}"] = r
        conditional[f"Precision@{k}"] = p

    n_hist = dict(zip(gt_df["user_id"].astype(str),
                      gt_df["n_history_events"].astype(int)))
    item_in_hist = dict(zip(gt_df["user_id"].astype(str),
                            gt_df["item_in_history"].astype(int)))
    cohorts = {}
    for name in ("0", "1", "2", "3+"):
        users = {u for u in gt if _cohort_of(n_hist.get(u, 0)) == name}
        sub_gt = {u: g for u, g in gt.items() if u in users}
        rep = {}
        for k in KS:
            r, p, n = recall_precision_at_k(topk, sub_gt, k)
            rep[f"Recall@{k}"] = r
            rep[f"Precision@{k}"] = p
        cohorts[name] = {"n_users": len(users), **rep}
    item_cohorts = {}
    for name, flag in (("warm_target_item", 1), ("cold_target_item", 0)):
        users = {u for u in gt if item_in_hist.get(u, 0) == flag}
        sub_gt = {u: g for u, g in gt.items() if u in users}
        rep = {}
        for k in KS:
            r, p, n = recall_precision_at_k(topk, sub_gt, k)
            rep[f"Recall@{k}"] = r
        item_cohorts[name] = {"n_users": len(users), **rep}

    return {
        "n_groundtruth_users": len(gt),
        "overall": overall.metrics,
        "candidate_ceiling_retrieved_coverage": ceiling,
        "conditional_given_retrieved": {
            "n_users_retrieved": len(retrieved_users), **conditional,
        },
        "history_cohorts": cohorts,
        "target_item_cohorts": item_cohorts,
    }


def run(
    category: str,
    *,
    max_epochs: int = 20,
    batch_size: int = 8192,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 3,
    seed: int = 42,
    device: Optional[str] = None,
    smoke: bool = False,
    max_users_per_snapshot: int = 0,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    variant: Optional[str] = None,
    stage_b_only: bool = False,
) -> Dict:
    """Full run: selection -> refit -> single locked test eval.

    Phase 3A additions: `variant` reads candidates from
    temporal_ranker/variants/{variant}/ (ground truth still comes from the
    base dir -- it is split-defined, not config-defined). `stage_b_only`
    stops after model selection and reports selection-snapshot metrics ONLY;
    the test snapshot is neither read nor required to exist.
    """
    t0 = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tdir = Path(processed_dir) / category / "temporal_ranker"
    cdir = (tdir / "variants" / variant) if variant else tdir

    # Memory-bounded runs (login-node smoke): deterministic per-snapshot user
    # subsample, applied AT READ TIME via a parquet filter -- loading the full
    # candidate table first and subsetting after peaks at several GB and gets
    # OOM-killed inside the 6GB login-node cgroup. Ground truth is subset to
    # the SAME users so denominators stay honest (scale cap, not a filter).
    cands = {}
    kept_users: Dict[str, set] = {}
    rng = np.random.default_rng(seed)
    snaps_needed = (("ranker_train", "model_selection") if stage_b_only
                    else ("ranker_train", "model_selection", "test"))
    for snap in snaps_needed:
        path = cdir / f"candidates_{snap}.parquet"
        if max_users_per_snapshot:
            users = np.array(sorted(pd.read_parquet(
                path, columns=["user_id"])["user_id"].astype(str).unique()))
            if len(users) > max_users_per_snapshot:
                keep = sorted(rng.choice(users, size=max_users_per_snapshot,
                                         replace=False))
                kept_users[snap] = set(keep)
                df = pd.read_parquet(path, filters=[("user_id", "in", keep)])
            else:
                df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)
        df["user_id"] = df["user_id"].astype(str)
        df["parent_asin"] = df["parent_asin"].astype(str)
        cands[snap] = df.reset_index(drop=True)
    gt_sel = pd.read_parquet(tdir / "groundtruth_model_selection.parquet")
    gt_test = (None if stage_b_only
               else pd.read_parquet(tdir / "groundtruth_test.parquet"))
    if "model_selection" in kept_users:
        gt_sel = gt_sel[gt_sel["user_id"].astype(str).isin(
            kept_users["model_selection"])]
    if gt_test is not None and "test" in kept_users:
        gt_test = gt_test[gt_test["user_id"].astype(str).isin(kept_users["test"])]
    if max_users_per_snapshot:
        print(f"[ranker] capped users/snapshot to {max_users_per_snapshot:,} "
              f"(smoke scale cap, filtered at read)", flush=True)

    # Dense feature list: frozen base + any Phase 3A extra-source columns.
    features, log1p = resolve_feature_list(cands["ranker_train"])

    # Selection grid. Architecture + learning rate; epoch count comes from
    # early stopping against model_selection. Smoke restricts to one config.
    grid: List[Tuple[str, float]] = (
        [("mlp", 1e-3)] if smoke
        else [("mlp", 1e-3), ("mlp", 3e-4), ("deep_cross", 1e-3), ("deep_cross", 3e-4)]
    )
    if smoke:
        max_epochs = min(max_epochs, 3)

    # Feature spec for SELECTION runs: ranker_train rows only.
    spec_sel = build_temporal_feature_spec(cands["ranker_train"], features, log1p)
    train_inputs = build_temporal_tensors(cands["ranker_train"], spec_sel,
                                          features, log1p)
    sel_inputs = build_temporal_tensors(cands["model_selection"], spec_sel,
                                        features, log1p)
    sel_gt = {u: {pa} for u, pa in zip(gt_sel["user_id"].astype(str),
                                       gt_sel["parent_asin"].astype(str))}
    sel_pool = set(cands["model_selection"]["parent_asin"])
    n_pos = int(cands["ranker_train"]["label"].sum())
    n_neg = int((cands["ranker_train"]["label"] == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    print(f"[ranker] train rows={len(cands['ranker_train']):,} "
          f"(pos={n_pos:,}) selection rows={len(cands['model_selection']):,} "
          f"pos_weight={pos_weight:.1f} device={device}", flush=True)

    selection_runs = []
    best = None  # (recall, -idx) maximize; tie -> earlier grid entry
    best_model_state = None
    for gi, (arch, lr) in enumerate(grid):
        print(f"[ranker] selection run {gi+1}/{len(grid)}: arch={arch} lr={lr}",
              flush=True)
        # Seed BEFORE construction: model init draws from the global RNG, and
        # unseeded init makes selection irreproducible across processes.
        torch.manual_seed(seed + 1000 * (gi + 1))
        model = _make_model(arch, spec_sel).to(device)
        summary = train_pointwise_bce(
            model,
            train_inputs=train_inputs,
            val_inputs=sel_inputs,
            val_user_ids=cands["model_selection"]["user_id"].to_numpy(),
            val_pa=cands["model_selection"]["parent_asin"].to_numpy(),
            val_groundtruth=sel_gt,
            candidate_pool=sel_pool,
            epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            pos_weight=pos_weight,
        )
        hist = summary["history"]
        recalls = [h["ranker_val_metrics"]["Recall@100"] for h in hist]
        best_epoch = int(np.argmax(recalls)) + 1
        run_rec = {
            "arch": arch, "lr": lr,
            "best_selection_recall@100": summary["best_ranker_val_recall@100"],
            "best_epoch": best_epoch,
            "n_epochs_run": len(hist),
        }
        selection_runs.append(run_rec)
        key = (summary["best_ranker_val_recall@100"], -gi)
        if best is None or key > best[0]:
            best = (key, run_rec)
            if stage_b_only:
                # train_pointwise_bce reloaded the run's best epoch already.
                best_model_state = {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
        del model

    chosen = best[1]
    print(f"[ranker] SELECTED arch={chosen['arch']} lr={chosen['lr']} "
          f"epochs={chosen['best_epoch']} "
          f"(selection R@100={chosen['best_selection_recall@100']:.4f})", flush=True)

    if stage_b_only:
        # ---- Stage B report: selection-snapshot metrics ONLY ------------------
        model = _make_model(chosen["arch"], spec_sel).to(device)
        model.load_state_dict(best_model_state)
        sel_df = cands["model_selection"]
        sel_eval = _eval_test(model, sel_df,
                              build_temporal_tensors(sel_df, spec_sel,
                                                     features, log1p),
                              gt_sel, device)
        report = {
            "category": category,
            "variant": variant,
            "stage": "B (model-selection confirmation; test snapshot untouched)",
            "started_utc": datetime.now(tz=timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - t0, 2),
            "features": list(features),
            "selection": {"grid": selection_runs, "chosen": chosen,
                          "metric": "model_selection Recall@100"},
            "model_selection_eval": sel_eval,
        }
        stage_b_dir = REPO_ROOT / "results" / "phase3a"
        stage_b_dir.mkdir(parents=True, exist_ok=True)
        out = stage_b_dir / f"{category}_stageB_{variant or 'base'}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[ranker] Stage B wrote {out} | selection R@100="
              f"{sel_eval['overall']['Recall@100']:.4f} "
              f"ceiling={sel_eval['candidate_ceiling_retrieved_coverage']:.4f}",
              flush=True)
        return report

    # Free the selection-phase tensors before building the refit set -- the
    # login-node cgroup cannot hold both generations at once.
    import gc
    del train_inputs, sel_inputs
    gc.collect()

    # ---- refit on train + selection rows, fixed epochs, no early stopping ----
    combined = pd.concat([cands["ranker_train"], cands["model_selection"]],
                         ignore_index=True)
    spec_final = build_temporal_feature_spec(combined, features, log1p)
    combined_inputs = build_temporal_tensors(combined, spec_final,
                                             features, log1p)
    n_pos_c = int(combined["label"].sum())
    n_neg_c = int((combined["label"] == 0).sum())
    pos_weight_c = n_neg_c / max(1, n_pos_c)
    torch.manual_seed(seed + 999)
    final_model = _make_model(chosen["arch"], spec_final).to(device)
    print(f"[ranker] refit on {len(combined):,} rows for exactly "
          f"{chosen['best_epoch']} epoch(s)...", flush=True)
    refit_log = train_fixed_epochs(
        final_model, combined_inputs, chosen["best_epoch"],
        batch_size=batch_size, lr=chosen["lr"], weight_decay=weight_decay,
        device=device, seed=seed, pos_weight=pos_weight_c,
    )

    # ---- final test -----------------------------------------------------------
    del combined_inputs
    gc.collect()
    test_inputs = build_temporal_tensors(cands["test"], spec_final,
                                         features, log1p)
    test_eval = _eval_test(final_model, cands["test"], test_inputs, gt_test, device)

    # Retrieval-only context on the same test candidates.
    retrieval_only = {}
    gt_all = {u: {pa} for u, pa in zip(gt_test["user_id"].astype(str),
                                       gt_test["parent_asin"].astype(str))}
    test_pool = set(cands["test"]["parent_asin"])
    for col, name in (("two_tower_score", "two_tower"),
                      ("popularity_score", "popularity"),
                      ("rule_score", "rule_based")):
        topk = _scores_to_topk(
            cands["test"][col].to_numpy(np.float32),
            cands["test"]["user_id"].to_numpy(),
            cands["test"]["parent_asin"].to_numpy(), k=100,
        )
        rep = build_split_report(f"test_{name}", topk, gt_all, test_pool,
                                 "candidate_union_top100", KS)
        retrieval_only[name] = rep.metrics

    report = {
        "category": category,
        "variant": variant,
        "features": list(features),
        "protocol": "three_snapshot_walk_forward",
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "device": device,
        "provenance": {
            "training_data": "candidates_ranker_train.parquet (labels T0->T1)",
            "selection_data": "candidates_model_selection.parquet (labels T1->T2)",
            "refit_data": "ranker_train + model_selection candidates",
            "refit_fixed_epochs": chosen["best_epoch"],
            "no_early_stopping_on_refit": True,
            "test_labels_used_for_fitting_or_tuning": False,
        },
        "selection": {
            "grid": selection_runs,
            "chosen": chosen,
            "metric": "model_selection Recall@100",
        },
        "refit": {
            "n_rows": int(len(combined)),
            "pos_weight": pos_weight_c,
            "log": refit_log,
        },
        "final_test": test_eval,
        "retrieval_only_on_test_candidates": retrieval_only,
        "config": {
            "max_epochs": max_epochs, "batch_size": batch_size,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed, "smoke": smoke,
            "dense_features": list(TEMPORAL_DENSE_FEATURES),
            "categorical_features": list(TEMPORAL_CATEGORICALS),
        },
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    out = results_dir / f"{category}_ranker{suffix}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[ranker] wrote {out} | final test R@10="
          f"{test_eval['overall']['Recall@10']:.4f} "
          f"R@100={test_eval['overall']['Recall@100']:.4f} "
          f"ceiling={test_eval['candidate_ceiling_retrieved_coverage']:.4f}",
          flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-users", type=int, default=0,
                   help="Per-snapshot deterministic user cap (0 = all users). "
                        "For memory-bounded smoke runs only.")
    p.add_argument("--variant", default=None,
                   help="Read candidates from temporal_ranker/variants/{name}/")
    p.add_argument("--stage-b", action="store_true",
                   help="Stop after model selection; never touch the test "
                        "snapshot (Phase 3A Stage B confirmation).")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(args.category, max_epochs=args.max_epochs, batch_size=args.batch_size,
        seed=args.seed, smoke=args.smoke, max_users_per_snapshot=args.max_users,
        variant=args.variant, stage_b_only=args.stage_b)


if __name__ == "__main__":
    main()
