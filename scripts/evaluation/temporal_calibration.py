"""Evaluation & Calibration on the three-snapshot ranker prediction dumps.

Advisor spec (2026-07-26): prediction distribution / bias analysis on the
three snapshots (by set / category / item review-count bucket / user history
cohort); wire the vectorized metrics (AUC / gAUC / MRR / nDCG@m / P / R)
with proper per-user grouping; and a rule calibration layer — a
<bucket, threshold> config that zeroes calibrated probabilities below the
bucket's threshold — with its metric impact quantified.

Inputs are the per-row dumps written by
    python -m scripts.ranker.train_temporal_ranker --dump-predictions \
        --results-dir results/calibration/ranker_rerun ...
under {candidate_dir}/predictions/ .

CLI:
    python -m scripts.evaluation.temporal_calibration \
        --category Video_Games --variant C2_k500_recent --stage fit
    ... --stage test   (only after fit wrote the frozen config)

Test-lock protocol (Phase 3A paradigm):
    fit   reads ONLY the train/selection dumps. Fits Platt scaling (a, b)
          and the per-cohort thresholds on the SELECTION model's
          out-of-sample model_selection predictions, then freezes the
          config. The test dump is never opened (it may not even exist).
    test  requires the frozen config; applies the frozen chain to the refit
          model's test predictions ONCE and reports.

Why calibration at all: the ranker trains weighted BCE with
pos_weight = n_neg / n_pos (~200-500), which shifts the learned log-odds up
by ~log(pos_weight) — raw sigmoid probabilities are inflated by orders of
magnitude; only the ranking is meaningful. Chains evaluated (a fixed,
pre-declared set — no test-time shopping):
    raw     sigmoid(logit)                          incumbent baseline
    prior   sigmoid(logit - log(pos_weight))        parameter-free rule; uses
            the scored model's own training pos_weight (never test-derived)
    platt   sigmoid(a * logit + b)                  fitted on selection, frozen
    final   chosen calibrator, then p < t_cohort -> 0   (the rule layer)

Success criteria (per MEMO): bucket calibration error goes DOWN and ranking
metrics do NOT regress — Recall gains are explicitly not the goal. Monotone
chains (a > 0, asserted) cannot change ranking metrics; only the rule layer
can, and its thresholds are chosen under a no-regression constraint on the
selection snapshot.

Known caveat (inherent to walk-forward refit, documented in every report):
calibration is fitted on the SELECTION model (trained on T0 only) but
applied to the REFIT model (trained on T0+T1, its own pos_weight). Fitting
on the refit model's train/selection dumps instead would calibrate against
memorized (in-sample) labels — those dumps feed distribution analysis only.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from scripts.evaluation.grouped_eval import evaluate_flat

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "calibration"

KS = (10, 50, 100)
PROB_BINS = 15
COHORT_NAMES = ("0", "1", "2", "3+")
# item_n_reviews_hist buckets; pool eligibility is >=5 historical events but
# TT/recency sources can surface below-threshold items, hence the [0,10) bin.
ITEM_BUCKET_EDGES = (10.0, 30.0, 100.0, 300.0)
ITEM_BUCKET_NAMES = ("[0,10)", "[10,30)", "[30,100)", "[100,300)", "300+")
# Sweep grid: per-cohort calibrated-probability quantiles tried as thresholds.
THRESHOLD_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
DIST_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _cohort_codes(n_history_events: np.ndarray) -> np.ndarray:
    """0/1/2/3+ history cohorts as codes 0..3 (same mapping as the ranker)."""
    return np.clip(n_history_events.astype(np.int64), 0, 3)


def _item_bucket_codes(item_n_reviews_hist: np.ndarray) -> np.ndarray:
    return np.digitize(item_n_reviews_hist.astype(np.float64),
                       ITEM_BUCKET_EDGES)


def _quantile_dict(x: np.ndarray) -> Dict[str, float]:
    qs = np.quantile(x, DIST_QUANTILES)
    return {f"q{int(q * 100):02d}": float(v)
            for q, v in zip(DIST_QUANTILES, qs)}


def dist_stats(logits: np.ndarray, probs: np.ndarray,
               labels: np.ndarray) -> Dict:
    """Distribution + bias summary for one row subset."""
    if len(logits) == 0:
        return {"n": 0}
    pos_rate = float(labels.mean())
    pos = labels > 0.5
    return {
        "n": int(len(logits)),
        "pos_rate": pos_rate,
        "logit": {"mean": float(logits.mean()), "std": float(logits.std()),
                  **_quantile_dict(logits)},
        "prob": {"mean": float(probs.mean()), "std": float(probs.std()),
                 **_quantile_dict(probs)},
        # ranking-separation view (per MEMO: read the distribution through the
        # ranking decomposition, not as literal probabilities)
        "logit_pos_mean": float(logits[pos].mean()) if pos.any() else None,
        "logit_neg_mean": float(logits[~pos].mean()) if (~pos).any() else None,
        # calibration-bias view
        "bias_mean_prob_over_pos_rate": (float(probs.mean()) / pos_rate
                                         if pos_rate > 0 else None),
        "brier": float(np.mean((probs - labels) ** 2)),
    }


def ece_table(probs: np.ndarray, labels: np.ndarray,
              n_bins: int = PROB_BINS) -> Dict:
    """Equal-width-bin Expected Calibration Error + reliability table."""
    if len(probs) == 0:
        return {"ece": None, "n": 0, "bins": []}
    idx = np.minimum((probs * n_bins).astype(np.int64), n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=probs, minlength=n_bins)
    sum_y = np.bincount(idx, weights=labels, minlength=n_bins)
    n = float(len(probs))
    bins: List[Dict] = []
    ece = 0.0
    for b in range(n_bins):
        if cnt[b] == 0:
            continue
        mp, pr = sum_p[b] / cnt[b], sum_y[b] / cnt[b]
        ece += cnt[b] / n * abs(mp - pr)
        bins.append({"bin": f"[{b / n_bins:.3f},{(b + 1) / n_bins:.3f})",
                     "n": int(cnt[b]), "mean_prob": float(mp),
                     "pos_rate": float(pr), "gap": float(mp - pr)})
    return {"ece": float(ece), "n": int(n), "bins": bins}


def ece_quantile(probs: np.ndarray, labels: np.ndarray,
                 n_bins: int = PROB_BINS) -> Dict:
    """Quantile-bin (adaptive) ECE — resolves the low-probability mass that a
    single equal-width bin would swallow at pos_rate ~1e-3."""
    if len(probs) == 0:
        return {"ece": None, "n": 0, "bins": []}
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return ece_table(probs, labels, n_bins=n_bins)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, len(edges) - 2)
    nb = len(edges) - 1
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    sum_p = np.bincount(idx, weights=probs, minlength=nb)
    sum_y = np.bincount(idx, weights=labels, minlength=nb)
    n = float(len(probs))
    bins, ece = [], 0.0
    for b in range(nb):
        if cnt[b] == 0:
            continue
        mp, pr = sum_p[b] / cnt[b], sum_y[b] / cnt[b]
        ece += cnt[b] / n * abs(mp - pr)
        bins.append({"bin": f"[{edges[b]:.3e},{edges[b + 1]:.3e}]",
                     "n": int(cnt[b]), "mean_prob": float(mp),
                     "pos_rate": float(pr), "gap": float(mp - pr)})
    return {"ece": float(ece), "n": int(n), "bins": bins}


def fit_platt(logits: np.ndarray, labels: np.ndarray,
              init_shift: float, max_iter: int = 100) -> Dict:
    """Deterministic full-batch Platt scaling: minimize UNWEIGHTED BCE of
    sigmoid(a * logit + b). Initialized at the prior correction (a=1,
    b=-log pos_weight) so LBFGS starts from the theoretically-motivated
    point."""
    x = torch.from_numpy(logits.astype(np.float64))
    y = torch.from_numpy(labels.astype(np.float64))
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(init_shift), dtype=torch.float64,
                     requires_grad=True)
    bce = torch.nn.BCEWithLogitsLoss()
    init_nll = float(bce(a.detach() * x + b.detach(), y))
    opt = torch.optim.LBFGS([a, b], max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = bce(a * x + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    a_f, b_f = float(a.detach()), float(b.detach())
    return {"a": a_f, "b": b_f, "init_shift": float(init_shift),
            "init_nll": init_nll,
            "final_nll": float(bce(torch.tensor(a_f) * x + torch.tensor(b_f), y)),
            "monotone": a_f > 0}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _pred_dir(category: str, variant: Optional[str],
              processed_dir: Path = PROCESSED_DIR,
              pred_subdir: str = "predictions") -> Path:
    tdir = processed_dir / category / "temporal_ranker"
    cdir = (tdir / "variants" / variant) if variant else tdir
    return cdir / pred_subdir


def _dump_fingerprint(meta: Dict) -> Dict:
    """Identity of one dump generation; frozen configs are bound to it so the
    test stage refuses to run against regenerated/mismatched dumps."""
    return {
        "created_utc": meta.get("created_utc"),
        "n_rows": meta.get("n_rows"),
        "max_users_per_snapshot": meta.get("max_users_per_snapshot", 0),
        "smoke": meta.get("smoke", False),
        "seed": meta.get("seed"),
    }


def _is_capped(meta: Dict) -> bool:
    return bool(meta.get("max_users_per_snapshot")) or bool(meta.get("smoke"))


def _dump_gt_counts(d: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Cohort denominators derived from the dump itself — used for capped
    (smoke) dumps, where the full ground-truth counts would silently deflate
    every overall recall."""
    counts, total = {}, 0
    for i, name in enumerate(COHORT_NAMES):
        m = d["cohort"] == i
        n = int(len(np.unique(d["user"][m]))) if m.any() else 0
        counts[name] = n
        total += n
    counts["total"] = total
    return counts


def load_dump(path: Path, *, need_user: bool) -> Dict[str, np.ndarray]:
    """Load one prediction dump as flat numpy arrays; user ids are factorized
    to int codes immediately (62M raw strings would dominate memory)."""
    cols = ["logit", "label", "n_history_events", "item_n_reviews_hist"]
    if need_user:
        cols.append("user_id")
    df = pd.read_parquet(path, columns=cols)
    out = {
        "logit": df["logit"].to_numpy(np.float64),
        "label": df["label"].to_numpy(np.float64),
        "cohort": _cohort_codes(df["n_history_events"].to_numpy()),
        "item_bucket": _item_bucket_codes(
            df["item_n_reviews_hist"].to_numpy()),
    }
    if need_user:
        codes, uniq = pd.factorize(df["user_id"])
        out["user"] = codes
        # Keep the code -> user_id map: |P_u| lives in the groundtruth frame
        # keyed by the string id, while every metric here is keyed by code.
        out["user_ids"] = np.asarray(uniq.astype(str), dtype=object)
    return out


def _gt_path(category: str, snapshot: str, processed_dir: Path,
             label_mode: str = "first_positive") -> Path:
    """The groundtruth frame for `snapshot` under the requested label rule.

    "first_positive" -> groundtruth_{snapshot}.parquet (each user's first
    positive in the window; the frozen frame every published number keys off)
    "all_positive"   -> groundtruth_all_{snapshot}.parquet (every distinct
    positive). Reading the wrong one against an all-positive prediction dump
    silently sets |P_u| = 1 for every user and inflates recall.
    """
    if label_mode not in ("first_positive", "all_positive"):
        raise ValueError(f"unknown label_mode {label_mode!r}")
    prefix = "groundtruth_" if label_mode == "first_positive" else "groundtruth_all_"
    return processed_dir / category / "temporal_ranker" / f"{prefix}{snapshot}.parquet"


def _gt_cohort_counts(category: str, snapshot: str,
                      processed_dir: Path = PROCESSED_DIR,
                      label_mode: str = "first_positive") -> Dict[str, int]:
    """DISTINCT ground-truth USERS, overall and per history cohort.

    These are the n_gt_users denominators of recall@k_overall /
    hit_rate@k_overall / hits@k_per_gt_user. The groundtruth frame carries one
    row per (user, positive item), so `len(gt)` and a bincount over every row
    count POSITIVES: at 2.4 positives per user the denominators come out 2.4x
    too large and every headline rate is deflated by the same factor (and the
    inflation is uneven across cohorts, so the cohort ordering moves too).
    n_history_events is a property of the user's history, constant within a
    user, so deduplicating on user_id first is lossless — asserted, because a
    user with two different values would be counted in two cohorts at once.
    """
    gt = pd.read_parquet(_gt_path(category, snapshot, processed_dir, label_mode),
                         columns=["user_id", "n_history_events"])
    users = gt.drop_duplicates(subset=["user_id"])
    assert len(users) == len(gt.drop_duplicates(
        subset=["user_id", "n_history_events"])), (
        "n_history_events varies within a user — cohort assignment ambiguous")
    codes = _cohort_codes(users["n_history_events"].to_numpy())
    counts = np.bincount(codes, minlength=4)
    return {"total": int(len(users)),
            **{COHORT_NAMES[i]: int(counts[i]) for i in range(4)}}


def _true_positive_counts_lm(label_mode, *args, **kw):
    """Thin adapter: keeps label_mode out of the positional signature that
    existing call sites and tests rely on."""
    return _true_positive_counts(*args, label_mode=label_mode, **kw)


def _true_positive_counts(user_ids: np.ndarray, category: str, snapshot: str,
                          processed_dir: Path = PROCESSED_DIR,
                          label_mode: str = "first_positive"
                          ) -> Optional[Dict[int, int]]:
    """|P_u| per dump user CODE — every positive the user has in the label
    window, including the ones retrieval never surfaced.

    The dump's `label` column only marks positives that made the candidate
    list, so without this evaluate_flat divides by the retrieved-positive
    count: a user who bought 4 items, of which we retrieved 1 and ranked it
    first, scores recall 1.0 instead of 0.25. Under one-positive-per-user
    groundtruth every value here is 1 and the whole thing is a no-op (a missed
    target leaves the user with no positive row at all), which is why it can be
    wired in unconditionally. Returns None when the frame is absent (capped
    smoke fixtures), leaving the old retrieved-positive behaviour.
    """
    import pyarrow.parquet as pq

    path = _gt_path(category, snapshot, processed_dir, label_mode)
    if not path.exists():
        return None
    cols = ["user_id"]
    if "parent_asin" in pq.read_schema(path).names:
        cols.append("parent_asin")
    gt = pd.read_parquet(path, columns=cols)
    gt["user_id"] = gt["user_id"].astype(str)
    if "parent_asin" in gt.columns:
        # Both label builders dedup (user, item) upstream; repeating it here
        # keeps a duplicated row from inflating |P_u| and deflating recall.
        gt = gt.drop_duplicates(subset=["user_id", "parent_asin"])
    n_pos = gt.groupby("user_id").size()
    aligned = n_pos.reindex(np.asarray(user_ids, dtype=object))
    missing = int(aligned.isna().sum())
    if missing:
        # Loud on purpose: a scored user with no groundtruth row means the dump
        # and the label frame are from different generations, and a silent
        # |P_u| = 0 would clamp that user's recall denominator to 1.
        raise ValueError(
            f"{missing} of {len(user_ids)} scored users have no row in "
            f"{path.name} — the prediction dump and the "
            "label frame are out of sync; regenerate one of them")
    return {i: int(v) for i, v in enumerate(aligned.to_numpy())}


def _label_mode_of(category: str, snapshot: str,
                   processed_dir: Path = PROCESSED_DIR) -> str:
    """Which label protocol the groundtruth frame for `snapshot` follows.

    Preferred source is the pipeline's own manifest, which names the mode even
    when an all_positive window happens to hold one positive per user; the
    fallback counts rows against distinct users, which is exactly what decides
    whether recall@k_overall and hits@k_per_gt_user are the same number.
    """
    tdir = processed_dir / category / "temporal_ranker"
    manifest = tdir / "snapshot_manifest.json"
    if manifest.exists():
        try:
            with open(manifest) as f:
                labels = json.load(f)["snapshots"][snapshot]["labels"]
            if labels.get("label_mode"):
                return str(labels["label_mode"])
        except (KeyError, TypeError, ValueError):
            pass
    gt_path = _gt_path(category, snapshot, processed_dir)
    if not gt_path.exists():
        return "unknown"
    u = pd.read_parquet(gt_path, columns=["user_id"])["user_id"]
    return "first_positive" if u.nunique() == len(u) else "all_positive"


def _published_label_mode(pub: Dict) -> Dict[str, str]:
    """Label mode behind a published results/phase2_temporal report."""
    ft = pub.get("final_test", {})
    n_users = ft.get("n_groundtruth_users")
    n_pos = ft.get("n_groundtruth_positives")
    if n_pos is not None and n_users:
        return {"mode": "all_positive" if n_pos > n_users else "first_positive",
                "source": "n_groundtruth_positives vs n_groundtruth_users"}
    return {"mode": "first_positive",
            "source": "assumed — the report predates the label_mode switch "
                      "and records no positive count"}


# ---------------------------------------------------------------------------
# Analysis blocks
# ---------------------------------------------------------------------------

def distribution_block(d: Dict[str, np.ndarray],
                       probs: np.ndarray) -> Dict:
    """Overall + per-cohort + per-item-bucket distribution stats."""
    block = {"overall": dist_stats(d["logit"], probs, d["label"]),
             "by_user_cohort": {}, "by_item_review_bucket": {}}
    for i, name in enumerate(COHORT_NAMES):
        m = d["cohort"] == i
        block["by_user_cohort"][name] = dist_stats(
            d["logit"][m], probs[m], d["label"][m])
    for i, name in enumerate(ITEM_BUCKET_NAMES):
        m = d["item_bucket"] == i
        block["by_item_review_bucket"][name] = dist_stats(
            d["logit"][m], probs[m], d["label"][m])
    return block


def calibration_block(d: Dict[str, np.ndarray], probs: np.ndarray,
                      with_tables: bool = True) -> Dict:
    """ECE overall + per-cohort + per-item-bucket for one probability chain."""
    overall_ew = ece_table(probs, d["label"])
    overall_q = ece_quantile(probs, d["label"])
    block = {
        "ece": overall_ew["ece"],
        "ece_quantile_bins": overall_q["ece"],
        "by_user_cohort": {}, "by_item_review_bucket": {},
    }
    if with_tables:
        block["reliability_equal_width"] = overall_ew["bins"]
        block["reliability_quantile"] = overall_q["bins"]
    for i, name in enumerate(COHORT_NAMES):
        m = d["cohort"] == i
        block["by_user_cohort"][name] = ece_table(
            probs[m], d["label"][m])["ece"]
    for i, name in enumerate(ITEM_BUCKET_NAMES):
        m = d["item_bucket"] == i
        block["by_item_review_bucket"][name] = ece_table(
            probs[m], d["label"][m])["ece"]
    return block


def metrics_block(d: Dict[str, np.ndarray], scores: np.ndarray,
                  gt_counts: Dict[str, int], *, batch_users: int,
                  device: str,
                  true_positives: Optional[Dict[int, int]] = None) -> Dict:
    """Grouped ranking metrics overall + per user cohort."""
    block = {"overall": evaluate_flat(
        d["user"], scores, d["label"], ks=KS, batch_users=batch_users,
        n_gt_users=gt_counts["total"],
        true_positives_per_user=true_positives, device=device)}
    for i, name in enumerate(COHORT_NAMES):
        m = d["cohort"] == i
        block[f"cohort_{name}"] = evaluate_flat(
            d["user"][m], scores[m], d["label"][m], ks=KS,
            batch_users=batch_users, n_gt_users=max(
                gt_counts[name],
                len(np.unique(d["user"][m])) if m.any() else 0),
            true_positives_per_user=true_positives, device=device)
    return block


def apply_thresholds(probs: np.ndarray, cohort: np.ndarray,
                     thresholds: Dict[str, float]) -> np.ndarray:
    """The rule calibration layer: p < t_cohort -> 0."""
    out = probs.copy()
    for i, name in enumerate(COHORT_NAMES):
        t = float(thresholds.get(name, 0.0))
        if t > 0:
            m = (cohort == i) & (out < t)
            out[m] = 0.0
    return out


def sweep_thresholds(d: Dict[str, np.ndarray], probs_cal: np.ndarray,
                     gt_counts: Dict[str, int], *, batch_users: int,
                     device: str,
                     true_positives: Optional[Dict[int, int]] = None) -> Dict:
    """Per-cohort threshold sweep on the SELECTION snapshot only.

    Pre-declared choice rule: the LARGEST grid threshold whose cohort metrics
    show no regression (recall@100_overall, recall@10_overall, gauc) AND
    whose cohort equal-width ECE strictly improves; 0.0 (no-op) otherwise.
    Cohorts partition users, so per-cohort choices are independent.

    Zeroed rows sink deterministically below every surviving row, and tied
    rows earn no AUC/top-k credit for positives (grouped_eval tie policy) —
    a threshold that zeroes a user's target is therefore honestly penalized,
    never rescued by argsort accidents.
    """
    result = {"grid_quantiles": list(THRESHOLD_QUANTILES),
              "cohorts": {}, "chosen": {}}
    for i, name in enumerate(COHORT_NAMES):
        m = d["cohort"] == i
        if not m.any():
            result["cohorts"][name] = {"n_rows": 0, "sweep": []}
            result["chosen"][name] = 0.0
            continue
        users, probs, labels = d["user"][m], probs_cal[m], d["label"][m]
        # Hoisted deliberately: a threshold only rewrites SCORES, never the row
        # or user set, so the cohort's denominator is the same on every rung of
        # the grid (verified: rows / distinct users / positives are constant
        # across the sweep). It therefore cancels in the accept test AND is the
        # right divisor for the reported recalls. What it must be is the count
        # of distinct ground-truth USERS — gt_counts supplies that.
        n_gt = max(gt_counts[name], len(np.unique(users)))
        base = evaluate_flat(users, probs, labels, ks=KS,
                             batch_users=batch_users, n_gt_users=n_gt,
                             true_positives_per_user=true_positives,
                             device=device)
        base_ece = ece_table(probs, labels)["ece"]
        rows = []
        chosen = 0.0
        for q in THRESHOLD_QUANTILES:
            t = float(np.quantile(probs, q))
            if t <= 0:
                continue
            s = np.where(probs >= t, probs, 0.0)
            met = evaluate_flat(users, s, labels, ks=KS,
                                batch_users=batch_users, n_gt_users=n_gt,
                                true_positives_per_user=true_positives,
                                device=device)
            e = ece_table(s, labels)["ece"]
            ok = (met["recall@100_overall"] >= base["recall@100_overall"] - 1e-12
                  and met["recall@10_overall"] >= base["recall@10_overall"] - 1e-12
                  and met["gauc"] >= base["gauc"] - 1e-9
                  and e < base_ece)
            rows.append({
                "quantile": q, "threshold": t,
                "frac_zeroed": float((s == 0).mean()),
                "recall@10_overall": met["recall@10_overall"],
                "recall@100_overall": met["recall@100_overall"],
                "gauc": met["gauc"], "ndcg@10": met["ndcg@10"],
                "mrr": met["mrr"], "ece": e, "accepted": bool(ok),
            })
            if ok:
                chosen = max(chosen, t)
        result["cohorts"][name] = {
            "n_rows": int(m.sum()),
            "baseline": {"recall@10_overall": base["recall@10_overall"],
                         "recall@100_overall": base["recall@100_overall"],
                         "gauc": base["gauc"], "ndcg@10": base["ndcg@10"],
                         "mrr": base["mrr"], "ece": base_ece},
            "sweep": rows,
        }
        result["chosen"][name] = chosen
    return result


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_fit(category: str, variant: Optional[str], *,
            batch_users: int = 4096, device: str = "cpu",
            processed_dir: Path = PROCESSED_DIR,
            results_dir: Path = RESULTS_DIR,
            pred_subdir: str = "predictions",
            label_mode: str = "first_positive",
            force: bool = False) -> Dict:
    """Distribution analysis on the pre-test sets + calibration fitting +
    threshold sweep. NEVER touches the test dump."""
    tag = f"{category}_{variant or 'base'}"
    spent = results_dir / f"{tag}_test.json"
    if spent.exists() and not force:
        raise RuntimeError(
            f"{spent} exists — the locked test application was already "
            "spent; refitting would restart the protocol. Pass force=True "
            "(--force) only to do that deliberately.")
    pdir = _pred_dir(category, variant, processed_dir, pred_subdir)
    with open(pdir / "meta.json") as f:
        meta = json.load(f)
    w_sel = float(meta["selection_model"]["pos_weight"])
    w_refit = float(meta["refit_model"]["pos_weight"])
    capped = _is_capped(meta)
    if capped:
        print("[calibration] WARNING: dump was produced under a user cap / "
              "smoke run — denominators come from the dump itself and "
              "results are NOT full-scale", flush=True)

    # --- distribution analysis over the pre-test sets (raw sigmoid) --------
    distributions = {}
    for key, fname in (
            ("refit_model@ranker_train", "refit_model_ranker_train.parquet"),
            ("refit_model@model_selection",
             "refit_model_model_selection.parquet")):
        d = load_dump(pdir / fname, need_user=False)
        distributions[key] = distribution_block(d, _sigmoid(d["logit"]))
        del d

    # --- the calibration fitting set: selection model, out-of-sample -------
    sel = load_dump(pdir / "selection_model_model_selection.parquet",
                    need_user=True)
    probs_raw = _sigmoid(sel["logit"])
    distributions["selection_model@model_selection"] = distribution_block(
        sel, probs_raw)

    gt_counts = (_dump_gt_counts(sel) if capped else
                 _gt_cohort_counts(category, "model_selection", processed_dir,
                                   label_mode))
    # |P_u| for the fitting snapshot. Without it every recall below divides by
    # the positives retrieval happened to surface instead of the ones the user
    # actually has, so a threshold could be accepted on a recall that cannot
    # see the misses it is being judged on.
    true_pos = _true_positive_counts_lm(label_mode, sel["user_ids"], category,
                                     "model_selection", processed_dir)

    # --- grouped ranking metrics on raw scores -----------------------------
    metrics_raw = metrics_block(sel, sel["logit"], gt_counts,
                                batch_users=batch_users, device=device,
                                true_positives=true_pos)

    # --- calibrators --------------------------------------------------------
    prior_shift = -math.log(w_sel)
    probs_prior = _sigmoid(sel["logit"] + prior_shift)
    platt = fit_platt(sel["logit"], sel["label"], prior_shift)
    probs_platt = _sigmoid(platt["a"] * sel["logit"] + platt["b"])

    calibration = {
        "raw": calibration_block(sel, probs_raw),
        "prior": calibration_block(sel, probs_prior),
        "platt": calibration_block(sel, probs_platt),
    }
    # Pre-declared rule: platt iff it is monotone and beats prior on overall
    # equal-width ECE; otherwise the parameter-free prior correction.
    chosen = ("platt" if platt["monotone"]
              and calibration["platt"]["ece"] <= calibration["prior"]["ece"]
              else "prior")
    probs_cal = probs_platt if chosen == "platt" else probs_prior

    # --- rule layer: per-cohort threshold sweep -----------------------------
    sweep = sweep_thresholds(sel, probs_cal, gt_counts,
                             batch_users=batch_users, device=device,
                             true_positives=true_pos)

    frozen = {
        "category": category,
        "variant": variant,
        "fitted_on": "selection_model_model_selection.parquet "
                     "(out-of-sample, pre-test)",
        "pred_subdir": pred_subdir,
        "capped_run": capped,
        "dump_fingerprint": _dump_fingerprint(meta),
        "pos_weight_selection": w_sel,
        "pos_weight_refit": w_refit,
        "prior_shift_rule": "logit - log(pos_weight of the scored model)",
        "platt": platt,
        "chosen_calibrator": chosen,
        "cohort_thresholds": sweep["chosen"],
        "threshold_quantile_grid": list(THRESHOLD_QUANTILES),
        "prob_bins": PROB_BINS,
        "item_bucket_edges": list(ITEM_BUCKET_EDGES),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "test_dump_untouched": True,
    }

    report = {
        "stage": "fit",
        "category": category,
        "variant": variant,
        "capped_run": capped,
        "pred_meta": meta,
        "groundtruth_cohort_counts": gt_counts,
        "distributions": distributions,
        "metrics_raw_selection_model": metrics_raw,
        "calibration_selection": calibration,
        "threshold_sweep": sweep,
        "frozen": frozen,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{category}_{variant or 'base'}"
    with open(results_dir / f"{tag}_fit.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(results_dir / f"{tag}_frozen.json", "w") as f:
        json.dump(frozen, f, indent=2, default=str)
    print(f"[calibration] fit: chosen={chosen} platt=(a={platt['a']:.4f}, "
          f"b={platt['b']:.4f}) thresholds={sweep['chosen']} -> "
          f"{results_dir / (tag + '_frozen.json')}", flush=True)
    return report


def run_test(category: str, variant: Optional[str], *,
             batch_users: int = 4096, device: str = "cpu",
             processed_dir: Path = PROCESSED_DIR,
             results_dir: Path = RESULTS_DIR,
             pred_subdir: str = "predictions",
             label_mode: str = "first_positive",
             force: bool = False) -> Dict:
    """Single locked application of the frozen calibration chain to the refit
    model's test predictions."""
    tag = f"{category}_{variant or 'base'}"
    frozen_path = results_dir / f"{tag}_frozen.json"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"{frozen_path} missing — run --stage fit first (test-lock)")
    out_path = results_dir / f"{tag}_test.json"
    if out_path.exists() and not force:
        raise RuntimeError(
            f"{out_path} exists — the test stage is one-shot; pass "
            "force=True (--force) only to deliberately re-spend it.")
    with open(frozen_path) as f:
        frozen = json.load(f)
    pdir = _pred_dir(category, variant, processed_dir, pred_subdir)

    # The frozen config is bound to one dump generation; a regenerated or
    # different-scale dump must be re-fitted, not silently scored.
    with open(pdir / "meta.json") as f:
        meta = json.load(f)
    fp = _dump_fingerprint(meta)
    if fp != frozen.get("dump_fingerprint"):
        raise RuntimeError(
            "prediction dumps changed since the frozen config was fitted "
            f"(dump {fp} vs frozen {frozen.get('dump_fingerprint')}) — "
            "rerun --stage fit against the current dumps")

    test = load_dump(pdir / "refit_model_test.parquet", need_user=True)
    capped = _is_capped(meta)
    if capped:
        print("[calibration] WARNING: capped/smoke dump — denominators from "
              "the dump itself; NOT full-scale results", flush=True)
    gt_counts = (_dump_gt_counts(test) if capped else
                 _gt_cohort_counts(category, "test", processed_dir, label_mode))
    true_pos = _true_positive_counts_lm(label_mode, test["user_ids"], category, "test",
                                     processed_dir)

    w_refit = float(frozen["pos_weight_refit"])
    platt = frozen["platt"]
    probs_raw = _sigmoid(test["logit"])
    probs_prior = _sigmoid(test["logit"] - math.log(w_refit))
    probs_platt = _sigmoid(platt["a"] * test["logit"] + platt["b"])
    chosen = frozen["chosen_calibrator"]
    probs_cal = probs_platt if chosen == "platt" else probs_prior
    probs_final = apply_thresholds(probs_cal, test["cohort"],
                                   frozen["cohort_thresholds"])

    distributions = {
        "refit_model@test_raw": distribution_block(test, probs_raw),
        "refit_model@test_calibrated": distribution_block(test, probs_cal),
    }
    calibration = {
        "raw": calibration_block(test, probs_raw),
        "prior": calibration_block(test, probs_prior),
        "platt": calibration_block(test, probs_platt),
        "final_with_thresholds": calibration_block(test, probs_final),
    }

    # Ranking metrics: monotone chains are ranking-identical to raw (a > 0
    # asserted at fit time), so raw covers raw/prior/platt; the rule layer is
    # the only chain that can move them.
    metrics = {
        "raw": metrics_block(test, test["logit"], gt_counts,
                             batch_users=batch_users, device=device,
                             true_positives=true_pos),
        "final_with_thresholds": metrics_block(
            test, probs_final, gt_counts,
            batch_users=batch_users, device=device,
            true_positives=true_pos),
    }

    # Consistency anchor vs the published locked ranker report (read-only).
    anchor = None
    published = (REPO_ROOT / "results" / "phase2_temporal" /
                 f"{category}_ranker_{variant}.json" if variant else
                 REPO_ROOT / "results" / "phase2_temporal" /
                 f"{category}_ranker.json")
    if published.exists():
        with open(published) as f:
            pub = json.load(f)
        # Both sides are macro per-user recall, but only against the SAME
        # ground truth. A first_positive number answers "did we surface the
        # user's next purchase"; an all_positive one answers "what fraction of
        # everything they bought did we surface" — on the same model the second
        # is the smaller number. Comparing them silently would read a protocol
        # change as a reproducibility failure (or, worse, hide a real drift
        # behind an expected one), so the mode is recorded on both sides and
        # the tripwire only means anything when they match.
        pub_mode = _published_label_mode(pub)
        rerun_mode = _label_mode_of(category, "test", processed_dir)
        modes_match = pub_mode["mode"] == rerun_mode
        anchor = {
            "published_recall@100": pub["final_test"]["overall"].get("Recall@100"),
            "published_label_mode": pub_mode["mode"],
            "published_label_mode_source": pub_mode["source"],
            "this_rerun_recall@100_overall":
                metrics["raw"]["overall"].get("recall@100_overall"),
            "this_rerun_label_mode": rerun_mode,
            "label_modes_match": bool(modes_match),
            "note": "rerun uses identical frozen config + seed; small drift "
                    "= GPU nondeterminism, large drift = investigate",
        }
        if not modes_match:
            anchor["note"] += (
                f" — NOT VALID HERE: published label_mode={pub_mode['mode']} "
                f"vs rerun label_mode={rerun_mode}, so the two sides answer "
                "different questions and any gap is the protocol change, not "
                "the model; re-anchor against a report built in the same mode")
            print("[calibration] WARNING: consistency anchor is comparing "
                  f"label_mode={pub_mode['mode']} (published) with "
                  f"label_mode={rerun_mode} (this rerun) — not like for like",
                  flush=True)

    ece_raw = calibration["raw"]["ece"]
    ece_final = calibration["final_with_thresholds"]["ece"]
    cohort_improved = {
        c: (calibration["final_with_thresholds"]["by_user_cohort"][c] is not None
            and calibration["raw"]["by_user_cohort"][c] is not None
            and calibration["final_with_thresholds"]["by_user_cohort"][c]
            <= calibration["raw"]["by_user_cohort"][c])
        for c in COHORT_NAMES
    }
    m_raw, m_fin = metrics["raw"]["overall"], \
        metrics["final_with_thresholds"]["overall"]
    success = {
        "overall_ece_down": bool(ece_final < ece_raw),
        "cohort_ece_down": cohort_improved,
        "no_regression_recall@100": bool(
            m_fin["recall@100_overall"] >= m_raw["recall@100_overall"] - 1e-9),
        "no_regression_recall@10": bool(
            m_fin["recall@10_overall"] >= m_raw["recall@10_overall"] - 1e-9),
        "no_regression_gauc": bool(m_fin["gauc"] >= m_raw["gauc"] - 1e-9),
    }

    report = {
        "stage": "test (single locked application)",
        "category": category,
        "variant": variant,
        "capped_run": capped,
        "frozen": frozen,
        "groundtruth_cohort_counts": gt_counts,
        "distributions": distributions,
        "calibration_test": calibration,
        "metrics_test": metrics,
        "consistency_anchor": anchor,
        "success_criteria": success,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[calibration] test: ece raw={ece_raw:.4g} -> final={ece_final:.4g} "
          f"| R@100 {m_raw['recall@100_overall']:.4f} -> "
          f"{m_fin['recall@100_overall']:.4f} | success={success} -> "
          f"{results_dir / (tag + '_test.json')}", flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--variant", default="C2_k500_recent")
    p.add_argument("--stage", required=True, choices=["fit", "test"])
    p.add_argument("--batch-users", type=int, default=4096)
    p.add_argument("--device", default="cpu")
    p.add_argument("--label-mode", default="first_positive",
                   choices=["first_positive", "all_positive"],
                   help="Which groundtruth frame supplies n_gt_users and |P_u|. "
                        "Must match the label rule the prediction dump was "
                        "produced under, or recall is computed against the "
                        "wrong denominator.")
    p.add_argument("--pred-subdir", default="predictions",
                   help="predictions_smoke to analyze a quarantined capped "
                        "dump (results are flagged capped_run)")
    p.add_argument("--force", action="store_true",
                   help="Deliberately restart/re-spend a one-shot stage.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    variant = args.variant or None
    fn = run_fit if args.stage == "fit" else run_test
    fn(args.category, variant, batch_users=args.batch_users,
       device=args.device, pred_subdir=args.pred_subdir,
       label_mode=args.label_mode, force=args.force)


if __name__ == "__main__":
    main()
