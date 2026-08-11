"""Wave-0 window recomputes from the existing all-positive prediction dumps.

Two fixes that need NO retraining, both pure evaluation-time restrictions of
the ground truth:

  truncfix   All_Beauty only: restrict the label window to [T2, 2023-03-28) --
             the last 166 days of its native window sit in a collection
             truncation ramp (24.5% of window events missing, survivors
             popularity-biased), so the native numbers are optimistically
             censored (audit finding M1).
  aligned365 all categories: [T2, min(T2+365d, T3)) -- native horizons differ
             1.71x across categories (381-652 days), so cross-category tables
             get a second, horizon-aligned reading instead of a re-cut T3
             (which would disturb the frozen quantile machinery).

A user counts as a GT user for a window iff they have >= 1 positive inside it;
users whose positives all fall outside drop out of that window's denominator.
Native-window numbers are recomputed with the same code path as a self-check:
they must reproduce the published locked values before the restricted numbers
can be trusted.

    python -m scripts.evaluation.horizon_window_eval
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUT_PATH = REPO_ROOT / "results" / "label_modes" / "test_horizon_windows.json"
CATEGORIES = ("All_Beauty", "Video_Games", "Books", "Electronics")
VARIANT = "C2_k500_recent_allpos"
KS = (10, 100)
DAY_MS = 86_400_000
AB_RAMP_START_MS = int(datetime(2023, 3, 28, tzinfo=timezone.utc).timestamp() * 1000)


def eval_window(cat: str, gt: pd.DataFrame, lo: int, hi: int) -> dict:
    """Stream the user-sorted refit dump once and score against the GT
    restricted to [lo, hi). Same mechanics as the migration recompute that
    reproduced the published values bit-for-bit."""
    win = gt[(gt["timestamp"] >= lo) & (gt["timestamp"] < hi)]
    gt_sets = win.groupby("user_id")["parent_asin"].agg(set).to_dict()
    n_gt_users = len(gt_sets)
    n_pos_total = sum(len(v) for v in gt_sets.values())

    path = (PROCESSED_DIR / cat / "temporal_ranker" / "variants" / VARIANT
            / "predictions" / "refit_model_test.parquet")
    pf = pq.ParquetFile(path)

    acc = {f"rec@{k}": 0.0 for k in KS}
    acc.update({f"hit@{k}": 0 for k in KS})
    acc.update({f"prec@{k}": 0.0 for k in KS})
    cov_macro = 0.0

    def flush(uid, pa, lg):
        nonlocal cov_macro
        truth = gt_sets.get(uid)
        if not truth:
            return
        lab = np.fromiter((p in truth for p in pa), dtype=np.int8, count=len(pa))
        n_true = len(truth)
        cov_macro += int(lab.sum()) / n_true
        order = np.lexsort((lab, -lg))       # ties: negatives before positives
        la = lab[order]
        for k in KS:
            h = int(la[:k].sum())
            acc[f"rec@{k}"] += h / n_true
            acc[f"hit@{k}"] += int(h > 0)
            acc[f"prec@{k}"] += h / k

    carry_u, carry_p, carry_l = None, [], []
    for rg in range(pf.metadata.num_row_groups):
        d = pf.read_row_group(rg, columns=["user_id", "parent_asin", "logit"]).to_pandas()
        u = d["user_id"].to_numpy()
        p = d["parent_asin"].to_numpy()
        g = d["logit"].to_numpy()
        bounds = np.flatnonzero(u[1:] != u[:-1]) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(u)]])
        for s, e in zip(starts, ends):
            uid = u[s]
            if carry_u is not None and uid == carry_u:
                carry_p.append(p[s:e]); carry_l.append(g[s:e])
                continue
            if carry_u is not None:
                flush(carry_u, np.concatenate(carry_p), np.concatenate(carry_l))
            carry_u, carry_p, carry_l = uid, [p[s:e]], [g[s:e]]
        del d, u, p, g
    if carry_u is not None:
        flush(carry_u, np.concatenate(carry_p), np.concatenate(carry_l))

    out = {
        "window_days": round((hi - lo) / DAY_MS, 1),
        "n_gt_users": n_gt_users,
        "n_positives": n_pos_total,
        "ceiling_macro": cov_macro / n_gt_users if n_gt_users else 0.0,
    }
    for k in KS:
        out[f"recall@{k}"] = acc[f"rec@{k}"] / n_gt_users if n_gt_users else 0.0
        out[f"hit_rate@{k}"] = acc[f"hit@{k}"] / n_gt_users if n_gt_users else 0.0
        out[f"precision@{k}"] = acc[f"prec@{k}"] / n_gt_users if n_gt_users else 0.0
    return out


def main():
    results = {}
    for cat in CATEGORIES:
        tdir = PROCESSED_DIR / cat / "temporal_ranker"
        cut = json.loads((tdir / "snapshot_manifest.json").read_text())["cutoffs"]
        t2, t3 = int(cut["t2_ms"]), int(cut["t3_ms"])
        gt = pd.read_parquet(tdir / "groundtruth_all_test.parquet",
                             columns=["user_id", "parent_asin", "timestamp"])
        gt["user_id"] = gt["user_id"].astype(str)
        gt["parent_asin"] = gt["parent_asin"].astype(str)

        windows = {"native": (t2, t3),
                   "aligned365": (t2, min(t2 + 365 * DAY_MS, t3))}
        if cat == "All_Beauty":
            windows["truncfix"] = (t2, AB_RAMP_START_MS)

        results[cat] = {}
        for name, (lo, hi) in windows.items():
            r = eval_window(cat, gt, lo, hi)
            results[cat][name] = r
            print(f"  {cat:12s} {name:10s} {r['window_days']:>6.0f}d "
                  f"users={r['n_gt_users']:>7,} pos={r['n_positives']:>7,} "
                  f"R@100={r['recall@100']:.5f} HR@100={r['hit_rate@100']:.4f} "
                  f"ceil={r['ceiling_macro']:.4f}", flush=True)
        del gt

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": VARIANT,
        "note": ("Evaluation-time window restrictions of groundtruth_all_test "
                 "against the frozen all-positive refit dumps. 'native' must "
                 "reproduce the published locked values (self-check). "
                 "'truncfix' is the M1 All_Beauty censoring fix; 'aligned365' "
                 "is the horizon-aligned secondary reading for cross-category "
                 "tables."),
        "ab_ramp_start_utc": "2023-03-28T00:00:00Z",
        "categories": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=1))
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
