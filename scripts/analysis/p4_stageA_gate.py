"""P4.2 Stage A gate: per-config decision table + double gate (spec P4.2).

Reads, per category, the builder reports results/{cat}_candidates_report_{V}.json
(V in A0..A5) plus variants/{V}/candidates_{snapshot}.parquet, and emits
results/p4_stageA/{cat}_gate.json with:

  per config x snapshot (ranker_train, model_selection):
    union ceiling            coverage.retrieved_union (macro) +
                             retrieved_union_per_positive (from the report)
    per-source R@100/R@500   macro over label users, computed here from the
                             candidates parquet ({src}_rank <= K & label == 1;
                             the report's k_sweep stops at K=100)
    content unique-hit share fraction of hit (user,item) label rows reached by
                             content_i2i and nothing else
    hit-by-item-history      union hits bucketed by the target's history count
                             in that snapshot (0 / 1-2 / 3-4 / 5+)
    n_rows                   builder report

  gate (per config, rows summed over the two snapshots):
    rows_ratio_vs_A0 <= 1.6          (A0 == post-C1 C2 analogue)
    projected Stage-B MaxRSS <= 120G (rows x bytes/row anchor x 1.10 margin
                                      for the three content feature columns)

Anchors: allpos (2026-08-05/06) batch MaxRSS / the rows of the candidates the
ranker actually consumed -- variants/C2_k500_recent_allpos (NOT the base
top_k=100 candidates; dividing by those overstates bytes/row ~7x): VG 1099 /
Books 1231 / Elec 1103 / AB 1112 B/row. VG measured 126.9 GiB of 128 at
123.9M rows -- the D8 memory wall is real. The content margin is 1.03: the
three content columns add ~36B on ~1.1KB/row through the feature pipeline.

Shortlist policy (spec): rank configs by model_selection union ceiling (macro)
among gate-passing configs; report top-2 plus A0 for reference. Selection
NEVER reads the test snapshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
RESULTS = REPO / "results"
REPORTS = RESULTS / "phase2_temporal"     # builder RESULTS_DIR

CONFIGS = ["A0", "A1", "A2", "A3", "A4", "A5"]
VARIANT_SUFFIX = "_ap"            # 2026-08-21: all-positive rebuild variants
GT_PREFIX = "groundtruth_all_"    # the project objective frame
SNAPSHOTS = ["ranker_train", "model_selection"]
BYTES_PER_ROW = {"All_Beauty": 1112, "Video_Games": 1099,
                 "Books": 1231, "Electronics": 1103}
CONTENT_MARGIN = 1.03            # content configs only: +3 feature cols
CONTENT_CONFIGS = {"A1", "A2", "A3", "A4"}
ROWS_GATE = 1.6
MAXRSS_GATE_GIB = 120.0
BUCKETS = [(0, 0, "0"), (1, 2, "1-2"), (3, 4, "3-4"), (5, 10**12, "5+")]


def _bucket(n: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= n <= hi:
            return name
    return "5+"


def analyze_snapshot(cat: str, variant: str, snapshot: str) -> dict:
    tdir = PROCESSED / cat / "temporal_ranker"
    cdir = tdir / "variants" / (variant + VARIANT_SUFFIX)
    cand = pd.read_parquet(cdir / f"candidates_{snapshot}.parquet")
    gt = pd.read_parquet(tdir / f"{GT_PREFIX}{snapshot}.parquet",
                         columns=["user_id", "parent_asin"])
    hist = pd.read_parquet(tdir / f"history_{snapshot}.parquet",
                           columns=["parent_asin"])
    hist_counts = hist["parent_asin"].astype(str).value_counts()

    for c in ("user_id", "parent_asin"):
        cand[c] = cand[c].astype(str)
        gt[c] = gt[c].astype(str)

    label_users = sorted(gt["user_id"].unique())
    n_users = len(label_users)
    targets_per_user = gt.groupby("user_id")["parent_asin"].nunique()

    hits = cand[cand["label"] == 1]
    src_cols = [c for c in cand.columns
                if c.startswith("source_") and c != "source_count"]
    sources = [c[len("source_"):] for c in src_cols]

    # per-source R@K macro: mean over label users of (targets hit by that
    # source within rank<=K) / (targets of that user)
    per_source = {}
    for s in sources:
        rcol = f"{s}_rank"
        out = {}
        for K in (100, 500):
            sh = hits[(hits[f"source_{s}"] == 1) & (hits[rcol] > 0)
                      & (hits[rcol] <= K)]
            got = sh.groupby("user_id")["parent_asin"].nunique()
            macro = float((got.reindex(label_users).fillna(0)
                           / targets_per_user.reindex(label_users)).mean())
            out[f"R@{K}"] = round(macro, 6)
        per_source[s] = out

    # content unique-hit share over hit label rows
    unique_share = None
    if "source_content_i2i" in cand.columns:
        n_hit_rows = len(hits)
        only = hits[(hits["source_content_i2i"] == 1)
                    & (hits["num_sources"] == 1)]
        unique_share = float(len(only) / n_hit_rows) if n_hit_rows else 0.0

    # union hits bucketed by target item history support
    hit_pairs = hits[["user_id", "parent_asin"]].drop_duplicates()
    b_counts = {name: 0 for _, _, name in BUCKETS}
    for pa in hit_pairs["parent_asin"]:
        b_counts[_bucket(int(hist_counts.get(pa, 0)))] += 1
    gt_b = {name: 0 for _, _, name in BUCKETS}
    for pa in gt["parent_asin"]:
        gt_b[_bucket(int(hist_counts.get(pa, 0)))] += 1

    return {
        "n_label_users": n_users,
        "n_rows_parquet": int(len(cand)),
        "per_source": per_source,
        "content_unique_hit_share": unique_share,
        "union_hits_by_item_history": b_counts,
        "gt_targets_by_item_history": gt_b,
    }


def run(cat: str, skip_parquet: bool = False) -> dict:
    per_cfg = {}
    for v in CONFIGS:
        rp = REPORTS / f"{cat}_candidates_report_{v}{VARIANT_SUFFIX}.json"
        if not rp.exists():
            print(f"[gate] {cat} {v}: report missing, skip")
            continue
        rep = json.loads(rp.read_text())
        snaps = {}
        rows_total = 0
        for s in SNAPSHOTS:
            sr = rep["snapshots"][s]
            rows_total += int(sr["n_rows"])
            snaps[s] = {
                "union_macro": sr["coverage"]["retrieved_union"],
                "union_per_positive":
                    sr["coverage"].get("retrieved_union_per_positive"),
                "n_rows": sr["n_rows"],
                "elapsed_seconds": sr.get("elapsed_seconds"),
            }
            if not skip_parquet:
                snaps[s].update(analyze_snapshot(cat, v, s))
        margin = CONTENT_MARGIN if v in CONTENT_CONFIGS else 1.0
        proj_gib = rows_total * BYTES_PER_ROW[cat] * margin / 2**30
        per_cfg[v] = {"snapshots": snaps, "rows_total": rows_total,
                      "projected_stageB_maxrss_gib": round(proj_gib, 1)}

    if "A0" in per_cfg:
        base = per_cfg["A0"]["rows_total"]
        for v, d in per_cfg.items():
            ratio = d["rows_total"] / base if base else float("inf")
            d["rows_ratio_vs_A0"] = round(ratio, 3)
            d["gate_rows"] = ratio <= ROWS_GATE
            d["gate_maxrss"] = (d["projected_stageB_maxrss_gib"]
                                <= MAXRSS_GATE_GIB)
            d["gate_pass"] = d["gate_rows"] and d["gate_maxrss"]

    passing = [v for v, d in per_cfg.items() if d.get("gate_pass")]
    ranked = sorted(
        passing,
        key=lambda v: per_cfg[v]["snapshots"]["model_selection"]["union_macro"],
        reverse=True)
    payload = {"category": cat, "configs": per_cfg,
               "gate": {"rows_ratio_max": ROWS_GATE,
                        "maxrss_gib_max": MAXRSS_GATE_GIB,
                        "bytes_per_row_anchor": BYTES_PER_ROW[cat],
                        "content_margin": CONTENT_MARGIN},
               "shortlist_top2": ranked[:2],
               "ranked_passing": ranked}
    out = RESULTS / "p4_stageA_ap"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cat}_gate.json").write_text(json.dumps(payload, indent=2))

    print(f"\n=== {cat} Stage A gate ===")
    hdr = (f"{'cfg':>4} {'union_ms':>9} {'perpos':>7} {'rows(rt+ms)':>12} "
           f"{'xA0':>6} {'projGiB':>8} {'gate':>5}")
    print(hdr)
    for v in CONFIGS:
        if v not in per_cfg:
            continue
        d = per_cfg[v]
        ms = d["snapshots"]["model_selection"]
        pp = ms.get("union_per_positive")
        print(f"{v:>4} {ms['union_macro']:>9.4f} "
              f"{(pp if pp is not None else float('nan')):>7.4f} "
              f"{d['rows_total']:>12,} {d.get('rows_ratio_vs_A0', 0):>6.2f} "
              f"{d['projected_stageB_maxrss_gib']:>8.1f} "
              f"{'PASS' if d.get('gate_pass') else 'FAIL':>5}")
    print(f"shortlist(top-2 by model_selection union macro): {ranked[:2]}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--category", required=True)
    ap.add_argument("--skip-parquet", action="store_true",
                    help="report-only pass (no per-source/bucket recompute)")
    args = ap.parse_args()
    run(args.category, skip_parquet=args.skip_parquet)


if __name__ == "__main__":
    main()
