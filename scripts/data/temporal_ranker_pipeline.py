"""Three-snapshot walk-forward data layer for the temporal ranker.

CLI:
    python -m scripts.data.temporal_ranker_pipeline --category All_Beauty

Four global cutoffs shared by every user, T0 < T1 < T2 < T3:

    ranker_train      history ts < T0    labels = first positive in [T0, T1)
    model_selection   history ts < T1    labels = first positive in [T1, T2)
    test              history ts < T2    labels = first positive in [T2, T3)

Histories therefore expand: history(T0) subset history(T1) subset history(T2).
Every snapshot gets its OWN feature stores (all `_hist` aggregates recomputed
from that snapshot's history; see temporal_feature_store.py) and its own
ground-truth table. Nothing is shared or copied across snapshots.

Outputs (data/processed/{category}/temporal_ranker/):

    history_ranker_train.parquet          groundtruth_ranker_train.parquet
    history_model_selection.parquet       groundtruth_model_selection.parquet
    history_test.parquet                  groundtruth_test.parquet
    user_features_{snapshot}.parquet      item_features_{snapshot}.parquet
    snapshot_manifest.json

The 2-snapshot baseline (temporal_pipeline.py) and the legacy leave-last-two
pipeline are untouched; all three protocols coexist on disk. By default
T1/T2/T3 equal the 2-snapshot train_end/val_end/test_end (same quantile
basis), so the final-test snapshot here is directly comparable with
results/phase1_temporal.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from .temporal_feature_store import (
    CATALOG_STATIC_COLUMNS,
    build_snapshot_item_features,
    build_snapshot_user_features,
)
from .temporal_split import Snapshot, build_snapshot, iso_to_ms, ms_to_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "preprocessing.yaml"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SNAPSHOT_NAMES = ("ranker_train", "model_selection", "test")

TEMPORAL_RANKER_OUTPUT_FILES = {
    *(f"history_{s}.parquet" for s in SNAPSHOT_NAMES),
    *(f"groundtruth_{s}.parquet" for s in SNAPSHOT_NAMES),
    *(f"user_features_{s}.parquet" for s in SNAPSHOT_NAMES),
    *(f"item_features_{s}.parquet" for s in SNAPSHOT_NAMES),
    "snapshot_manifest.json",
}


def resolve_four_cutoffs(
    clean_df: pd.DataFrame,
    time_col: str = "timestamp",
    explicit: Optional[Dict[str, str]] = None,
    quantiles: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Resolve T0..T3 (epoch ms). Explicit ISO dates win; otherwise GLOBAL
    timestamp quantiles of the full clean set (never per-user). T3 defaults to
    max timestamp + 1ms. Returns a manifest-ready dict."""
    ts = clean_df[time_col]
    max_ms = int(ts.max())
    q = {"t0": 0.70, "t1": 0.80, "t2": 0.90, **(quantiles or {})}
    if explicit and any(explicit.get(k) for k in ("t0", "t1", "t2")) \
            and not all(explicit.get(k) for k in ("t0", "t1", "t2")):
        raise ValueError(
            f"partial explicit cutoffs {explicit} -- specify all of t0/t1/t2 "
            f"(t3 optional) or none; silently mixing explicit dates with "
            f"quantiles is not allowed"
        )
    if explicit and all(explicit.get(k) for k in ("t0", "t1", "t2")):
        cut = {
            "t0": iso_to_ms(explicit["t0"]),
            "t1": iso_to_ms(explicit["t1"]),
            "t2": iso_to_ms(explicit["t2"]),
            "t3": iso_to_ms(explicit["t3"]) if explicit.get("t3") else max_ms + 1,
        }
        source = "explicit"
    else:
        cut = {
            "t0": int(ts.quantile(q["t0"])),
            "t1": int(ts.quantile(q["t1"])),
            "t2": int(ts.quantile(q["t2"])),
            "t3": max_ms + 1,
        }
        source = "quantile"
    if not (cut["t0"] < cut["t1"] < cut["t2"] < cut["t3"]):
        raise ValueError(f"cutoffs must satisfy T0 < T1 < T2 < T3, got {cut}")
    out = {f"{k}_ms": v for k, v in cut.items()}
    out.update({f"{k}_iso": ms_to_iso(v) for k, v in cut.items()})
    out["source"] = source
    out["quantiles"] = q if source == "quantile" else None
    return out


def build_three_snapshots(
    clean_df: pd.DataFrame,
    cutoffs: Dict[str, Any],
    min_item_history_interactions: int = 5,
    warm_user_min_history: int = 3,
) -> Dict[str, Snapshot]:
    """The three walk-forward snapshots. Reuses the audited 2-snapshot builder
    (strict `ts < cutoff` history, first-positive labels, history-only
    eligibility, warm/cold flags)."""
    bounds = {
        "ranker_train": (cutoffs["t0_ms"], cutoffs["t1_ms"]),
        "model_selection": (cutoffs["t1_ms"], cutoffs["t2_ms"]),
        "test": (cutoffs["t2_ms"], cutoffs["t3_ms"]),
    }
    return {
        name: build_snapshot(
            clean_df, name, hist_end, label_end,
            min_item_history_interactions, warm_user_min_history,
        )
        for name, (hist_end, label_end) in bounds.items()
    }


def run_category(
    category: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Dict[str, Any]:
    t_start = time.time()
    config = yaml.safe_load(Path(config_path).read_text())
    tcfg = config.get("temporal_snapshots", {})
    three = tcfg.get("three_snapshot", {})
    eligibility = tcfg.get("eligibility", {})
    min_item_hist = int(eligibility.get("min_item_history_interactions", 5))
    warm_user_min = int(eligibility.get("warm_user_min_history", 3))
    recent_days = int(tcfg.get("features", {}).get("recent_days", 30))
    if recent_days != 30:
        # Downstream consumers (two-tower dense cols, candidate builder,
        # ranker features) hardcode the *_last_30d column names; a different
        # window would silently zero those features. Fail loudly instead.
        raise ValueError(
            f"temporal_snapshots.features.recent_days={recent_days} is not "
            f"supported by the three-snapshot chain yet (consumers hardcode "
            f"*_last_30d columns); use 30 or extend the consumers first."
        )

    cat_dir = Path(processed_dir) / category
    clean_path = cat_dir / "interactions_clean.parquet"
    catalog_path = cat_dir / "item_features.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found -- run the Phase 0 pipeline first"
        )
    out_dir = cat_dir / "temporal_ranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[temporal-ranker] category={category} -> {out_dir}", flush=True)

    print("[1/4] loading interactions_clean + catalog statics...", flush=True)
    clean = pd.read_parquet(clean_path)
    if catalog_path.exists():
        import pyarrow.parquet as pq
        available = set(pq.read_schema(catalog_path).names)
        catalog_cols = [c for c in CATALOG_STATIC_COLUMNS if c in available]
        catalog = pd.read_parquet(catalog_path, columns=catalog_cols)
    else:
        catalog = pd.DataFrame({"parent_asin": clean["parent_asin"].unique()})
        print("        WARNING: item_features.parquet missing -- catalog "
              "statics defaulted", flush=True)

    print("[2/4] resolving T0..T3 + building three snapshots...", flush=True)
    cutoffs = resolve_four_cutoffs(
        clean,
        explicit=(three.get("cutoffs") or {}).get(category),
        quantiles=three.get("cutoff_quantiles"),
    )
    print(f"        T0={cutoffs['t0_iso'][:10]} T1={cutoffs['t1_iso'][:10]} "
          f"T2={cutoffs['t2_iso'][:10]} T3={cutoffs['t3_iso'][:10]} "
          f"({cutoffs['source']})", flush=True)
    snapshots = build_three_snapshots(clean, cutoffs, min_item_hist, warm_user_min)
    for name in SNAPSHOT_NAMES:
        s = snapshots[name]
        print(f"        {name}: history={len(s.history):,} rows | "
              f"labels={len(s.labels):,} users "
              f"(warm={s.stats['labels']['n_warm_users']:,} "
              f"cold={s.stats['labels']['n_cold_users']:,}) | "
              f"pool={s.stats['eligibility']['n_eligible_items']:,}", flush=True)

    # Nesting sanity (guaranteed by construction; asserted for the manifest).
    n_rt, n_ms, n_te = (len(snapshots[n].history) for n in SNAPSHOT_NAMES)
    assert n_rt <= n_ms <= n_te, "histories must expand T0 -> T1 -> T2"

    print("[3/4] snapshot feature stores (independent per snapshot)...", flush=True)
    stores = {}
    for name in SNAPSHOT_NAMES:
        s = snapshots[name]
        stores[f"user_features_{name}"] = build_snapshot_user_features(
            s.history, as_of_ms=s.history_end_ms, recent_days=recent_days,
        )
        stores[f"item_features_{name}"] = build_snapshot_item_features(
            catalog, s.history, s.eligible_items, as_of_ms=s.history_end_ms,
        )

    print("[4/4] writing temporal_ranker artifacts...", flush=True)
    for name in SNAPSHOT_NAMES:
        s = snapshots[name]
        s.history.to_parquet(out_dir / f"history_{name}.parquet", index=False)
        s.labels.to_parquet(out_dir / f"groundtruth_{name}.parquet", index=False)
    for fname, df in stores.items():
        df.to_parquet(out_dir / f"{fname}.parquet", index=False)

    manifest = {
        "strategy": "global_temporal_three_snapshots",
        "cutoffs": cutoffs,
        "snapshot_order": list(SNAPSHOT_NAMES),
        "history_nesting_rows": {
            "ranker_train": n_rt, "model_selection": n_ms, "test": n_te,
        },
        "base_universe": {
            "n_rows": int(len(clean)),
            "n_users": int(clean["user_id"].nunique()),
            "n_items": int(clean["parent_asin"].nunique()),
        },
        "snapshots": {name: snapshots[name].stats for name in SNAPSHOT_NAMES},
        "config": {
            "config_path": str(config_path),
            "three_snapshot": three,
            "eligibility": {
                "min_item_history_interactions": min_item_hist,
                "warm_user_min_history": warm_user_min,
            },
            "recent_days": recent_days,
        },
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "outputs": sorted(TEMPORAL_RANKER_OUTPUT_FILES),
    }
    with open(out_dir / "snapshot_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[done] {category} three-snapshot data layer in "
          f"{manifest['elapsed_seconds']}s", flush=True)
    return manifest


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_category(args.category, Path(args.config), Path(args.processed_dir))


if __name__ == "__main__":
    main()
