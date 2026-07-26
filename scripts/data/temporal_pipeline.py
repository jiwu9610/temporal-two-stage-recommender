"""Global temporal snapshot orchestrator (mode: ``global_temporal_snapshots``).

CLI:
    python -m scripts.data.temporal_pipeline --category Video_Games

Reads the Phase 0 artifacts it depends on (never rewrites them):
    data/processed/{category}/interactions_clean.parquet   base event universe
    data/processed/{category}/item_features.parquet        catalog statics only

and writes the temporal snapshot artifacts to a SEPARATE subdirectory so the
existing per_user_leave_last_two outputs stay untouched:

    data/processed/{category}/temporal/
        val_history.parquet         events with ts <  train_end
        val_labels.parquet          first positive per user in [train_end, val_end)
        test_history.parquet        events with ts <  val_end (includes val window)
        test_labels.parquet         first positive per user in [val_end, test_end)
        val_user_features.parquet   recomputed from val history only
        val_item_features.parquet
        test_user_features.parquet  recomputed from test history only
        test_item_features.parquet
        snapshot_manifest.json      cutoffs + per-snapshot stats + config snapshot

Config: `temporal_snapshots` section of configs/preprocessing.yaml. The old
pipeline (scripts.data.preprocessing_pipeline) is unchanged and both modes can
coexist on disk.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from .temporal_feature_store import (
    CATALOG_STATIC_COLUMNS,
    build_snapshot_item_features,
    build_snapshot_user_features,
)
from .temporal_split import build_temporal_snapshots, resolve_cutoffs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "preprocessing.yaml"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TEMPORAL_OUTPUT_FILES = {
    "val_history.parquet", "val_labels.parquet",
    "test_history.parquet", "test_labels.parquet",
    "val_user_features.parquet", "val_item_features.parquet",
    "test_user_features.parquet", "test_item_features.parquet",
    "snapshot_manifest.json",
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def run_category(
    category: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Dict[str, Any]:
    """Build both snapshots + feature stores for one category."""
    t_start = time.time()
    config = yaml.safe_load(Path(config_path).read_text())
    tcfg = config.get("temporal_snapshots", {})
    quantiles = tcfg.get("cutoff_quantiles", {})
    explicit = (tcfg.get("cutoffs") or {}).get(category)
    eligibility = tcfg.get("eligibility", {})
    min_item_hist = int(eligibility.get("min_item_history_interactions", 5))
    warm_user_min = int(eligibility.get("warm_user_min_history", 3))
    recent_days = int(tcfg.get("features", {}).get("recent_days", 30))

    cat_dir = Path(processed_dir) / category
    clean_path = cat_dir / "interactions_clean.parquet"
    catalog_path = cat_dir / "item_features.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found -- run the Phase 0 pipeline first "
            f"(python -m scripts.data.preprocessing_pipeline --category {category})"
        )
    out_dir = cat_dir / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[temporal] category={category} -> {out_dir}", flush=True)

    # ---- 1. load base universe + catalog statics -----------------------------
    print("[1/4] loading interactions_clean + catalog statics...", flush=True)
    clean = pd.read_parquet(clean_path)
    if catalog_path.exists():
        import pyarrow.parquet as pq
        available = set(pq.read_schema(catalog_path).names)
        catalog_cols = [c for c in CATALOG_STATIC_COLUMNS if c in available]
        catalog = pd.read_parquet(catalog_path, columns=catalog_cols)
    else:
        # Degraded catalog: ids only; statics fall back to defaults.
        catalog = pd.DataFrame({"parent_asin": clean["parent_asin"].unique()})
        print("        WARNING: item_features.parquet missing -- catalog statics "
              "defaulted", flush=True)

    # ---- 2. resolve cutoffs + build snapshots --------------------------------
    print("[2/4] resolving cutoffs + building snapshots...", flush=True)
    cutoffs = resolve_cutoffs(
        clean,
        explicit=explicit,
        train_end_quantile=float(quantiles.get("train_end", 0.80)),
        val_end_quantile=float(quantiles.get("val_end", 0.90)),
    )
    val, test, manifest = build_temporal_snapshots(
        clean, cutoffs,
        min_item_history_interactions=min_item_hist,
        warm_user_min_history=warm_user_min,
    )
    for snap in (val, test):
        print(f"        {snap.name}: history={len(snap.history):,} rows | "
              f"labels={len(snap.labels):,} users "
              f"(warm={snap.stats['labels']['n_warm_users']:,} "
              f"cold={snap.stats['labels']['n_cold_users']:,}) | "
              f"pool={snap.stats['eligibility']['n_eligible_items']:,} items",
              flush=True)

    # ---- 3. snapshot-specific feature stores ---------------------------------
    print("[3/4] snapshot feature stores (independent per snapshot)...", flush=True)
    stores = {}
    for snap in (val, test):
        stores[f"{snap.name}_user_features"] = build_snapshot_user_features(
            snap.history, as_of_ms=snap.history_end_ms, recent_days=recent_days,
        )
        stores[f"{snap.name}_item_features"] = build_snapshot_item_features(
            catalog, snap.history, snap.eligible_items,
            as_of_ms=snap.history_end_ms,
        )

    # ---- 4. write artifacts ---------------------------------------------------
    print("[4/4] writing temporal artifacts...", flush=True)
    val.history.to_parquet(out_dir / "val_history.parquet", index=False)
    val.labels.to_parquet(out_dir / "val_labels.parquet", index=False)
    test.history.to_parquet(out_dir / "test_history.parquet", index=False)
    test.labels.to_parquet(out_dir / "test_labels.parquet", index=False)
    for name, df in stores.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    manifest["config"] = {
        "config_path": str(config_path),
        "temporal_snapshots": tcfg,
        "recent_days": recent_days,
    }
    manifest["started_utc"] = _now_iso()
    manifest["elapsed_seconds"] = round(time.time() - t_start, 2)
    manifest["outputs"] = sorted(TEMPORAL_OUTPUT_FILES)
    with open(out_dir / "snapshot_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[done] {category} temporal snapshots in "
          f"{manifest['elapsed_seconds']}s", flush=True)
    return manifest


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   help="One of: All_Beauty, Video_Games, Books, Electronics")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_category(
        category=args.category,
        config_path=Path(args.config),
        processed_dir=Path(args.processed_dir),
    )


if __name__ == "__main__":
    main()
