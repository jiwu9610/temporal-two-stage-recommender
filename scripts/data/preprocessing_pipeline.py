"""Phase 0 preprocessing orchestrator.

CLI:
    python -m scripts.data.preprocessing_pipeline --category Video_Games --profile default
    python -m scripts.data.preprocessing_pipeline --category Video_Games --profile strict

Runs the modular Phase 0 pipeline for one category:

    canonicalize  ->  filtering  ->  splitting  ->  feature_store  ->  text_alignment

and writes Phase 0 artifacts under ``data/processed/{category}/``. The exact set
is the union of ``PHASE0_OUTPUT_FILES`` defined below; ``configs/preprocessing.yaml``
documents each artifact's contents and schema.

The orchestrator is intentionally thin: each step is a small function call into
its module. All decisions (label semantics, k-core thresholds, split strategy,
feature-store schema) live in those modules + ``configs/preprocessing.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from .canonicalize import canonicalize
from .feature_store import build_item_features, build_user_features
from .filtering import iterative_kcore
from .loader import load_metadata, load_sampled
from .splitting import leave_last_two_split
from .text_alignment import align_text_embeddings


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "preprocessing.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Files this orchestrator owns. --clean-phase0 will remove these before a fresh run.
PHASE0_OUTPUT_FILES = {
    "canonical_item_map.parquet",
    "interactions_clean.parquet",
    "interactions_filtered.parquet",
    "train.parquet", "val.parquet", "test.parquet",
    "user_features.parquet", "item_features.parquet",
    "item_text_embeddings.npz",
    "filtering_report.json", "split_manifest.json",
    "text_alignment_report.json", "pipeline_run.json",
}

# Stale legacy artifacts left behind by the deprecated pipeline. --clean-phase0
# also removes these so they don't confuse downstream consumers.
LEGACY_LEFTOVERS = {
    "missing_values_metadata.csv",
    "missing_values_reviews.csv",
    "pipeline_stats.json",
    "user_histories.json",
}

# Anything else in out_dir is preserved (most importantly title_bert.npz, which
# is the *input* to text_alignment.py).


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _clean_phase0_outputs(out_dir: Path) -> list[str]:
    """Remove Phase 0 artifacts + known legacy leftovers, preserve everything else."""
    if not out_dir.exists():
        return []
    targets = PHASE0_OUTPUT_FILES | LEGACY_LEFTOVERS
    removed = []
    for name in targets:
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
    return sorted(removed)


def run_category(
    category: str,
    profile: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    bert_source_path: Path | None = None,
    clean_phase0: bool = False,
) -> Dict[str, Any]:
    """Run the Phase 0 pipeline for one category. Returns the run summary dict."""
    t_start = time.time()
    config = yaml.safe_load(config_path.read_text())
    if profile not in config["profiles"]:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(config['profiles'])}")

    pos_threshold = float(config["label"]["positive_threshold"])
    profile_cfg = config["profiles"][profile]
    out_dir = Path(config["output"]["base_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir = out_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean_phase0:
        removed = _clean_phase0_outputs(out_dir)
        if removed:
            print(f"[clean] removed {len(removed)} stale artifact(s): {removed}", flush=True)

    print(f"[orchestrator] category={category} profile={profile} -> {out_dir}", flush=True)

    # ---- 1. load raw inputs ---------------------------------------------------
    print("[1/6] loading raw reviews + metadata...", flush=True)
    reviews = load_sampled(category, data_dir=data_dir)
    meta = load_metadata(category, data_dir=data_dir)

    # ---- 2. canonicalize ------------------------------------------------------
    print("[2/6] canonicalize: dedup metadata, remap parent_asin, label rows...", flush=True)
    interactions_clean, canonical_map, canonical_meta, canon_stats = canonicalize(
        reviews_df=reviews,
        meta_df=meta,
        positive_threshold=pos_threshold,
    )
    interactions_clean.to_parquet(out_dir / "interactions_clean.parquet", index=False)
    canonical_map.to_parquet(out_dir / "canonical_item_map.parquet", index=False)
    print(f"        clean interactions: {len(interactions_clean):,}  "
          f"unique users: {canon_stats.n_unique_users:,}  "
          f"unique items: {canon_stats.n_unique_items:,}", flush=True)

    # ---- 3. iterative k-core filter ------------------------------------------
    print("[3/6] iterative k-core filter...", flush=True)
    fcfg = profile_cfg["filter"]
    filtered, filter_report = iterative_kcore(
        interactions_clean,
        min_user_interactions=int(fcfg["min_user_interactions"]),
        min_item_interactions=int(fcfg["min_item_interactions"]),
        filter_order=tuple(fcfg.get("filter_order", ("item", "user"))),
        max_iterations=int(fcfg.get("max_iterations", 50)),
    )
    filtered.to_parquet(out_dir / "interactions_filtered.parquet", index=False)
    _save_json(filter_report.to_dict(), out_dir / "filtering_report.json")
    print(f"        converged in {filter_report.n_iterations} iter(s)  "
          f"-> {filter_report.final}", flush=True)

    # ---- 4. per-user leave-last-two split ------------------------------------
    print("[4/6] per-user leave-last-two split...", flush=True)
    train, val, test, manifest = leave_last_two_split(filtered)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    _save_json(manifest.to_dict(), out_dir / "split_manifest.json")
    print(f"        train={len(train):,}  val={len(val):,}  test={len(test):,}  "
          f"users={manifest.n_users_total:,}", flush=True)
    print(f"        train-catalog covers val-pos {manifest.coverage['val_positive_items_in_train_catalog_rate']:.4f} | "
          f"test-pos {manifest.coverage['test_positive_items_in_train_catalog_rate']:.4f}", flush=True)

    # ---- 5. feature stores (TRAIN ONLY) --------------------------------------
    print("[5/6] feature stores (train-only)...", flush=True)
    user_features = build_user_features(train)
    item_features = build_item_features(canonical_meta, train, filtered_df=filtered)
    user_features.to_parquet(out_dir / "user_features.parquet", index=False)
    item_features.to_parquet(out_dir / "item_features.parquet", index=False)
    print(f"        user_features: {user_features.shape[0]:,} x {user_features.shape[1]} cols", flush=True)
    print(f"        item_features: {item_features.shape[0]:,} x {item_features.shape[1]} cols", flush=True)

    # ---- 6. text embedding alignment -----------------------------------------
    print("[6/6] text embedding alignment...", flush=True)
    if bert_source_path is None:
        # Default convention: legacy file lives next to outputs as title_bert.npz.
        candidate = out_dir / "title_bert.npz"
        bert_source_path = candidate if candidate.exists() else None
    text_report = align_text_embeddings(
        source_npz_path=bert_source_path,
        item_features=item_features,
        canonical_item_map=canonical_map,
        output_npz_path=out_dir / "item_text_embeddings.npz",
    )
    _save_json(text_report.to_dict(), out_dir / "text_alignment_report.json")
    print(f"        status={text_report.status} aligned={text_report.n_aligned_to_item_features}/"
          f"{text_report.n_items_in_features}", flush=True)

    # ---- pipeline run summary -------------------------------------------------
    summary = {
        "category": category,
        "profile": profile,
        "config_path": str(config_path),
        "config_snapshot": config,
        "started_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "canonicalize": canon_stats.to_dict(),
        "filtering": filter_report.to_dict(),
        "split": manifest.to_dict(),
        "text_alignment": text_report.to_dict(),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    _save_json(summary, out_dir / "pipeline_run.json")
    print(f"[done] {category} ({profile}) in {summary['elapsed_seconds']}s", flush=True)
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   help="One of: All_Beauty, Video_Games, Books, Electronics")
    p.add_argument("--profile", default="default", choices=["default", "strict"])
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                   help="Path to preprocessing.yaml")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="Repo data root (raw + processed under here)")
    p.add_argument("--bert-source",
                   default=None,
                   help="Optional path to legacy title_bert.npz; defaults to "
                        "data/processed/{category}/title_bert.npz if present")
    p.add_argument("--clean-phase0", action="store_true",
                   help="Before running, delete this category's Phase 0 outputs and known "
                        "legacy leftovers (missing_values_*.csv, pipeline_stats.json, "
                        "user_histories.json). Preserves title_bert.npz (the input).")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_category(
        category=args.category,
        profile=args.profile,
        config_path=Path(args.config),
        data_dir=Path(args.data_dir),
        bert_source_path=Path(args.bert_source) if args.bert_source else None,
        clean_phase0=args.clean_phase0,
    )


if __name__ == "__main__":
    main()
