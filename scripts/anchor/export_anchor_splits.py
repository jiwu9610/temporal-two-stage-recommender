"""Export anchor (official-protocol) splits from the raw review slices.

CLI:
    python -m scripts.anchor.export_anchor_splits --category All_Beauty
    python -m scripts.anchor.export_anchor_splits --category all

Deliberately bypasses canonicalize.py: anchor identity is the
*raw* parent_asin, exactly as in the official Amazon-Reviews-2023 benchmark
(hyp1231/AmazonReviews2023 benchmark_scripts). The protocol is:

    1. dedup (user_id, parent_asin) keep-earliest       [official protocol]
    2. iterative 5/5-core                               [reuses scripts.data.filtering.iterative_kcore]
    3. per-user leave-last-two (latest=test,
       second-latest=valid)                             [reuses scripts.data.splitting.leave_last_two_split]

Outputs under ``data/anchor/{category}/``:

    {category}.inter        RecBole atomic file of the FULL filtered set, rows
                            pre-sorted by (user_id, timestamp, parent_asin) so a
                            stable timestamp sort (RecBole ``order: TO``)
                            reproduces our deterministic tie-break.
    train/val/test.parquet  leave_last_two_split output, phase1-compatible schema.
    user_features.parquet   build_user_features(train)  (phase1 layout, so the
    item_features.parquet   two-tower leg can monkeypatch PROCESSED_DIR here).
    split_manifest.json     SplitManifest of the LOO split.
    anchor_manifest.json    commit hash, dedup/k-core stats, timestamp-tie
                            rates, gate results, isolation declaration.

Gates (hard asserts, script aborts on failure):
    * .inter-derived LOO (valid, test) pairs == leave_last_two_split parquet,
      per user, exact.
    * VG / Books post-k-core stats == the review-recomputed dedup-first values
      (protocol note). AB / Electronics are recorded, not
      asserted (spec: filled from the actual pipeline run).

ISOLATION: per-user LOO ignores global time, so anchor train rows fall inside
the production locked test window. Nothing under data/anchor/ or results/anchor/
may ever be read by a production candidate / ranker / calibration pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from scripts.data.feature_store import build_item_features, build_user_features
from scripts.data.filtering import iterative_kcore
from scripts.data.splitting import leave_last_two_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "anchor"
CONFIG_PATH = REPO_ROOT / "configs" / "preprocessing.yaml"

CATEGORIES = ["All_Beauty", "Books", "Electronics", "Video_Games"]

MIN_USER_CORE = 5
MIN_ITEM_CORE = 5

# Independently recomputed dedup-first 5/5-core expectations:
# (n_interactions, n_users, n_items). Only VG and Books were
# independently recomputed; AB / Electronics are recorded from this run.
EXPECTED_KCORE_STATS = {
    "Video_Games": (263_097, 28_621, 11_909),
    "Books": (145_286, 12_390, 14_059),
}

REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp",
                  "verified_purchase", "helpful_vote"]
CRITICAL_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]

META_COLUMNS = ["parent_asin", "main_category", "store", "price",
                "average_rating", "rating_number", "features", "description",
                "categories", "bought_together"]

ISOLATION_DECLARATION = (
    "ANCHOR ISOLATION: this split follows the official per-user leave-one-out "
    "protocol, which ignores the global temporal cutoffs of the production "
    "pipeline. Anchor training interactions fall inside the production locked "
    "test window [t2, t3). Every artifact under data/anchor/ and results/anchor/ "
    "(including any anchor two-tower checkpoint) is for protocol/pipeline "
    "health-checking only and MUST NOT enter a production candidate, ranker, or "
    "calibration pipeline. This is isolation, not an exemption: anchor produces "
    "no production metric."
)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    return {
        "commit_hash": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(run("status", "--porcelain")),
    }


def _file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()}


def _positive_threshold() -> float:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return float(config["label"]["positive_threshold"])


def dedup_keep_earliest(reviews: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Official-protocol dedup: one row per (user, item), earliest timestamp.

    Deterministic under timestamp ties: rows are mergesorted by
    (user_id, parent_asin, timestamp, rating) before keep-first, so equal-ts
    duplicates resolve to the lowest rating (arbitrary but stable choice).
    """
    n_raw = len(reviews)
    rev = reviews.dropna(subset=CRITICAL_COLUMNS)
    n_after_dropna = len(rev)

    rev = rev.sort_values(
        ["user_id", "parent_asin", "timestamp", "rating"], kind="mergesort"
    )
    rev = rev.drop_duplicates(subset=["user_id", "parent_asin"], keep="first")
    rev = rev.reset_index(drop=True)

    stats = {
        "n_raw_rows": int(n_raw),
        "n_after_dropna": int(n_after_dropna),
        "n_dropped_null_critical": int(n_raw - n_after_dropna),
        "n_after_dedup": int(len(rev)),
        "n_duplicates_removed": int(n_after_dropna - len(rev)),
        "n_users_after_dedup": int(rev["user_id"].nunique()),
        "n_items_after_dedup": int(rev["parent_asin"].nunique()),
    }
    return rev, stats


def timestamp_tie_rates(filtered: pd.DataFrame) -> dict:
    """Per-user timestamp-tie rates on the post-k-core set.

    any_ts_tie_user_rate      user has >=1 duplicated timestamp among own rows.
    boundary_ts_tie_user_rate the (train|valid) or (valid|test) boundary row
                              shares a timestamp with its neighbor, i.e. the
                              val/test assignment depends on the lexicographic
                              parent_asin tie-break.
    """
    d = filtered.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")
    n_users = int(d["user_id"].nunique())

    same_user = d["user_id"].eq(d["user_id"].shift())
    same_ts = d["timestamp"].eq(d["timestamp"].shift())
    users_any_tie = int(d.loc[same_user & same_ts, "user_id"].nunique())

    rev_rank = d.groupby("user_id").cumcount(ascending=False)
    tail3 = d[rev_rank <= 2].copy()
    tail3["_rr"] = rev_rank[rev_rank <= 2]
    wide = tail3.pivot(index="user_id", columns="_rr", values="timestamp")
    # 5-core guarantees >= 5 rows/user, so columns 0,1,2 are all populated.
    assert not wide[[0, 1, 2]].isna().any().any(), "user with <3 rows survived k-core"
    boundary = (wide[0] == wide[1]) | (wide[1] == wide[2])
    users_boundary_tie = int(boundary.sum())

    return {
        "n_users": n_users,
        "n_users_any_ts_tie": users_any_tie,
        "any_ts_tie_user_rate": users_any_tie / n_users,
        "n_users_boundary_ts_tie": users_boundary_tie,
        "boundary_ts_tie_user_rate": users_boundary_tie / n_users,
    }


def write_inter(filtered: pd.DataFrame, path: Path) -> int:
    """Write the FULL filtered set as one RecBole atomic .inter file.

    Rows are pre-sorted by (user_id, timestamp, parent_asin) — the exact
    ordering leave_last_two_split uses — so any *stable* timestamp sort
    (RecBole ``order: TO``) reproduces our deterministic LOO assignment.
    Timestamps are written as integer ms (exact under float64 parsing).
    """
    for col in ["user_id", "parent_asin"]:
        bad = filtered[col].str.contains("\t|\n|\r", regex=True)
        assert not bad.any(), f"{col} contains tab/newline; .inter would be corrupt"

    d = filtered.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")
    out = pd.DataFrame({
        "user_id:token": d["user_id"].to_numpy(),
        "item_id:token": d["parent_asin"].to_numpy(),
        "rating:float": d["rating"].astype(float).to_numpy(),
        "timestamp:float": d["timestamp"].astype("int64").to_numpy(),
    })
    out.to_csv(path, sep="\t", index=False)
    return len(out)


def load_meta_for_items(category: str, items: set, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Stream raw metadata row-group batches, keeping only rows for `items`.

    Books metadata is 4.4M rows; streaming keeps the login-node footprint
    bounded. Dedup mirrors canonicalize.py step 1 (keep highest rating_number)
    but does NOT collapse titles — anchor identity is the raw parent_asin.
    Items with no metadata row at all are appended as bare parent_asin rows so
    item_features covers the full filtered universe (build_item_features fills
    Unknown/median defaults and sets missing_flag=1).
    """
    path = data_dir / "raw" / category / "metadata.parquet"
    pf = pq.ParquetFile(path)
    available = [c.name for c in pf.schema_arrow]
    cols = [c for c in META_COLUMNS if c in available]
    value_set = pa.array(sorted(items))

    chunks = []
    for batch in pf.iter_batches(batch_size=65536, columns=cols):
        mask = pc.is_in(batch.column("parent_asin"), value_set=value_set)
        kept = pa.Table.from_batches([batch]).filter(mask)
        if len(kept):
            chunks.append(kept.to_pandas())
    meta = (pd.concat(chunks, ignore_index=True)
            if chunks else pd.DataFrame(columns=cols))

    n_meta_rows_matched = len(meta)
    if "rating_number" in meta.columns:
        meta = meta.sort_values("rating_number", ascending=False, kind="mergesort")
    meta = meta.drop_duplicates(subset=["parent_asin"], keep="first").reset_index(drop=True)

    missing = sorted(items - set(meta["parent_asin"]))
    if missing:
        meta = pd.concat(
            [meta, pd.DataFrame({"parent_asin": missing})], ignore_index=True
        )

    stats = {
        "n_meta_rows_matched": int(n_meta_rows_matched),
        "n_meta_items": int(len(meta)),
        "n_items_missing_metadata": len(missing),
        "meta_columns_used": cols,
    }
    return meta, stats


def derive_loo_from_inter(inter_path: Path) -> tuple[pd.Series, pd.Series]:
    """Re-read the .inter and derive (valid, test) per user the RecBole way.

    Stable sort by (user, timestamp) ONLY — timestamp ties keep file order,
    which is how a stable ``order: TO`` sort behaves. Returns two Series
    (index=user_id): valid item, test item.
    """
    df = pd.read_csv(
        inter_path, sep="\t",
        dtype={"user_id:token": str, "item_id:token": str,
               "rating:float": float, "timestamp:float": float},
    )
    df = df.rename(columns={"user_id:token": "user_id", "item_id:token": "item_id",
                            "timestamp:float": "timestamp"})
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    rev_rank = df.groupby("user_id").cumcount(ascending=False)
    test = df[rev_rank == 0].set_index("user_id")["item_id"]
    valid = df[rev_rank == 1].set_index("user_id")["item_id"]
    return valid, test


def gate_inter_matches_parquet(out_dir: Path, category: str) -> dict:
    """HARD GATE: .inter-derived LOO == leave_last_two_split parquet, per user."""
    valid_i, test_i = derive_loo_from_inter(out_dir / f"{category}.inter")
    val_pq = pd.read_parquet(out_dir / "val.parquet", columns=["user_id", "parent_asin"])
    test_pq = pd.read_parquet(out_dir / "test.parquet", columns=["user_id", "parent_asin"])

    assert val_pq["user_id"].is_unique and test_pq["user_id"].is_unique, \
        "val/test parquet must have exactly one row per user"
    val_p = val_pq.set_index("user_id")["parent_asin"]
    test_p = test_pq.set_index("user_id")["parent_asin"]

    assert set(valid_i.index) == set(val_p.index), (
        f"user set mismatch: .inter LOO {len(valid_i)} users vs "
        f"val.parquet {len(val_p)} users"
    )
    for name, inter_side, pq_side in [("valid", valid_i, val_p), ("test", test_i, test_p)]:
        joined = inter_side.rename("inter").to_frame().join(pq_side.rename("parquet"))
        mism = joined[joined["inter"] != joined["parquet"]]
        assert mism.empty, (
            f"{name} LOO mismatch for {len(mism)} users; first 5:\n{mism.head()}"
        )
    return {
        "gate": "inter_loo_matches_leave_last_two_parquet",
        "passed": True,
        "n_users_compared": int(len(val_p)),
    }


def export_category(category: str, data_dir: Path, out_root: Path) -> dict:
    t0 = time.time()
    out_dir = out_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = data_dir / "raw" / category / "reviews.parquet"
    print(f"[anchor:{category}] loading {reviews_path}", flush=True)
    reviews = pd.read_parquet(reviews_path, columns=REVIEW_COLUMNS)

    # ---- 1. dedup (user,item) keep-earliest (official protocol) --------------
    deduped, dedup_stats = dedup_keep_earliest(reviews)
    del reviews
    print(f"[anchor:{category}] dedup: {dedup_stats['n_after_dropna']:,} -> "
          f"{dedup_stats['n_after_dedup']:,} "
          f"(-{dedup_stats['n_duplicates_removed']:,} dup, "
          f"-{dedup_stats['n_dropped_null_critical']:,} null)", flush=True)

    # Label columns: not part of the official protocol (which ignores rating),
    # but leave_last_two_split's summaries and the feature builders expect them.
    pos_threshold = _positive_threshold()
    deduped["label"] = (deduped["rating"] >= pos_threshold).astype(np.int8)
    deduped["label_type"] = np.where(deduped["label"] == 1, "positive", "hard_negative")

    # ---- 2. iterative 5/5-core ----------------------------------------------
    filtered, filter_report = iterative_kcore(
        deduped,
        min_user_interactions=MIN_USER_CORE,
        min_item_interactions=MIN_ITEM_CORE,
    )
    del deduped
    final = filter_report.final
    kcore_measured = (final["n_interactions"], final["n_users"], final["n_items"])
    print(f"[anchor:{category}] 5/5-core in {filter_report.n_iterations} iters: "
          f"{kcore_measured[0]:,} inter / {kcore_measured[1]:,} users / "
          f"{kcore_measured[2]:,} items", flush=True)
    assert filter_report.converged, f"k-core did not converge for {category}"

    # HARD GATE: reviewer-recomputed dedup-first expectations (VG, Books).
    expected = EXPECTED_KCORE_STATS.get(category)
    if expected is not None:
        assert kcore_measured == expected, (
            f"{category} dedup-first 5/5-core stats {kcore_measured} != "
            f"spec expectation {expected} (inter/users/items). ABORT — the "
            f"slice or protocol drifted from the reviewed computation."
        )
        print(f"[anchor:{category}] GATE PASS: k-core stats == spec expectation", flush=True)

    tie_stats = timestamp_tie_rates(filtered)
    print(f"[anchor:{category}] ts ties: any={tie_stats['any_ts_tie_user_rate']:.4f} "
          f"boundary={tie_stats['boundary_ts_tie_user_rate']:.4f} of "
          f"{tie_stats['n_users']:,} users", flush=True)

    # ---- 3. per-user leave-last-two ------------------------------------------
    train, val, test, split_manifest = leave_last_two_split(filtered)
    assert len(val) == len(test) == final["n_users"], (
        f"LOO row counts (val={len(val)}, test={len(test)}) != n_users "
        f"{final['n_users']}"
    )
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest.to_dict(), f, indent=2, default=str)
    print(f"[anchor:{category}] split: train={len(train):,} val={len(val):,} "
          f"test={len(test):,}", flush=True)

    # ---- 4. RecBole .inter ----------------------------------------------------
    inter_path = out_dir / f"{category}.inter"
    n_inter_rows = write_inter(filtered, inter_path)
    assert n_inter_rows == final["n_interactions"]

    # ---- 5. feature parquets (phase1 layout for the two-tower leg) -----------
    # NOTE: item_features carries lifetime
    # catalog aggregates (average_rating / rating_number) — under per-user LOO
    # that is a future-information channel inherited from phase1.
    meta, meta_stats = load_meta_for_items(
        category, set(filtered["parent_asin"].unique()), data_dir
    )
    user_features = build_user_features(train)
    item_features = build_item_features(meta, train, filtered_df=filtered)
    assert len(item_features) == final["n_items"], (
        f"item_features rows {len(item_features)} != filtered universe "
        f"{final['n_items']}"
    )
    user_features.to_parquet(out_dir / "user_features.parquet", index=False)
    item_features.to_parquet(out_dir / "item_features.parquet", index=False)
    print(f"[anchor:{category}] features: users={len(user_features):,} "
          f"items={len(item_features):,} "
          f"(missing metadata: {meta_stats['n_items_missing_metadata']})", flush=True)

    # ---- 6. HARD GATE: .inter-derived LOO == parquet -------------------------
    gate = gate_inter_matches_parquet(out_dir, category)
    print(f"[anchor:{category}] GATE PASS: .inter LOO == leave_last_two parquet "
          f"({gate['n_users_compared']:,} users)", flush=True)

    # ---- 7. manifest ----------------------------------------------------------
    manifest = {
        "category": category,
        "created_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "git": _git_state(),
        "isolation_declaration": ISOLATION_DECLARATION,
        "protocol": {
            "identity": "raw parent_asin (canonicalize.py bypassed on purpose)",
            "dedup": "(user_id, parent_asin) keep-earliest, mergesort tie-break "
                     "(user, item, timestamp, rating)",
            "kcore": f"iterative {MIN_USER_CORE}/{MIN_ITEM_CORE}-core "
                     "(scripts.data.filtering.iterative_kcore)",
            "split": "per-user leave-last-two, stable (user, ts, parent_asin) "
                     "ordering (scripts.data.splitting.leave_last_two_split)",
            "positive_threshold": pos_threshold,
            "label_note": "label/label_type are carried for feature builders "
                          "only; the anchor protocol itself ignores rating.",
        },
        "inputs": {
            "reviews": _file_fingerprint(data_dir / "raw" / category / "reviews.parquet"),
            "metadata": _file_fingerprint(data_dir / "raw" / category / "metadata.parquet"),
        },
        "dedup": dedup_stats,
        "kcore": filter_report.to_dict(),
        "kcore_final_stats": {
            "n_interactions": final["n_interactions"],
            "n_users": final["n_users"],
            "n_items": final["n_items"],
            "spec_expectation": list(expected) if expected else None,
            "spec_expectation_checked": expected is not None,
        },
        "timestamp_ties": tie_stats,
        "metadata_coverage": meta_stats,
        "gates": [
            gate,
            {
                "gate": "kcore_stats_match_spec_expectation",
                "passed": True,
                "checked": expected is not None,
                "measured_inter_users_items": list(kcore_measured),
            },
        ],
        "split_manifest": split_manifest.to_dict(),
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    with open(out_dir / "anchor_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[anchor:{category}] done in {manifest['elapsed_seconds']}s -> {out_dir}",
          flush=True)
    return manifest


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   help=f"One of {CATEGORIES} or 'all'")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT),
                   help="Anchor output root (default data/anchor)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cats = CATEGORIES if args.category == "all" else [args.category]
    for cat in cats:
        if cat not in CATEGORIES:
            print(f"unknown category {cat!r}; choose from {CATEGORIES}", file=sys.stderr)
            return 2
    for cat in cats:
        export_category(cat, Path(args.data_dir), Path(args.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
