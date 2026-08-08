"""Emit groundtruth_all_{snapshot}.parquet for existing temporal_ranker snapshots.

The three-snapshot pipeline now writes these itself, but the artifacts on disk
predate that change and rerunning the whole pipeline would rewrite ~25GB of
history and feature stores for a label-only change. This backfills just the
all-positive label frames from artifacts already on disk.

Correctness is not assumed: for every snapshot the FIRST positive of each user
in the generated frame must reproduce the frozen groundtruth_{snapshot}.parquet
exactly, on every shared column. A mismatch aborts before anything is written.

    python -m scripts.data.make_all_positive_groundtruth [--category CAT] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.data.temporal_split import all_positive_labels, first_positive_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SNAPSHOT_NAMES = ("ranker_train", "model_selection", "test")
CATEGORIES = ("All_Beauty", "Video_Games", "Books", "Electronics")
WARM_USER_MIN_HISTORY = 3
MIN_ITEM_HISTORY = 5


def _annotate(labels: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Exactly build_snapshot's label annotation, from the snapshot's history."""
    item_hist_counts = history["parent_asin"].value_counts()
    eligible = set(item_hist_counts[item_hist_counts >= MIN_ITEM_HISTORY].index)
    user_hist_counts = history.groupby("user_id").size()
    labels = labels.assign(
        n_history_events=labels["user_id"].map(user_hist_counts).fillna(0).astype(int),
        item_in_history=labels["parent_asin"].isin(item_hist_counts.index).astype(int),
        item_in_eligible_pool=labels["parent_asin"].isin(eligible).astype(int),
    )
    labels["is_warm_user"] = (
        labels["n_history_events"] >= WARM_USER_MIN_HISTORY
    ).astype(int)
    return labels


def run_category(category: str, dry_run: bool = False) -> dict:
    tdir = PROCESSED_DIR / category / "temporal_ranker"
    clean = pd.read_parquet(PROCESSED_DIR / category / "interactions_clean.parquet")
    cut = json.loads((tdir / "snapshot_manifest.json").read_text())["cutoffs"]
    bounds = {
        "ranker_train": (cut["t0_ms"], cut["t1_ms"]),
        "model_selection": (cut["t1_ms"], cut["t2_ms"]),
        "test": (cut["t2_ms"], cut["t3_ms"]),
    }

    out = {}
    for name, (hist_end, label_end) in bounds.items():
        history = pd.read_parquet(tdir / f"history_{name}.parquet")
        ts = clean["timestamp"]
        window = clean[(ts >= hist_end) & (ts < label_end)]

        alls = _annotate(all_positive_labels(window), history)
        firsts = _annotate(first_positive_labels(window), history)

        # The frozen frame is the contract. Regenerating it from the same inputs
        # must reproduce it exactly, or the generation logic has drifted and the
        # all-positive frame cannot be trusted either.
        frozen = pd.read_parquet(tdir / f"groundtruth_{name}.parquet")
        cols = [c for c in frozen.columns if c in firsts.columns]
        a = firsts[cols].sort_values(["user_id", "parent_asin"]).reset_index(drop=True)
        b = frozen[cols].sort_values(["user_id", "parent_asin"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)

        # And the frozen targets must all survive into the all-positive frame.
        fp = set(map(tuple, frozen[["user_id", "parent_asin"]].to_numpy()))
        ap = set(map(tuple, alls[["user_id", "parent_asin"]].to_numpy()))
        assert fp <= ap, f"{category}/{name}: {len(fp - ap)} frozen targets lost"
        assert len(ap) == len(alls), f"{category}/{name}: duplicate (user, item) rows"

        n_rows, n_users = len(alls), int(alls["user_id"].nunique())
        assert n_users == frozen["user_id"].nunique(), (
            f"{category}/{name}: label user set changed"
        )
        out[name] = {
            "n_rows": n_rows,
            "n_users": n_users,
            "positives_per_user": n_rows / n_users,
            "discarded_by_first_only": 1 - len(frozen) / n_rows,
        }
        print(f"  {category:12s} {name:16s} {len(frozen):>8,} -> {n_rows:>8,} rows "
              f"over {n_users:>8,} users ({n_rows / n_users:.2f}/user, "
              f"first-only discards {1 - len(frozen) / n_rows:>6.1%})", flush=True)

        if not dry_run:
            alls[frozen.columns.tolist()].to_parquet(
                tdir / f"groundtruth_all_{name}.parquet", index=False)
        del history, window, alls, firsts, frozen
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", action="append", choices=list(CATEGORIES))
    p.add_argument("--dry-run", action="store_true",
                   help="verify against the frozen frames, write nothing")
    args = p.parse_args(argv)
    cats = args.category or list(CATEGORIES)
    summary = {c: run_category(c, args.dry_run) for c in cats}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
