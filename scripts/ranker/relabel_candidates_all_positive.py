"""Relabel an existing candidate variant against the all-positive ground truth.

Retrieval never sees labels, so switching the ground-truth rule changes only the
`label` column -- the candidate rows themselves are identical. Rebuilding them
would cost hours and ~12GB of rewrite for no change; this streams the frozen
tables row group by row group and rewrites that one column into a NEW variant
directory. The frozen tables are never touched.

Reports the pos_weight shift per snapshot, because that is the number the
ranker's loss and every downstream calibration parameter key off: Platt's
intercept satisfies b ~ -log(pos_weight), so a shift here invalidates the
frozen calibration by construction.

    python -m scripts.ranker.relabel_candidates_all_positive \
        --category Video_Games [--variant C2_k500_recent] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SNAPSHOT_NAMES = ("ranker_train", "model_selection", "test")
CATEGORIES = ("All_Beauty", "Video_Games", "Books", "Electronics")


def relabel_snapshot(tdir: Path, variant: str, snapshot: str,
                     out_variant: str, dry_run: bool) -> dict:
    src = tdir / "variants" / variant / f"candidates_{snapshot}.parquet"
    dst_dir = tdir / "variants" / out_variant
    dst = dst_dir / f"candidates_{snapshot}.parquet"

    gt = pd.read_parquet(tdir / f"groundtruth_all_{snapshot}.parquet",
                         columns=["user_id", "parent_asin"])
    gt_idx = pd.MultiIndex.from_frame(
        gt.astype(str).rename(columns={"user_id": "u", "parent_asin": "i"}))

    pf = pq.ParquetFile(src)
    writer = None
    n_rows = n_pos_old = n_pos_new = n_flipped = 0
    users_affected = set()

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".parquet.tmp")

    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        key = pd.MultiIndex.from_arrays(
            [df["user_id"].astype(str), df["parent_asin"].astype(str)],
            names=["u", "i"])
        new = key.isin(gt_idx)
        old = df["label"].to_numpy().astype(bool)

        n_rows += len(df)
        n_pos_old += int(old.sum())
        n_pos_new += int(new.sum())
        flipped = new & ~old
        n_flipped += int(flipped.sum())
        users_affected.update(df.loc[flipped, "user_id"].unique().tolist())
        # A row that was positive must stay positive: the first positive is
        # always a member of the all-positive set. If this fires, the two
        # ground-truth frames disagree and nothing downstream is trustworthy.
        lost = int((old & ~new).sum())
        if lost:
            raise AssertionError(
                f"{snapshot} row group {rg}: {lost} rows were positive under the "
                f"frozen ground truth but are negative under all-positive")

        if not dry_run:
            df["label"] = new.astype(df["label"].dtype)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
        del df, key, new, old

    if writer is not None:
        writer.close()
        tmp.replace(dst)

    pw_old = (n_rows - n_pos_old) / max(1, n_pos_old)
    pw_new = (n_rows - n_pos_new) / max(1, n_pos_new)
    return {
        "n_rows": n_rows,
        "n_pos_old": n_pos_old,
        "n_pos_new": n_pos_new,
        "positive_growth": n_pos_new / max(1, n_pos_old),
        "n_rows_flipped_0_to_1": n_flipped,
        "n_users_affected": len(users_affected),
        "pos_weight_old": pw_old,
        "pos_weight_new": pw_new,
        "pos_weight_ratio": pw_new / max(1e-12, pw_old),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", action="append", choices=list(CATEGORIES))
    p.add_argument("--variant", default="C2_k500_recent")
    p.add_argument("--out-variant", default=None,
                   help="default: {variant}_allpos")
    p.add_argument("--dry-run", action="store_true",
                   help="compute the pos_weight shift, write nothing")
    args = p.parse_args(argv)
    out_variant = args.out_variant or f"{args.variant}_allpos"

    summary = {}
    for cat in (args.category or list(CATEGORIES)):
        tdir = PROCESSED_DIR / cat / "temporal_ranker"
        summary[cat] = {}
        for snap in SNAPSHOT_NAMES:
            s = relabel_snapshot(tdir, args.variant, snap, out_variant, args.dry_run)
            summary[cat][snap] = s
            print(f"  {cat:12s} {snap:16s} pos {s['n_pos_old']:>7,} -> {s['n_pos_new']:>7,} "
                  f"(x{s['positive_growth']:.2f})  pos_weight {s['pos_weight_old']:>9.1f} -> "
                  f"{s['pos_weight_new']:>9.1f} (x{s['pos_weight_ratio']:.3f})  "
                  f"users touched {s['n_users_affected']:>7,}", flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
