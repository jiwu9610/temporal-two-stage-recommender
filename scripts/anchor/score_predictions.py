"""Score a leg's top-K dump against the anchor test parquet (spec P1.2 item 3).

CLI (tae env; needs pyarrow for the two-tower parquet dump):
    python -m scripts.anchor.score_predictions --category All_Beauty
        [--leg two_tower_anchor_v1] [--predictions PATH] [--keep-valid-comparison]
        [--self-test-only]

With only --category, discovers every dump in results/anchor/{cat}/
(``*_top*.csv.gz`` from run_recbole_anchor + ``{cat}_two_tower_anchor_v1_
predictions.parquet`` from train_two_tower_anchor) and scores each.

The seen-mask asymmetry fix (Track C item 3, the point of this script):
    our two-tower dump masks train-seen items only, so the user's VALID item is
    still rankable at test time — RecBole's general-model full-sort masks
    train+valid at test. Before scoring, each user's valid item(s) are REMOVED
    from that user's ranked list (list compacts upward). Only after this
    removal is our metric definition identical to RecBole's ("剔除后恒等").
    Both with/without-removal numbers are reported so the delta is visible;
    an inline self-test (constructed example) asserts the removal changes the
    metric and runs before any real scoring.

Metrics: Recall@{10,50,100} (single held-out item => hit rate) and
NDCG@{10,50,100} (= 1/log2(1+rank)), under two ground-truth families:
    official_rating_blind  gt = the held-out test item for EVERY user
                           (the official protocol ignores rating; matches RecBole)
    positives_only         gt restricted to test rows with label==1
                           (the phase1 evaluator convention)

Also reported: out-of-catalog test-item rate (test items absent from the
train catalogue can never be retrieved by a train-vocab model — VG ~0.8%),
users missing from the dump (scored as misses), and how many users actually
had a valid item removed.

Determinism: dumps are ordered by explicit rank when present, else by
(score desc, parent_asin asc), always with a stable mergesort.

Output: results/anchor/{cat}/score_{leg}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.anchor.anchor_common import (
    ANCHOR_DATA_ROOT,
    ANCHOR_RESULTS_ROOT,
    CATEGORIES,
    ISOLATION_DECLARATION,
    KS,
    MEMO_LIFETIME_ITEM_FEATURES,
    MEMO_SELECTION_ASYMMETRY,
    SEEN_MASK_CONVENTIONS,
    write_anchor_memo,
    write_comparison_skeleton,
)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---- Dump loading ------------------------------------------------------------

def load_dump(path: Path) -> pd.DataFrame:
    """Load a top-K dump (parquet / csv / csv.gz) into user_id, parent_asin,
    plus whatever of {rank, score} it carries."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"user_id": str, "parent_asin": str})
    missing = {"user_id", "parent_asin"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: dump missing columns {sorted(missing)}")
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    return df


def ranked_lists(dump: pd.DataFrame) -> dict[str, list[str]]:
    """user_id -> ranked item list, deterministic under ties.

    Order: explicit rank asc when present; else score desc; parent_asin asc as
    the final tie-break; stable mergesort throughout.
    """
    d = dump
    if "rank" in d.columns:
        d = d.sort_values(["user_id", "rank", "parent_asin"], kind="mergesort")
    elif "score" in d.columns:
        d = d.sort_values("parent_asin", kind="mergesort")
        d = d.sort_values(["user_id", "score"], ascending=[True, False],
                          kind="mergesort")
    else:
        raise ValueError("dump has neither 'rank' nor 'score' — order undefined")
    return {u: g["parent_asin"].tolist() for u, g in d.groupby("user_id", sort=True)}


# ---- Metric core -------------------------------------------------------------

def _rank_of(lst: list[str], item: str) -> int | None:
    """1-based rank of item in lst, None if absent."""
    try:
        return lst.index(item) + 1
    except ValueError:
        return None


def score_ranks(ranks: list[int | None], ks=KS) -> dict[str, float]:
    """ranks: one entry per eval user (None = miss/absent). Mean Recall@K
    (hit rate, single held-out item) and NDCG@K (1/log2(1+rank))."""
    n = len(ranks)
    out: dict[str, float] = {"n_eval_users": n}
    if n == 0:
        for k in ks:
            out[f"Recall@{k}"] = 0.0
            out[f"NDCG@{k}"] = 0.0
        return out
    arr = np.array([r if r is not None else np.inf for r in ranks])
    for k in ks:
        hits = arr <= k
        out[f"Recall@{k}"] = float(hits.mean())
        gains = np.where(hits, 1.0 / np.log2(1.0 + np.minimum(arr, k + 1)), 0.0)
        out[f"NDCG@{k}"] = float(gains.mean())
    return out


def evaluate_dump(
    lists: dict[str, list[str]],
    test_df: pd.DataFrame,
    valid_map: dict[str, set],
    remove_valid: bool,
    ks=KS,
) -> dict:
    """Score one dump under one removal setting, both gt families."""
    ranks_all: list[int | None] = []
    ranks_pos: list[int | None] = []
    n_missing_from_dump = 0
    n_valid_removed = 0

    for row in test_df.itertuples(index=False):
        uid, test_item, label = str(row.user_id), str(row.parent_asin), int(row.label)
        lst = lists.get(uid)
        if lst is None:
            n_missing_from_dump += 1
            rank = None
        else:
            if remove_valid:
                vset = valid_map.get(uid)
                if vset and any(x in vset for x in lst):
                    lst = [x for x in lst if x not in vset]
                    n_valid_removed += 1
            rank = _rank_of(lst, test_item)
        ranks_all.append(rank)
        if label == 1:
            ranks_pos.append(rank)

    return {
        "valid_item_removed_before_scoring": remove_valid,
        "n_users_missing_from_dump": n_missing_from_dump,
        "n_users_valid_item_removed": n_valid_removed,
        "official_rating_blind": score_ranks(ranks_all, ks),
        "positives_only": score_ranks(ranks_pos, ks),
    }


# ---- Self-test: removal must change the metric -------------------------------

def self_test() -> dict:
    """Constructed example proving the valid-item removal changes the metric.

    u1: dump = [V, T, x] — valid item V sits ABOVE the test item T.
        without removal: T at rank 2 -> Recall@1 = 0,   NDCG@2 = 1/log2(3)=0.6309
        with removal:    T at rank 1 -> Recall@1 = 1,   NDCG@2 = 1.0
    u2: dump = [T, V, x] — valid item BELOW the test item: removal is a no-op
        on T's rank (rank 1 both ways), isolating the asymmetry to u1.
    """
    lists = {"u1": ["V1", "T1", "x1"], "u2": ["T2", "V2", "x2"]}
    test_df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "parent_asin": ["T1", "T2"],
        "label": [1, 1],
    })
    valid_map = {"u1": {"V1"}, "u2": {"V2"}}
    ks = (1, 2)
    without = evaluate_dump(lists, test_df, valid_map, remove_valid=False, ks=ks)
    with_ = evaluate_dump(lists, test_df, valid_map, remove_valid=True, ks=ks)

    w0 = without["official_rating_blind"]
    w1 = with_["official_rating_blind"]
    assert w0["Recall@1"] == 0.5 and w1["Recall@1"] == 1.0, (w0, w1)
    assert w1["NDCG@2"] > w0["NDCG@2"], (w0, w1)
    assert with_["n_users_valid_item_removed"] == 2
    return {
        "passed": True,
        "construction": self_test.__doc__.strip().splitlines()[0],
        "without_removal": {k: w0[k] for k in ("Recall@1", "NDCG@2")},
        "with_removal": {k: w1[k] for k in ("Recall@1", "NDCG@2")},
    }


# ---- Per-leg scoring ---------------------------------------------------------

def discover_dumps(category: str) -> dict[str, Path]:
    """leg name -> dump path, from results/anchor/{cat}/."""
    out_dir = ANCHOR_RESULTS_ROOT / category
    found: dict[str, Path] = {}
    if not out_dir.exists():
        return found
    for p in sorted(out_dir.glob("*_top*.csv.gz")):
        leg = p.name.rsplit("_top", 1)[0]
        found[leg] = p
    tt = out_dir / f"{category}_two_tower_anchor_v1_predictions.parquet"
    if tt.exists():
        found["two_tower_anchor_v1"] = tt
    return found


def score_leg(category: str, leg: str, dump_path: Path, st: dict) -> dict:
    t0 = time.time()
    cat_dir = ANCHOR_DATA_ROOT / category
    test_df = pd.read_parquet(cat_dir / "test.parquet",
                              columns=["user_id", "parent_asin", "label"])
    val_df = pd.read_parquet(cat_dir / "val.parquet",
                             columns=["user_id", "parent_asin"])
    train_items = set(
        pd.read_parquet(cat_dir / "train.parquet", columns=["parent_asin"])
        ["parent_asin"].astype(str).unique()
    )
    assert test_df["user_id"].is_unique, "anchor test parquet must be one row/user"
    test_df = test_df.copy()
    test_df["user_id"] = test_df["user_id"].astype(str)
    test_df["parent_asin"] = test_df["parent_asin"].astype(str)
    valid_map = (val_df.assign(user_id=val_df["user_id"].astype(str),
                               parent_asin=val_df["parent_asin"].astype(str))
                 .groupby("user_id")["parent_asin"].agg(set).to_dict())

    dump = load_dump(dump_path)
    lists = ranked_lists(dump)
    dump_k = int(dump.groupby("user_id").size().max())

    with_removal = evaluate_dump(lists, test_df, valid_map, remove_valid=True)
    without_removal = evaluate_dump(lists, test_df, valid_map, remove_valid=False)

    oob = ~test_df["parent_asin"].isin(train_items)
    removal_delta = {
        m: round(with_removal["official_rating_blind"][m]
                 - without_removal["official_rating_blind"][m], 6)
        for m in with_removal["official_rating_blind"]
        if m != "n_eval_users"
    }

    report = {
        "category": category,
        "leg": leg,
        "dump": {
            "path": str(dump_path),
            "rows": int(len(dump)),
            "n_users": int(dump["user_id"].nunique()),
            "max_list_length": dump_k,
            "distinct_items_in_dump": int(dump["parent_asin"].nunique()),
        },
        "created_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "self_test_valid_removal_changes_metric": st,
        "protocol": {
            "primary": "with_valid_removed / official_rating_blind "
                       "(RecBole-comparable after removal — '剔除后恒等')",
            "tie_break": "explicit rank asc if present, else score desc, "
                         "parent_asin asc; stable mergesort",
            "tie_caveat": (
                "dump ranks resolve tied model scores by torch.topk order at "
                "k=101; RecBole's internal evaluator resolves the same ties "
                "independently at k=100. Under heavy score ties (Pop counts "
                "especially) the per-K hit set can differ inside a tie group "
                "even though both are deterministic — measured on All_Beauty: "
                "pop matches RecBole exactly at @100 but not @10, itemknn "
                "matches exactly at every K. Cross-check on tie-robust points."
            ),
            "ndcg": "single held-out item: NDCG@K = 1/log2(1+rank)",
            "gt_families": {
                "official_rating_blind": "held-out test item for every user "
                                         "(official protocol ignores rating)",
                "positives_only": "test rows with label==1 only (phase1 "
                                  "evaluator convention)",
            },
            "catalog_definition": "distinct parent_asin in anchor train.parquet",
            "seen_mask_conventions": SEEN_MASK_CONVENTIONS,
            "selection_asymmetry_memo": MEMO_SELECTION_ASYMMETRY,
            "item_feature_leakage_memo": MEMO_LIFETIME_ITEM_FEATURES,
            "isolation_declaration": ISOLATION_DECLARATION,
        },
        "n_test_users": int(len(test_df)),
        "out_of_catalog_test_item": {
            "n": int(oob.sum()),
            "rate": float(oob.mean()),
            "catalog_size": len(train_items),
        },
        "metrics": {
            "with_valid_removed": with_removal,
            "without_removal": without_removal,
            "removal_delta_official_rating_blind": removal_delta,
        },
    }

    out_path = ANCHOR_RESULTS_ROOT / category / f"score_{leg}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    m = with_removal["official_rating_blind"]
    print(f"[score:{category}/{leg}] out-of-catalog test items: "
          f"{report['out_of_catalog_test_item']['n']} "
          f"({report['out_of_catalog_test_item']['rate']:.4f}); "
          f"valid removed for {with_removal['n_users_valid_item_removed']} users",
          flush=True)
    print(f"[score:{category}/{leg}] R@10={m['Recall@10']:.4f} "
          f"N@10={m['NDCG@10']:.4f} R@100={m['Recall@100']:.4f} "
          f"(delta from removal: R@10 {removal_delta['Recall@10']:+.4f}) "
          f"-> {out_path}", flush=True)
    return report


# ---- CLI ---------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--leg", default=None,
                   help="score only this leg (default: discover all dumps)")
    p.add_argument("--predictions", default=None,
                   help="explicit dump path (requires --leg for the label)")
    p.add_argument("--self-test-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    st = self_test()
    print(f"[score] self-test (removal changes metric): {st}", flush=True)
    if args.self_test_only:
        return 0

    write_anchor_memo()
    if args.predictions:
        if not args.leg:
            print("--predictions requires --leg", file=sys.stderr)
            return 2
        targets = {args.leg: Path(args.predictions)}
    else:
        targets = discover_dumps(args.category)
        if args.leg:
            if args.leg not in targets:
                print(f"no dump found for leg {args.leg!r} in "
                      f"{ANCHOR_RESULTS_ROOT / args.category} "
                      f"(found: {sorted(targets)})", file=sys.stderr)
                return 2
            targets = {args.leg: targets[args.leg]}
    if not targets:
        print(f"no dumps found under {ANCHOR_RESULTS_ROOT / args.category}",
              file=sys.stderr)
        return 2

    for leg, path in targets.items():
        score_leg(args.category, leg, path, st)
    comp = write_comparison_skeleton()
    print(f"[score] comparison skeleton refreshed -> {comp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
