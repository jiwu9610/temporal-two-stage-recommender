"""Canon-v2 audit: are the v1 title-only merges genuine?

Classifies every multi-asin EXACT-TITLE group (the pre-C1 v1 rule) on raw
metadata by its store composition:

    cross_store       >=2 distinct known stores -- Live2Pedal-style bogus-merge
                      risk (same generic title, different sellers)
    single_store      exactly 1 distinct known store, no unknown members --
                      genuine-looking duplicate listing
    store_incomplete  some/all members have null or blank store -- cannot
                      adjudicate from the store field

and reports group / folded-asin / member-review-row weights per class. Then it
verifies the SHIPPED canon v2 key ((norm_title, store), scripts.data.canonicalize)
against that classification with hard asserts:

    * no v2 canonical id unites two members with different known stores
      (globally, over the whole catalog -- the identity-hygiene invariant);
    * every single_store group whose normalized title is non-empty STILL
      merges under v2 (same-store duplicates are the merges we keep).

Deterministic samples (top-by-size + seeded random, seed 0) of both classes go
to samples_{category}.json for eyeball review; summary numbers go to
genuineness_{category}.json. Both are committed for reproducibility.

Usage (repo root; All_Beauty only on the login node -- other categories via
the batch loop documented in canon_options.py):
    python -m scripts.analysis.canon_v2_audit.genuineness --category All_Beauty
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.data.canonicalize import _build_canonical_map, normalize_title, store_merge_key

from .common import (
    AUDIT_DIR,
    CATEGORIES,
    git_commit,
    load_meta_dedup,
    review_row_counts,
    utc_now_iso,
)

N_TOP_SAMPLES = 10
N_RANDOM_SAMPLES = 10
SAMPLE_SEED = 0


def classify_group(store_keys: pd.Series) -> str:
    known = store_keys[store_keys != ""]
    n_known_distinct = known.nunique()
    if n_known_distinct >= 2:
        return "cross_store"
    if n_known_distinct == 1 and len(known) == len(store_keys):
        return "single_store"
    return "store_incomplete"


def run_category(category: str) -> tuple:
    print(f"[genuineness] category={category}", flush=True)
    meta = load_meta_dedup(category)
    counts, total_rows = review_row_counts(category)

    work = meta[["parent_asin", "title", "store"]].copy()
    if "rating_number" in meta.columns:
        work["rating_number"] = meta["rating_number"]
    else:
        work["rating_number"] = np.nan
    work["_store_key"] = work["store"].map(store_merge_key)
    work["_n_review_rows"] = counts.reindex(work["parent_asin"]).fillna(0).to_numpy().astype(int)

    # ---- shipped v2 mapping + global identity-hygiene invariant --------------
    # (added to `work` BEFORE the group slices below are taken)
    v2_mapping, _, _ = _build_canonical_map(meta, "parent_asin")
    work["_v2_canon"] = work["parent_asin"].map(v2_mapping)
    assert work["_v2_canon"].notna().all(), "v2 mapping must cover every parent_asin"
    known = work[work["_store_key"] != ""]
    stores_per_canon = known.groupby("_v2_canon")["_store_key"].nunique()
    n_poisoned = int((stores_per_canon > 1).sum())
    assert n_poisoned == 0, (
        f"identity-hygiene violation: {n_poisoned} v2 canonical ids unite "
        f">1 known store -- the C1 bug would survive the rebuild"
    )

    # ---- v1 multi-asin exact-title groups -----------------------------------
    titled = work[work["title"].notna()]
    sizes = titled.groupby("title", sort=False)["parent_asin"].size()
    multi_titles = sizes[sizes > 1].index
    multi = titled[titled["title"].isin(multi_titles)]
    print(f"    v1 multi-asin exact-title groups: {len(multi_titles):,} "
          f"({len(multi):,} member asins)", flush=True)

    # ---- classify + verify each v1 group ------------------------------------
    classes: dict = {c: {"n_groups": 0, "n_parent_asins_folded": 0, "n_member_review_rows": 0}
                     for c in ("cross_store", "single_store", "store_incomplete")}
    group_records = {"cross_store": [], "single_store": [], "store_incomplete": []}

    for title, grp in multi.groupby("title", sort=False):
        cls = classify_group(grp["_store_key"])
        classes[cls]["n_groups"] += 1
        classes[cls]["n_parent_asins_folded"] += len(grp) - 1
        classes[cls]["n_member_review_rows"] += int(grp["_n_review_rows"].sum())

        if cls == "single_store" and normalize_title(title) != "":
            assert grp["_v2_canon"].nunique() == 1, (
                f"same-store duplicate group {title!r} no longer merges under "
                f"v2: {dict(zip(grp['parent_asin'], grp['_v2_canon']))}"
            )

        group_records[cls].append({
            "title": title,
            "n_members": int(len(grp)),
            "n_member_review_rows": int(grp["_n_review_rows"].sum()),
            "members": [
                {
                    "parent_asin": row["parent_asin"],
                    "store": None if pd.isna(row["store"]) else str(row["store"]),
                    "rating_number": None if pd.isna(row["rating_number"]) else int(row["rating_number"]),
                    "n_review_rows": int(row["_n_review_rows"]),
                    "v2_canonical": row["_v2_canon"],
                }
                for _, row in grp.iterrows()
            ],
        })

    # Cross-store groups must be broken apart by v2: no two members with
    # different known stores may share a v2 id (already guaranteed by the
    # global invariant above; this is the per-class restatement).
    for rec in group_records["cross_store"]:
        by_canon: dict = {}
        for m in rec["members"]:
            sk = store_merge_key(m["store"])
            if sk:
                by_canon.setdefault(m["v2_canonical"], set()).add(sk)
        assert all(len(s) <= 1 for s in by_canon.values()), (
            f"cross-store group {rec['title']!r} still merged under v2"
        )

    # ---- deterministic samples ----------------------------------------------
    samples: dict = {}
    rng = np.random.RandomState(SAMPLE_SEED)
    for cls in ("cross_store", "single_store"):
        recs = sorted(group_records[cls], key=lambda r: (-r["n_members"], r["title"]))
        top = recs[:N_TOP_SAMPLES]
        rest = recs[N_TOP_SAMPLES:]
        rand = (
            [rest[i] for i in sorted(rng.choice(len(rest), size=N_RANDOM_SAMPLES, replace=False))]
            if len(rest) > N_RANDOM_SAMPLES else rest
        )
        samples[cls] = {"top_by_size": top, "seeded_random": rand}

    n_multi = int(len(multi_titles))
    summary = {
        "category": category,
        "generated_utc": utc_now_iso(),
        "repo_commit": git_commit(),
        "totals": {
            "n_meta_rows_after_parent_dedup": len(meta),
            "n_review_rows": total_rows,
            "n_v1_multi_asin_title_groups": n_multi,
        },
        "classes": {
            cls: {
                **vals,
                "pct_of_groups": round(100.0 * vals["n_groups"] / n_multi, 2) if n_multi else 0.0,
            }
            for cls, vals in classes.items()
        },
        "v2_verification": {
            "global_no_canonical_id_unites_two_known_stores": True,
            "all_single_store_groups_with_nonempty_norm_title_still_merge": True,
            "note": "both enforced by hard asserts above; this report only "
                    "exists if they passed",
        },
    }
    return summary, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    args = parser.parse_args()

    summary, samples = run_category(args.category)

    summary_path = AUDIT_DIR / f"genuineness_{args.category}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    samples_path = AUDIT_DIR / f"samples_{args.category}.json"
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")

    print(f"    wrote {summary_path}", flush=True)
    print(f"    wrote {samples_path}", flush=True)
    for cls, vals in summary["classes"].items():
        print(f"    {cls:18s} groups={vals['n_groups']:>7,} ({vals['pct_of_groups']:5.1f}%) "
              f"folded={vals['n_parent_asins_folded']:>7,} "
              f"member_rows={vals['n_member_review_rows']:>10,}", flush=True)


if __name__ == "__main__":
    main()
