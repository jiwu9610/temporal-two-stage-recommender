"""Canon-v2 audit: merge-key option comparison (Track D; MEMO D5; spec P2.1).

Compares four canonicalization rules on one category's RAW metadata + reviews:

    current_v1_exact_title : the pre-C1 shipped rule -- exact raw title match,
                             null title never merges ("" titles DO merge, which
                             is part of the bug). Reproduction anchor for the
                             historical numbers (Books 348,580/583,214-scale).
    a_norm_title_only      : normalized title alone (lowercase / strip /
                             whitespace-collapse; empty never merges).
    b_norm_title_store     : (norm_title, store) -- the APPROVED canon v2 key,
                             computed by the SHIPPED scripts.data.canonicalize
                             _build_canonical_map so the recorded numbers come
                             from the exact predicate S1 will run with.
    c_no_merge             : trust raw parent_asin identity outright (merge
                             nothing; lower bound).

Per option: multi-asin merge groups, parent_asins folded, canonical catalog
size, review rows remapped to a different id (count + share). The option-(b)
numbers produced by this script are the recorded S1 rebuild expectations
(P2.1 gate) -- they are predicate-sensitive (+-0.5-1.5% between predicate
variants), which is why the predicate is imported, not redefined.

Usage (repo root):
    python -m scripts.analysis.canon_v2_audit.canon_options --category All_Beauty

The login node (6GB cgroup) is fine for All_Beauty ONLY. Books / Electronics /
Video_Games must go through sbatch (documented here, submitted manually):

    for CAT in Video_Games Books Electronics; do
      sbatch --account=PROJECT_ALLOC --partition=normal --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=64G --time=02:00:00 \
        --job-name=canon_audit_$CAT --output=logs/canon_audit_${CAT}_%j.out \
        --wrap "source ~/.bashrc && conda activate tae && \
                cd /projects/PROJECT_ALLOC/R-project && \
                python -m scripts.analysis.canon_v2_audit.canon_options --category $CAT && \
                python -m scripts.analysis.canon_v2_audit.genuineness --category $CAT"
    done

Output: scripts/analysis/canon_v2_audit/canon_options_{category}.json
(committed alongside the scripts for reproducibility).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from scripts.data.canonicalize import _build_canonical_map, normalize_title

from .common import (
    AUDIT_DIR,
    CATEGORIES,
    git_commit,
    load_meta_dedup,
    mapping_from_key,
    review_row_counts,
    summarize_mapping,
    utc_now_iso,
)


def run_category(category: str) -> dict:
    print(f"[canon-options] category={category}", flush=True)
    meta = load_meta_dedup(category)
    counts, total_rows = review_row_counts(category)
    print(f"    meta rows after parent dedup: {len(meta):,} | "
          f"review rows: {total_rows:,}", flush=True)

    options: dict = {}

    # ---- current_v1_exact_title (reproduction anchor for the pre-C1 rule) ----
    titles = meta["title"]
    v1_canon, v1_groups = mapping_from_key(meta, titles.to_numpy(), titles.notna().to_numpy())
    options["current_v1_exact_title"] = summarize_mapping(v1_canon, counts, total_rows, v1_groups)

    # ---- a_norm_title_only ---------------------------------------------------
    norm = titles.map(normalize_title)
    a_canon, a_groups = mapping_from_key(meta, norm.to_numpy(), (norm != "").to_numpy())
    options["a_norm_title_only"] = summarize_mapping(a_canon, counts, total_rows, a_groups)

    # ---- b_norm_title_store (SHIPPED implementation) -------------------------
    b_mapping, b_collapsed, b_groups = _build_canonical_map(meta, "parent_asin")
    b_canon = pd.Series(b_mapping, name="canonical_parent_asin")
    b_canon.index.name = "parent_asin"
    options["b_norm_title_store"] = summarize_mapping(b_canon, counts, total_rows, b_groups)
    assert options["b_norm_title_store"]["n_parent_asins_folded"] == b_collapsed, (
        "shipped n_collapsed disagrees with the audit recomputation -- "
        f"{b_collapsed} vs {options['b_norm_title_store']['n_parent_asins_folded']}"
    )

    # ---- c_no_merge ----------------------------------------------------------
    asins = meta["parent_asin"]
    c_canon = pd.Series(asins.to_numpy(), index=pd.Index(asins.to_numpy(), name="parent_asin"))
    options["c_no_merge"] = summarize_mapping(c_canon, counts, total_rows, 0)

    # Monotonicity sanity: every option merges no more than v1 folds plus what
    # normalization can add, and (b) folds strictly fewer asins than (a).
    assert options["c_no_merge"]["n_parent_asins_folded"] == 0
    assert (options["b_norm_title_store"]["n_parent_asins_folded"]
            <= options["a_norm_title_only"]["n_parent_asins_folded"]), \
        "(b) adds a store constraint on top of (a); it cannot fold more"

    report = {
        "category": category,
        "generated_utc": utc_now_iso(),
        "repo_commit": git_commit(),
        "predicate": {
            "title": "shipped scripts.data.canonicalize.normalize_title "
                     "(lowercase/strip/whitespace-collapse; empty never merges; FROZEN in tests)",
            "store": "shipped scripts.data.canonicalize.store_merge_key "
                     "(verbatim; null/whitespace-only never merges)",
            "winner": "highest rating_number, ties by parent_asin ascending "
                      "(deterministic; historical v1 runs used an unstable sort, "
                      "so tie winners -- and remapped-row counts on tie groups -- could differ)",
        },
        "totals": {
            "n_meta_rows_after_parent_dedup": len(meta),
            "n_review_rows": total_rows,
        },
        "options": options,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    args = parser.parse_args()

    report = run_category(args.category)
    out_path = AUDIT_DIR / f"canon_options_{args.category}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"    wrote {out_path}", flush=True)
    for name, o in report["options"].items():
        print(f"    {name:24s} groups={o['n_multi_asin_groups']:>9,} "
              f"folded={o['n_parent_asins_folded']:>9,} "
              f"rows_remapped={o['n_review_rows_remapped']:>11,} "
              f"({o['pct_review_rows_remapped_of_catalog_rows']:.2f}% of catalog rows)",
              flush=True)


if __name__ == "__main__":
    main()
