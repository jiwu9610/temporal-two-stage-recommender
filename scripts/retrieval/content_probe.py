"""P1.4 pre-C1 go/no-go probe: content_i2i on the EXISTING (poisoned) embeddings.

CLI:
    python -m scripts.retrieval.content_probe --category Books \
        [--configs A0,A1,A5] [--top-k 500] [--keep-candidates]

Purpose (REBUILD_WAVE_SPEC P1.4): before the Track-D rebuild regenerates
embeddings, measure whether the content_i2i source finds ANY unique hits on
the current identity-poisoned `item_text_embeddings.npz`. The poisoned
embeddings are a pessimistic lower bound; if even that bound is
substantively > 0 for Books/Electronics, Track B proceeds -- if it is ~0 the
recipe goes back to the advisor. NO freeze decision is taken here (the data
layer underneath is about to be replaced).

Configs (recorded verbatim in the report):
    A0  frozen C2_k500_recent shape (thr5, top_k/tt_k 500, recent_pop 30d)
        rebuilt on today's data layer -- the union-delta baseline
    A1  A0 + content_i2i over the catalog pool (the probe subject)
    A5  thr3, no content (threshold-attribution control)

TEST-LOCK: reads ranker_train + model_selection snapshots ONLY; the module
refuses to run on the test snapshot. Retrieval metrics only -- no ranker is
trained. Candidate parquets are written to temporal_ranker/variants/probe_*/
and deleted after metric extraction unless --keep-candidates is passed;
production artifacts (rule_weights.json included) are never rewritten.

Report per (config, snapshot): builder per-source recalls + union coverage,
content unique-hit count/share, and hit-by-item-history buckets
(0 / 1-2 / 3-4 / >=5 history events of the TARGET item) for content-covered
and content-unique targets. Plus the A1-vs-A0 union delta.

Cost note: recommend_content_knn is a per-user mat-vec over the full
embedding matrix (5.9 GB for Books) -- profile-holding users only. Expect
minutes for All_Beauty/Video_Games, ~1-3 h for Electronics and ~4-6 h for
Books per full probe; run via run_content_probe.sbatch (CPU, 64G), never on
the login node.

Output: results/phase3a/{category}_content_probe.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq

from scripts.ranker.temporal_candidate_builder import (
    PROCESSED_DIR,
    build_snapshot_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "phase3a"
PROBE_SNAPSHOTS = ("ranker_train", "model_selection")   # test is LOCKED
CONTENT = {"name": "content_i2i", "type": "content_i2i", "pool": "catalog",
           "recency_decay": 0.8, "max_events": 30}


def _recent_pop_for(category: str) -> dict:
    """The frozen C2 recent_pop config, which is PER-CATEGORY (AB windowed 30d,
    VG decayed 7d, Books decayed 90d, Elec decayed 30d). A0 exists to be the
    frozen baseline; hard-coding one flavor for all categories compares the
    content arm against a baseline that never shipped."""
    frozen = json.loads(
        (REPO_ROOT / "results" / "phase3a" / "frozen_config.json").read_text())
    cfg = frozen["config"]["recent_pop_per_category"][category]
    return {"name": "recent_pop", **cfg}


def probe_configs(category: str) -> Dict[str, dict]:
    recent_pop = _recent_pop_for(category)
    return {
        "A0": {"top_k": 500, "tt_k": 500, "min_item_history": None,
               "extra_sources": [recent_pop]},
        "A1": {"top_k": 500, "tt_k": 500, "min_item_history": None,
               "extra_sources": [recent_pop, CONTENT]},
        "A5": {"top_k": 500, "tt_k": 500, "min_item_history": 3,
               "extra_sources": [recent_pop]},
    }
BUCKETS = ("0", "1-2", "3-4", ">=5")


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1-2" if n <= 2 else "3-4" if n <= 4 else ">=5"


def _positive_rows(cand_path: Path) -> pd.DataFrame:
    """label==1 rows with the source indicator columns only (tiny frame)."""
    schema_names = pq.ParquetFile(cand_path).schema_arrow.names
    src_cols = [c for c in schema_names if c.startswith("source_")]
    tbl = pq.read_table(cand_path,
                        columns=["user_id", "parent_asin", "label", *src_cols],
                        filters=[("label", "==", 1)])
    return tbl.to_pandas()


def _content_metrics(pos: pd.DataFrame, n_positives: int,
                     hist_counts: pd.Series) -> Dict:
    """Unique-hit share + item-history buckets for the content source."""
    other = [c for c in pos.columns
             if c.startswith("source_") and c != "source_content_i2i"]
    covered = pos[pos["source_content_i2i"] == 1]
    unique = covered[covered[other].sum(axis=1) == 0]

    def _hist_buckets(df: pd.DataFrame) -> Dict[str, int]:
        out = {b: 0 for b in BUCKETS}
        for it in df["parent_asin"]:
            out[_bucket(int(hist_counts.get(it, 0)))] += 1
        return out

    n_covered_pairs = len(pos.drop_duplicates(["user_id", "parent_asin"]))
    return {
        "content_covered_positives": int(len(covered)),
        "content_unique_hit_positives": int(len(unique)),
        "content_unique_hit_share_of_positives":
            len(unique) / n_positives if n_positives else 0.0,
        "content_unique_hit_share_of_covered":
            len(unique) / n_covered_pairs if n_covered_pairs else 0.0,
        "content_hits_by_item_history": _hist_buckets(covered),
        "content_unique_hits_by_item_history": _hist_buckets(unique),
    }


def run_category(
    category: str,
    config_names: List[str],
    top_k: int = 500,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    keep_candidates: bool = False,
    seed: int = 42,
) -> Dict:
    t_start = time.time()
    tdir = Path(processed_dir) / category / "temporal_ranker"
    hist_counts = {
        snap: pd.read_parquet(
            tdir / f"history_{snap}.parquet", columns=["parent_asin"]
        )["parent_asin"].astype(str).value_counts()
        for snap in PROBE_SNAPSHOTS
    }

    per_config: Dict[str, Dict] = {}
    for cfg_name in config_names:
        rc = json.loads(json.dumps(probe_configs(category)[cfg_name]))  # deep copy
        rc["top_k"] = top_k
        per_config[cfg_name] = {"retrieval_config": rc, "snapshots": {}}
        for snap in PROBE_SNAPSHOTS:
            assert snap != "test", "probe must never touch the test snapshot"
            out_dir = tdir / "variants" / f"probe_{cfg_name}"
            print(f"[probe {category}] {cfg_name}/{snap} ...", flush=True)
            report = build_snapshot_candidates(
                category, snap, top_k=rc["top_k"], seed=seed,
                processed_dir=processed_dir, require_two_tower=True,
                min_item_history=rc["min_item_history"],
                extra_sources=rc["extra_sources"], tt_k=rc["tt_k"],
                out_dir=out_dir,
            )
            entry = {
                "coverage": report["coverage"],
                "sources": report["sources"],
                "n_positive_eval_users": report["n_positive_eval_users"],
                "n_positives": report["n_positives"],
                "candidate_catalog_size": report["candidate_catalog_size"],
                "n_rows": report["n_rows"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            has_content = any(s["type"] == "content_i2i"
                              for s in rc["extra_sources"])
            cand_path = out_dir / f"candidates_{snap}.parquet"
            if has_content:
                entry.update(_content_metrics(
                    _positive_rows(cand_path), report["n_positives"],
                    hist_counts[snap]))
            if not keep_candidates:
                cand_path.unlink()
            per_config[cfg_name]["snapshots"][snap] = entry

    union_delta = {}
    if "A0" in per_config and "A1" in per_config:
        for snap in PROBE_SNAPSHOTS:
            a0 = per_config["A0"]["snapshots"][snap]["coverage"]
            a1 = per_config["A1"]["snapshots"][snap]["coverage"]
            union_delta[snap] = {
                "retrieved_union_delta":
                    a1["retrieved_union"] - a0["retrieved_union"],
                "retrieved_union_per_positive_delta":
                    a1["retrieved_union_per_positive"]
                    - a0["retrieved_union_per_positive"],
            }

    report = {
        "category": category,
        "purpose": "P1.4 pre-C1 go/no-go probe on POISONED embeddings "
                   "(pessimistic lower bound; NO freeze decision here)",
        "test_lock": "ranker_train + model_selection only; "
                     "groundtruth_test never read",
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "top_k": top_k,
        "configs": per_config,
        "union_delta_A1_vs_A0": union_delta,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "peak_rss_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{category}_content_probe.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[probe {category}] written -> {out} "
          f"({report['elapsed_seconds']}s, peak {report['peak_rss_gb']}GB)",
          flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--configs", default="A0,A1,A5",
                   help="Comma-separated subset of [A0, A1, A5].")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--keep-candidates", action="store_true",
                   help="Keep probe candidate parquets under variants/probe_* "
                        "(default: delete after metric extraction).")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in names if c not in ("A0", "A1", "A5")]
    if unknown:
        raise SystemExit(f"unknown probe configs {unknown}; "
                         "choose from [A0, A1, A5]")
    run_category(args.category, names, top_k=args.top_k,
                 keep_candidates=args.keep_candidates)


if __name__ == "__main__":
    main()
