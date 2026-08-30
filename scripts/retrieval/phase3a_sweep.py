"""Phase 3A Stage A: retrieval-only diagnosis on the MODEL-SELECTION snapshot.

CLI:
    python -m scripts.retrieval.phase3a_sweep --category Video_Games \
        [--max-k 500] [--skip-covis]

TEST-LOCK: this module reads ONLY the model_selection snapshot ([T1,T2)
labels). It never touches groundtruth_test; the final [T2,T3) window stays
locked until one Phase 3A configuration is frozen.

Covers:
  1. candidate-K sweep (K = 100/200/500[/1000]) over popularity, rule,
     two-tower and every union -- single scoring pass at max K, all smaller
     Ks are slices; per-source unique incremental hits.
  2. eligibility-threshold ablation (min_item_history = 5/3/2) with the
     cold / low_support / eligible_not_retrieved / retrieved classification.
     Two-tower stays frozen at its trained (threshold-5) vocabulary.
  3. recent / exponentially-decayed popularity sources (7/30/90d windows,
     half-life grid) -- standalone + marginal unique hits over the base union.
  4. co-visitation grid -- standalone, marginal gain, history cohorts,
     low-support-target reach.

Ground truth is whatever the snapshot's label frame holds: one first-positive
row per user under the default label_mode, every distinct in-window positive
(rows > users) under "all_positive". Recalls stay USER-level (a user is a hit
once >=1 of their targets is retrieved) and carry a _per_positive sibling that
divides by targets instead; n_users and n_positives are reported separately.

Ranking determinism: pools are alphabetically sorted; scores are ranked with
a stable argsort on (-score) so ties break by item id, never by catalog row
order. Two-tower lists come from infer_snapshot_checkpoint (frozen weights).

Output: results/phase3a/{category}_stageA.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from scripts.retrieval.popularity import compute_popularity, user_seen_from_train
from scripts.retrieval.rule_based import build_user_facet_affinity
from scripts.retrieval.temporal_sources import (
    build_covisitation,
    classify_targets,
    decayed_popularity,
    recommend_covisitation,
    windowed_popularity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase3a"
SNAPSHOT = "model_selection"          # the ONLY snapshot Stage A may read
KS_DEFAULT = (10, 50, 100, 200, 500)


def _ru_maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _ranked_by_scores(
    pool_sorted: np.ndarray,
    scores: np.ndarray,
    seen: Set[str],
    pool_index: Mapping[str, int],
    k: int,
) -> List[str]:
    """Deterministic top-k: stable argsort of -scores over an alphabetically
    sorted pool (ties -> item id order), seen items excluded."""
    s = scores
    if seen:
        s = scores.copy()
        for it in seen:
            j = pool_index.get(it)
            if j is not None:
                s[j] = -np.inf
    order = np.argsort(-s, kind="stable")
    out = []
    for j in order:
        if not np.isfinite(s[j]):
            continue
        out.append(pool_sorted[j])
        if len(out) >= k:
            break
    return out


def _target_ranks(lst: Sequence[str], targets: Set[str]) -> Dict[str, int]:
    """1-based rank of EVERY target in lst, 0 for the ones absent.

    One pass over lst instead of a per-target list.index(): a user can hold
    several positives once the groundtruth is all_positive_labels.
    """
    if isinstance(targets, str):        # a bare item id would iterate its chars
        raise TypeError("targets must be a set of item ids, not a single id")
    # Single target is the overwhelmingly common case (it is the ONLY case
    # under first_positive labels): keep the original list.index() path, which
    # scans in C and stops at the hit. A plain Python sweep of the full list
    # measured ~3.4x slower here, and this is the hottest loop in the sweep.
    if len(targets) == 1:
        t = next(iter(targets))
        try:
            return {t: lst.index(t) + 1}
        except ValueError:
            return {t: 0}
    pos: Dict[str, int] = {}
    remaining = len(targets)
    for i, it in enumerate(lst):
        if it in targets and it not in pos:
            pos[it] = i + 1
            remaining -= 1
            if not remaining:                  # every target located; stop
                break
    return {t: pos.get(t, 0) for t in targets}


class SourceLists:
    """Per-user ranked lists at max K + the rank of every target, one source."""

    def __init__(self, name: str, lists: Dict[str, List[str]],
                 targets: Mapping[str, Set[str]]):
        self.name = name
        self.lists = lists
        self.ranks = {u: _target_ranks(lists.get(u, []), ts)
                      for u, ts in targets.items()}
        # `rank` keeps its old scalar meaning by collapsing to the BEST
        # (smallest non-zero) rank the user's targets reach, 0 if none does, so
        # hits_at() still answers "this user has >=1 target in the top-k" and is
        # unchanged while every user carries exactly one target.
        self.rank = {u: min((r for r in rs.values() if r > 0), default=0)
                     for u, rs in self.ranks.items()}

    def hits_at(self, k: int) -> Set[str]:
        return {u for u, r in self.rank.items() if 0 < r <= k}

    def hit_pairs_at(self, k: int) -> Set[Tuple[str, str]]:
        """(user, target) pairs covered at k -- the micro counterpart of
        hits_at(), which collapses a heavy user's n positives into one hit."""
        return {(u, t) for u, rs in self.ranks.items()
                for t, r in rs.items() if 0 < r <= k}


def _union_stats(sources: List[SourceLists], users: Sequence[str],
                 ks: Sequence[int], sample_for_size: int = 8000,
                 seed: int = 42) -> Dict:
    """Union hits at each K + avg unique candidates per user (sampled) +
    per-source unique incremental hits.

    `users` must be DISTINCT users -- it is the recall denominator and the
    row-count multiplier below. Every source is built from the same target map,
    so sources[0].ranks carries the per-user target counts for all of them.
    """
    out: Dict = {}
    rng = np.random.default_rng(seed)
    sample = (list(rng.choice(users, size=sample_for_size, replace=False))
              if len(users) > sample_for_size else list(users))
    uset = set(users)
    n_pos = sum(len(sources[0].ranks.get(u, ())) for u in users)
    for k in ks:
        hit_sets = {s.name: s.hits_at(k) for s in sources}
        union_hits = set().union(*hit_sets.values())
        pair_hits = {p for s in sources for p in s.hit_pairs_at(k)
                     if p[0] in uset}
        uniq = {
            s.name: len(hit_sets[s.name]
                        - set().union(*(h for n, h in hit_sets.items()
                                        if n != s.name)))
            for s in sources
        }
        sizes = []
        for u in sample:
            cand = set()
            for s in sources:
                cand.update(s.lists.get(u, [])[:k])
            sizes.append(len(cand))
        out[f"K={k}"] = {
            # union_hits / union_recall stay USER-level (>=1 target covered),
            # the frozen meaning every Stage A/B comparison keys off; the
            # _per_positive pair counts are the coverage-of-purchases view.
            "union_hits": len(union_hits),
            "union_recall": len(union_hits) / len(users) if users else 0.0,
            "union_hits_per_positive": len(pair_hits),
            "union_recall_per_positive": len(pair_hits) / n_pos if n_pos else 0.0,
            "per_source_hits": {n: len(h) for n, h in hit_sets.items()},
            "per_source_unique_hits": uniq,
            "avg_unique_candidates_per_user": float(np.mean(sizes)),
            "est_candidate_rows": int(np.mean(sizes) * len(users)),
            "est_ranker_tensor_gb": round(
                np.mean(sizes) * len(users) * 33 * 4 / 1e9, 2),
        }
    return out


def _pairwise_overlap(sources: List[SourceLists], users: Sequence[str],
                      k: int, sample: int = 4000, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    su = (list(rng.choice(users, size=sample, replace=False))
          if len(users) > sample else list(users))
    out = {}
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a, b = sources[i], sources[j]
            ov = [len(set(a.lists.get(u, [])[:k]) & set(b.lists.get(u, [])[:k]))
                  for u in su]
            out[f"{a.name}&{b.name}"] = float(np.mean(ov))
    return out


def _gt_target_sets(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """gt[u] = the SET of that user's target items (mirrors
    train_temporal_ranker._gt_sets / evaluator.build_groundtruth)."""
    s = pd.DataFrame({"u": gt_df["user_id"].astype(str),
                      "i": gt_df["parent_asin"].astype(str)}).drop_duplicates()
    out = {u: set(g) for u, g in s.groupby("u")["i"].agg(set).items()}
    assert sum(len(v) for v in out.values()) == len(s), "groundtruth rows dropped"
    return out


def run_category(
    category: str,
    max_k: int = 500,
    thresholds: Sequence[int] = (5, 3, 2),
    skip_covis: bool = False,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    seed: int = 42,
) -> Dict:
    t_start = time.time()
    tdir = Path(processed_dir) / category / "temporal_ranker"
    manifest = json.loads((tdir / "snapshot_manifest.json").read_text())
    as_of_ms = int(manifest["snapshots"][SNAPSHOT]["history_end_ms"])

    history = pd.read_parquet(tdir / f"history_{SNAPSHOT}.parquet")
    gt = pd.read_parquet(tdir / f"groundtruth_{SNAPSHOT}.parquet")
    item_features = pd.read_parquet(tdir / f"item_features_{SNAPSHOT}.parquet")
    for df, c in ((history, "user_id"), (history, "parent_asin"),
                  (gt, "user_id"), (gt, "parent_asin"),
                  (item_features, "parent_asin")):
        df[c] = df[c].astype(str)

    # targets[u] = the SET of u's positives in the label window; `users` must
    # be DISTINCT users. Per-row construction would duplicate users under
    # multi-positive labels and corrupt every recall denominator and row
    # estimate downstream.
    targets = _gt_target_sets(gt)
    users = sorted(targets)
    n_positives = sum(len(t) for t in targets.values())
    # n_history_events is a per-USER attribute repeated identically on each of
    # that user's label rows, so the last-row-wins zip stays correct here.
    n_hist_map = dict(zip(gt["user_id"], gt["n_history_events"].astype(int)))
    seen = user_seen_from_train(history, users=users)
    hist_item_counts = history["parent_asin"].value_counts()
    hist_items = set(hist_item_counts.index)
    rw = json.loads((tdir / "rule_weights.json").read_text())
    weights = (rw["w_store"], rw["w_cat"], rw["w_pop"])
    ks_report = [k for k in KS_DEFAULT if k <= max_k]

    def cohort(u):
        n = n_hist_map.get(u, 0)
        return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"

    timing: Dict[str, float] = {}

    # ---- base pool (threshold 5) + fast source builders -----------------------
    def build_pool(threshold: int) -> np.ndarray:
        elig = hist_item_counts[hist_item_counts >= threshold].index
        return np.array(sorted(elig.astype(str)))

    def pop_lists(pool_sorted: np.ndarray, pop_series: pd.Series,
                  k: int) -> Dict[str, List[str]]:
        scores = pop_series.reindex(pool_sorted).fillna(0.0).to_numpy(np.float64)
        order = np.argsort(-scores, kind="stable")
        ranked_global = pool_sorted[order]
        out = {}
        for u in users:
            s = seen.get(u, set())
            if not s:
                out[u] = list(ranked_global[:k])
            else:
                out[u] = [it for it in ranked_global if it not in s][:k]
        return out

    def rule_lists(pool_sorted: np.ndarray, k: int,
                   pop_series: pd.Series) -> Dict[str, List[str]]:
        """Vectorized rule scorer: same scores/tie-breaks as
        recommend_rule_based over an alphabetical pool, but O(pool) numpy
        fancy-indexing per user instead of a python generator (mandatory once
        threshold-2 pools reach 1e5 items)."""
        idx = {it: j for j, it in enumerate(pool_sorted)}
        feat = item_features.set_index("parent_asin")
        store_vals = feat.reindex(pool_sorted)["store"].fillna("Unknown").astype(str)
        cat_vals = feat.reindex(pool_sorted)["main_category"].fillna("Unknown").astype(str)
        store_vocab = {v: i for i, v in enumerate(sorted(store_vals.unique()))}
        cat_vocab = {v: i for i, v in enumerate(sorted(cat_vals.unique()))}
        store_idx = store_vals.map(store_vocab).to_numpy(np.int64)
        cat_idx = cat_vals.map(cat_vocab).to_numpy(np.int64)
        prior = pop_series.reindex(pool_sorted).fillna(0.0).to_numpy(np.float64)
        prior = prior / prior.max() if prior.max() > 0 else prior
        w_store, w_cat, w_pop = weights
        base = w_pop * prior
        store_aff = build_user_facet_affinity(history, item_features, "store")
        cat_aff = build_user_facet_affinity(history, item_features, "main_category")
        out = {}
        n_store, n_cat = len(store_vocab), len(cat_vocab)
        for u in users:
            score = base.copy()
            sa = store_aff.get(u)
            if sa:
                vec = np.zeros(n_store)
                for v, a in sa.items():
                    j = store_vocab.get(v)
                    if j is not None:
                        vec[j] = a
                score += w_store * vec[store_idx]
            ca = cat_aff.get(u)
            if ca:
                vec = np.zeros(n_cat)
                for v, a in ca.items():
                    j = cat_vocab.get(v)
                    if j is not None:
                        vec[j] = a
                score += w_cat * vec[cat_idx]
            out[u] = _ranked_by_scores(pool_sorted, score, seen.get(u, set()),
                                       idx, k)
        return out

    t0 = time.time()
    pool5 = build_pool(5)
    lifetime_pop = compute_popularity(history)
    pop_src = SourceLists("popularity", pop_lists(pool5, lifetime_pop, max_k),
                          targets)
    timing["popularity"] = round(time.time() - t0, 1)

    t0 = time.time()
    rule_src = SourceLists("rule", rule_lists(pool5, max_k, lifetime_pop),
                           targets)
    timing["rule"] = round(time.time() - t0, 1)

    t0 = time.time()
    tt_pred_path = tdir / f"two_tower_predictions_{SNAPSHOT}_k{max_k}.parquet"
    if tt_pred_path.exists():
        tt_df = pd.read_parquet(tt_pred_path)
    else:
        from scripts.retrieval.temporal_two_tower import infer_snapshot_checkpoint
        tt_df = infer_snapshot_checkpoint(category, SNAPSHOT, k=max_k,
                                          processed_dir=processed_dir,
                                          out_path=tt_pred_path)
    tt_df["user_id"] = tt_df["user_id"].astype(str)
    tt_df["parent_asin"] = tt_df["parent_asin"].astype(str)
    tt_lists = {u: g.sort_values("rank")["parent_asin"].tolist()
                for u, g in tt_df.groupby("user_id")}
    tt_src = SourceLists("two_tower", tt_lists, targets)
    timing["two_tower"] = round(time.time() - t0, 1)

    base_sources = [pop_src, rule_src, tt_src]

    # ---- 1. K sweep ------------------------------------------------------------
    print(f"[{category}] K sweep ...", flush=True)
    t0 = time.time()
    k_sweep = {
        "singles": {
            s.name: {f"K={k}": {
                "hits": len(s.hits_at(k)),
                "recall": len(s.hits_at(k)) / len(users),
                "recall_per_positive": (len(s.hit_pairs_at(k)) / n_positives
                                        if n_positives else 0.0),
            } for k in ks_report}
            for s in base_sources
        },
        "pop∪rule": _union_stats([pop_src, rule_src], users, ks_report),
        "pop∪tt": _union_stats([pop_src, tt_src], users, ks_report),
        "rule∪tt": _union_stats([rule_src, tt_src], users, ks_report),
        "all_three": _union_stats(base_sources, users, ks_report),
        "pairwise_overlap@100": _pairwise_overlap(base_sources, users, 100),
    }
    timing["k_sweep"] = round(time.time() - t0, 1)

    # ---- 2. eligibility ablation ------------------------------------------------
    print(f"[{category}] eligibility ablation ...", flush=True)
    t0 = time.time()
    ablation = {}
    for thr in thresholds:
        pool_t = build_pool(thr)
        pool_set = set(pool_t)
        pop_t = pop_src if thr == 5 else SourceLists(
            "popularity", pop_lists(pool_t, lifetime_pop, max_k), targets)
        rule_t = rule_src if thr == 5 else SourceLists(
            "rule", rule_lists(pool_t, max_k, lifetime_pop), targets)
        srcs = [pop_t, rule_t, tt_src]          # tt frozen at threshold-5 vocab
        retrieved_per_user = {
            u: set().union(*(s.lists.get(u, [])[:100] for s in srcs))
            for u in users
        }
        status = classify_targets(gt, hist_item_counts, pool_set,
                                  retrieved_per_user)
        # One entry per (user, target) pair, not per user: with several
        # positives per user the histogram sums to the number of targets.
        # Identical to the frozen numbers under one-gt-row-per-user.
        counts = status.value_counts().to_dict()
        pos_counts = lifetime_pop.reindex(pool_t).fillna(0.0)
        top1pct = max(1, int(len(pool_t) * 0.01))
        conc = float(pos_counts.nlargest(top1pct).sum() / max(pos_counts.sum(), 1))
        union100 = _union_stats(srcs, users, [100])["K=100"]
        # The two catalog-coverage ceilings are MACRO (mean over users of the
        # user's own covered fraction), matching the label_item_in_*_rate keys
        # temporal_split emits for exactly these quantities; the _per_positive
        # siblings are the micro rate over all targets. All three coincide, and
        # equal the old `target in pool` mean, while a user has one target.
        ablation[f"min_item_history={thr}"] = {
            "eligible_catalog_size": int(len(pool_t)),
            "historical_catalog_coverage": float(np.mean(
                [len(targets[u] & hist_items) / len(targets[u]) for u in users])),
            "historical_catalog_coverage_per_positive": (
                sum(len(targets[u] & hist_items) for u in users) / n_positives
                if n_positives else 0.0),
            "eligible_pool_coverage": float(np.mean(
                [len(targets[u] & pool_set) / len(targets[u]) for u in users])),
            "eligible_pool_coverage_per_positive": (
                sum(len(targets[u] & pool_set) for u in users) / n_positives
                if n_positives else 0.0),
            "retrieved_coverage@100": union100["union_recall"],
            "retrieved_coverage@100_per_positive":
                union100["union_recall_per_positive"],
            "avg_candidates_per_user@100":
                union100["avg_unique_candidates_per_user"],
            "target_status_counts": {k: int(v) for k, v in counts.items()},
            "popularity_top1pct_share": conc,
            "note_two_tower": "frozen at threshold-5 training vocabulary",
        }
    timing["ablation"] = round(time.time() - t0, 1)

    # ---- 3. recent / decayed popularity -----------------------------------------
    print(f"[{category}] recent/decayed popularity ...", flush=True)
    t0 = time.time()
    base_hits100 = set().union(*(s.hits_at(100) for s in base_sources))
    recent_results = {}
    variants: List[Tuple[str, pd.Series]] = [
        (f"popularity_last_{d}d", windowed_popularity(history, as_of_ms, d))
        for d in (7, 30, 90)
    ] + [
        (f"decayed_hl{hl}", decayed_popularity(history, as_of_ms, hl))
        for hl in (7, 30, 90)
    ]
    for name, series in variants:
        src = SourceLists(name, pop_lists(pool5, series, 100), targets)
        hits = src.hits_at(100)
        recent_results[name] = {
            "standalone_recall@100": len(hits) / len(users),
            "unique_hits_vs_base_union@100": len(hits - base_hits100),
            "n_items_scored": int((series > 0).sum()),
        }
    timing["recent_pop"] = round(time.time() - t0, 1)

    # ---- 4. co-visitation grid ----------------------------------------------------
    covis_results = {}
    if not skip_covis:
        print(f"[{category}] co-visitation ...", flush=True)
        t0 = time.time()
        grid = [
            {"max_seeds": 5, "seed_decay": 0.7, "min_support": 2, "normalization": "cosine"},
            {"max_seeds": 5, "seed_decay": 1.0, "min_support": 2, "normalization": "cosine"},
            {"max_seeds": 3, "seed_decay": 0.7, "min_support": 2, "normalization": "cosine"},
            {"max_seeds": 5, "seed_decay": 0.7, "min_support": 3, "normalization": "cosine"},
            {"max_seeds": 5, "seed_decay": 0.7, "min_support": 2, "normalization": "count"},
        ]
        built: Dict[Tuple[int, str], Dict] = {}
        for g in grid:
            key = (g["min_support"], g["normalization"])
            if key not in built:
                built[key] = build_covisitation(
                    history, as_of_ms, min_support=g["min_support"],
                    normalization=g["normalization"],
                )
            neighbors = built[key]
            lists = recommend_covisitation(
                history, users, neighbors, seen,
                max_seeds=g["max_seeds"], seed_decay=g["seed_decay"], k=100,
            )
            src = SourceLists("covis", lists, targets)
            hits = src.hits_at(100)
            by_cohort = {}
            for c in ("0", "1", "2", "3+"):
                cu = [u for u in users if cohort(u) == c]
                by_cohort[c] = {
                    "n_users": len(cu),
                    "recall@100": (len([u for u in cu if u in hits]) / len(cu))
                    if cu else 0.0,
                }
            # Counted over COVERED (user, target) pairs: `for u in hits` with a
            # scalar targets[u] only asked about the one target a user had.
            low_support_hits = sum(
                1 for _, t in src.hit_pairs_at(100)
                if 0 < hist_item_counts.get(t, 0) < 5
            )
            covis_results[json.dumps(g, sort_keys=True)] = {
                "standalone_recall@100": len(hits) / len(users),
                "unique_hits_vs_base_union@100": len(hits - base_hits100),
                "marginal_union_recall@100":
                    len(hits | base_hits100) / len(users),
                "by_cohort": by_cohort,
                "low_support_target_hits": int(low_support_hits),
                "n_items_with_neighbors": len(neighbors),
            }
        timing["covis"] = round(time.time() - t0, 1)

    # ---- base cohort metrics ------------------------------------------------------
    base_cohorts = {}
    for c in ("0", "1", "2", "3+"):
        cu = [u for u in users if cohort(u) == c]
        base_cohorts[c] = {
            "n_users": len(cu),
            "base_union_recall@100":
                (len([u for u in cu if u in base_hits100]) / len(cu)) if cu else 0.0,
        }

    report = {
        "category": category,
        "snapshot": SNAPSHOT,
        "test_lock": "groundtruth_test never read by this module",
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "n_users": len(users),          # DISTINCT users, == n_positives only
        "n_positives": n_positives,     # under first_positive groundtruth
        "max_k": max_k,
        "rule_weights": dict(zip(("w_store", "w_cat", "w_pop"), weights)),
        "k_sweep": k_sweep,
        "eligibility_ablation": ablation,
        "recent_popularity": recent_results,
        "covisitation": covis_results,
        "base_cohorts": base_cohorts,
        "timing_seconds": timing,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "peak_rss_gb": round(_ru_maxrss_gb(), 2),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{category}_stageA.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{category}] Stage A written -> {out} "
          f"({report['elapsed_seconds']}s, peak {report['peak_rss_gb']}GB)",
          flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--max-k", type=int, default=500)
    p.add_argument("--skip-covis", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_category(args.category, max_k=args.max_k, skip_covis=args.skip_covis)


if __name__ == "__main__":
    main()
