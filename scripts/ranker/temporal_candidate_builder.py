"""Snapshot-independent candidate construction for the temporal ranker.

CLI:
    python -m scripts.ranker.temporal_candidate_builder --category All_Beauty \
        [--snapshot all|ranker_train|model_selection|test] [--top-k 100]

For EACH snapshot independently: popularity Top-K + rule-based Top-K +
two-tower Top-K from that snapshot's own history/checkpoint, unioned into

    data/processed/{cat}/temporal_ranker/candidates_{snapshot}.parquet

One row per (user, candidate item): retrieval indicators/scores/ranks +
snapshot-specific user `_hist` features (incl. the three temporal features) +
snapshot-specific item `_hist` features + cross features + the label from
that snapshot's ground truth. NOTHING is copied across snapshots.

Point-in-time / correctness rules:
  * rule weights are tuned ONCE on the ranker_train snapshot (T0->T1 labels,
    development window) and frozen to rule_weights.json; the model_selection
    and test snapshots reuse them -- T1->T2 labels drive ranker model
    selection and must not also tune retrieval, T2->T3 labels tune nothing.
  * candidate pools are sorted alphabetically before any positional
    tie-breaking, so results are invariant to catalog row order (the
    2023-rating_number ordering leak found in review).
  * categorical missing values are filled BEFORE string conversion
    (the astype(str).fillna bug turned NaN into the literal "nan").
  * no dump-time aggregate (price / average_rating / rating_number) appears
    in any feature column.
  * every ground-truth user gets candidate rows (zero-history users receive
    popularity/rule candidates; two-tower covers >=1-history users only).

Rows are written per user-chunk through one ParquetWriter, so peak memory
stays bounded on the 6GB login-node cgroup.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval.popularity import (
    compute_popularity,
    recommend_popularity,
    user_seen_from_train,
)
from scripts.retrieval.rule_based import (
    build_user_facet_affinity,
    normalized_popularity_prior,
    recommend_rule_based,
    tune_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase2_temporal"
SNAPSHOT_NAMES = ("ranker_train", "model_selection", "test")
KS = (10, 50, 100)

USER_FEATURE_RENAME = {
    "n_reviews_hist": "user_n_reviews_hist",
    "avg_rating_hist": "user_avg_rating_hist",
    "std_rating_hist": "user_std_rating_hist",
    "n_unique_items_hist": "user_n_unique_items_hist",
    "active_days_hist": "user_active_days_hist",
    "verified_rate_hist": "user_verified_rate_hist",
    "days_since_last_interaction": "days_since_last_interaction",
    "interactions_last_30d": "interactions_last_30d",
    "positives_last_30d": "positives_last_30d",
}
ITEM_FEATURE_RENAME = {
    "n_reviews_hist": "item_n_reviews_hist",
    "n_positives_hist": "item_n_positives_hist",
    "avg_rating_hist": "item_avg_rating_hist",
    "n_unique_reviewers_hist": "item_n_unique_reviewers_hist",
    "n_features": "n_features",
    "n_description": "n_description",
    "n_categories": "n_categories",
}

FINAL_COLUMNS = [
    "snapshot", "user_id", "parent_asin", "label",
    "source_popularity", "source_rule", "source_two_tower", "num_sources",
    "popularity_score", "rule_score", "two_tower_score",
    "popularity_rank", "rule_rank", "two_tower_rank", "best_rank",
    "n_history_events", "user_has_history",
    *USER_FEATURE_RENAME.values(),
    *ITEM_FEATURE_RENAME.values(),
    "main_category", "store",
    "user_store_affinity", "user_category_affinity",
    "same_top_store", "same_top_category",
]


def _clean_cat(s: pd.Series) -> pd.Series:
    """Missing-safe categorical cleanup: fill BEFORE str conversion so NaN
    never becomes the literal string 'nan'."""
    s = s.where(s.notna(), "Unknown")
    s = s.astype(str)
    return s.replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})


def _topk_to_long(topk: Dict[str, List[str]], scores: Dict[str, List[float]],
                  source: str) -> pd.DataFrame:
    rows = []
    for u, items in topk.items():
        sc = scores[u]
        for r, (it, s) in enumerate(zip(items, sc)):
            rows.append((u, it, r + 1, float(s)))
    return pd.DataFrame(
        rows, columns=["user_id", "parent_asin", f"{source}_rank", f"{source}_score"],
    )


def _user_top_facet(history: pd.DataFrame, item_facet: pd.Series) -> Dict[str, str]:
    """Most frequent facet value over the user's history POSITIVES."""
    pos = history[history["label"] == 1].copy()
    pos["_f"] = pos["parent_asin"].map(item_facet)
    pos = pos.dropna(subset=["_f"])
    if not len(pos):
        return {}
    counts = pos.groupby(["user_id", "_f"]).size().reset_index(name="n")
    counts = counts.sort_values(["user_id", "n", "_f"],
                                ascending=[True, False, True], kind="mergesort")
    top = counts.drop_duplicates(subset=["user_id"], keep="first")
    return dict(zip(top["user_id"], top["_f"]))


def build_snapshot_candidates(
    category: str,
    snapshot: str,
    top_k: int = 100,
    n_tuning_users: int = 5000,
    seed: int = 42,
    chunk_users: int = 8000,
    processed_dir: Path = PROCESSED_DIR,
    require_two_tower: bool = True,
    min_item_history: Optional[int] = None,
    extra_sources: Optional[List[dict]] = None,
    tt_k: Optional[int] = None,
    out_dir: Optional[Path] = None,
    label_mode: str = "first_positive",
) -> Dict:
    """Build candidates_{snapshot}.parquet + per-source metrics for ONE snapshot.

    label_mode  "first_positive": groundtruth_{snap} (each user's FIRST window
                positive; the frozen Phase 0-3A frame, kept so those numbers
                stay byte-reproducible). "all_positive": groundtruth_all_{snap}
                (every distinct window positive -- the project objective).
                Drives the label column, every coverage/union number AND the
                rule-weight tuning targets. Rule weights are stored per frame
                (rule_weights.json vs rule_weights_all_positive.json) so the
                two objectives never silently share T0-tuned weights.

    Phase 3A extensions (all default-off; defaults reproduce the frozen
    baseline byte-for-byte):
      min_item_history  override the eligibility threshold (None = the
                        item_features in_eligible_pool flag, i.e. threshold 5)
      extra_sources     additional point-in-time sources, each
                        {"name": ..., "type": "windowed_pop"|"decayed_pop"|
                         "covis"|"content_i2i", ...params}; contributes
                        candidates plus source_{name}/{name}_rank/
                        {name}_score columns. content_i2i params:
                        embeddings_path (default {processed}/{cat}/
                        item_text_embeddings.npz), pool ("catalog"|"history"|
                        "eligible", default catalog), min_pool_history (for
                        pool="history", default 1), recency_decay (0.8),
                        max_events (30), top_k (per-source K override for
                        content only, default = global top_k; the D8
                        memory-wall knob). Its stored score is 1 + cosine in
                        [0, 2] (exact-filled for profile-holding users on
                        rows other sources contributed; users without a
                        profile keep the blanket 0.0 fill).
      tt_k              read two_tower_predictions_{snap}_k{tt_k}.parquet
                        (generated via infer_snapshot_checkpoint if missing)
      out_dir           variant output directory (base artifacts untouched)
    """
    t0 = time.time()
    tdir = Path(processed_dir) / category / "temporal_ranker"
    extra_sources = extra_sources or []
    history = pd.read_parquet(tdir / f"history_{snapshot}.parquet")
    if label_mode not in ("first_positive", "all_positive"):
        raise ValueError(f"unknown label_mode {label_mode!r}")
    gt_prefix = "groundtruth_" if label_mode == "first_positive" else "groundtruth_all_"
    gt_df = pd.read_parquet(tdir / f"{gt_prefix}{snapshot}.parquet")
    user_features = pd.read_parquet(tdir / f"user_features_{snapshot}.parquet")
    item_features = pd.read_parquet(tdir / f"item_features_{snapshot}.parquet")

    for df, cols in ((history, ("user_id", "parent_asin")),
                     (gt_df, ("user_id", "parent_asin")),
                     (user_features, ("user_id",)),
                     (item_features, ("parent_asin",))):
        for c in cols:
            df[c] = df[c].astype(str)

    # ---- candidate pool: eligible items, ALPHABETICAL (future-free ties) -----
    if min_item_history is None:
        pool_df = (
            item_features[item_features["in_eligible_pool"] == 1]
            .sort_values("parent_asin", kind="mergesort").reset_index(drop=True)
        )
    else:
        hist_counts_for_pool = history["parent_asin"].value_counts()
        elig = set(hist_counts_for_pool[
            hist_counts_for_pool >= min_item_history].index.astype(str))
        pool_df = (
            item_features[item_features["parent_asin"].isin(elig)]
            .sort_values("parent_asin", kind="mergesort").reset_index(drop=True)
        )
    pool = set(pool_df["parent_asin"])
    candidate_items = pool_df["parent_asin"].to_numpy()
    item_store_arr = _clean_cat(pool_df["store"]).to_numpy()
    item_cat_arr = _clean_cat(pool_df["main_category"]).to_numpy()
    item_pop_prior = normalized_popularity_prior(history, candidate_items)
    pop_prior_map = dict(zip(candidate_items, item_pop_prior.astype(float)))

    pop_counts = compute_popularity(history)
    pop_count_map = {pa: float(pop_counts.get(pa, 0.0)) for pa in pool}

    user_store_aff = build_user_facet_affinity(history, item_features, "store")
    user_cat_aff = build_user_facet_affinity(history, item_features, "main_category")

    # ---- rule weights: tune at T0 once, frozen afterwards ---------------------
    # The frozen file is keyed to the resolved cutoffs: a rerun with different
    # T0/T1 (e.g. explicit dates pinned after a quantile run) must NOT silently
    # reuse weights that were tuned on labels now inside a later window.
    manifest = json.loads((tdir / "snapshot_manifest.json").read_text())
    cutoff_fingerprint = {
        "t0_ms": manifest["cutoffs"]["t0_ms"],
        "t1_ms": manifest["cutoffs"]["t1_ms"],
        "t2_ms": manifest["cutoffs"]["t2_ms"],
        "t3_ms": manifest["cutoffs"]["t3_ms"],
    }
    weights_path = tdir / ("rule_weights.json" if label_mode == "first_positive"
                           else "rule_weights_all_positive.json")
    tuning_payload = None
    stale_weights = False
    if weights_path.exists():
        w = json.loads(weights_path.read_text())
        if w.get("cutoff_fingerprint") != cutoff_fingerprint:
            if snapshot == "ranker_train":
                stale_weights = True
                print(f"[cand-{snapshot}] rule_weights.json is stale (cutoffs "
                      f"changed) -- re-tuning on the new T0 window", flush=True)
            else:
                raise RuntimeError(
                    f"{weights_path} was tuned under different cutoffs "
                    f"({w.get('cutoff_fingerprint')} != {cutoff_fingerprint}). "
                    f"Rebuild the ranker_train snapshot first so weights are "
                    f"re-tuned on the current T0 window."
                )
        else:
            weights = (w["w_store"], w["w_cat"], w["w_pop"])
    if not weights_path.exists() or stale_weights:
        if snapshot != "ranker_train":
            raise FileNotFoundError(
                f"{weights_path} not found -- build the ranker_train snapshot "
                f"first (rule weights are tuned on T0 and frozen)"
            )
        rng = np.random.RandomState(seed)
        gt_users_all = sorted(gt_df["user_id"].unique())
        tuning_users = (
            list(rng.choice(gt_users_all, size=n_tuning_users, replace=False))
            if len(gt_users_all) > n_tuning_users else gt_users_all
        )
        seen_for_tuning = user_seen_from_train(history, users=tuning_users)
        print(f"[cand-{snapshot}] tuning rule weights on "
              f"{len(tuning_users)} T0 users...", flush=True)
        tune = tune_weights(
            val_df=gt_df, user_ids_for_tuning=tuning_users,
            candidate_items=candidate_items, item_store_arr=item_store_arr,
            item_cat_arr=item_cat_arr, item_pop_prior=item_pop_prior,
            user_store_aff=user_store_aff, user_cat_aff=user_cat_aff,
            user_seen=seen_for_tuning, k_for_tuning=max(KS),
        )
        weights = tune.best_weights
        tuning_payload = {"best_weights": {"w_store": weights[0],
                                           "w_cat": weights[1],
                                           "w_pop": weights[2]},
                          "best_recall": tune.best_recall,
                          "n_tuning_users": len(tuning_users)}
        with open(weights_path, "w") as f:
            json.dump({"w_store": weights[0], "w_cat": weights[1],
                       "w_pop": weights[2],
                       "tuned_on": "ranker_train (T0->T1 development labels)",
                       "cutoff_fingerprint": cutoff_fingerprint,
                       "seed": seed, "best_recall": tune.best_recall}, f, indent=2)
        print(f"[cand-{snapshot}] frozen rule weights {weights}", flush=True)

    # ---- two-tower predictions -------------------------------------------------
    if tt_k:
        tt_path = tdir / f"two_tower_predictions_{snapshot}_k{tt_k}.parquet"
        if not tt_path.exists():
            from scripts.retrieval.temporal_two_tower import (
                infer_snapshot_checkpoint,
            )
            print(f"[cand-{snapshot}] generating two-tower predictions at "
                  f"K={tt_k} from the frozen checkpoint...", flush=True)
            infer_snapshot_checkpoint(category, snapshot, k=tt_k,
                                      processed_dir=processed_dir,
                                      out_path=tt_path)
    else:
        tt_path = tdir / f"two_tower_predictions_{snapshot}.parquet"
    if tt_path.exists():
        tt_df = pd.read_parquet(tt_path)
        tt_df["user_id"] = tt_df["user_id"].astype(str)
        tt_df["parent_asin"] = tt_df["parent_asin"].astype(str)
        tt_df = tt_df[tt_df["rank"] <= top_k]
        tt_df = tt_df[np.isfinite(tt_df["score"])]
        tt_by_user = dict(tuple(tt_df.groupby("user_id")))
        tt_score_fill = float(tt_df["score"].min()) - 1.0 if len(tt_df) else 0.0
    elif require_two_tower:
        raise FileNotFoundError(
            f"{tt_path} not found -- run scripts.retrieval.temporal_two_tower first"
        )
    else:
        tt_by_user, tt_score_fill = {}, 0.0

    # ---- static lookups ----------------------------------------------------------
    hist_item_set = set(history["parent_asin"].unique())
    item_store_map = dict(zip(item_features["parent_asin"],
                              _clean_cat(item_features["store"])))
    item_cat_map = dict(zip(item_features["parent_asin"],
                            _clean_cat(item_features["main_category"])))
    top_store_map = _user_top_facet(history, pd.Series(item_store_map))
    top_cat_map = _user_top_facet(history, pd.Series(item_cat_map))

    item_feat_block = item_features.set_index("parent_asin")[
        list(ITEM_FEATURE_RENAME)
    ].rename(columns=ITEM_FEATURE_RENAME)
    user_feat_block = user_features.set_index("user_id")[
        list(USER_FEATURE_RENAME)
    ].rename(columns=USER_FEATURE_RENAME)

    # Set-valued: dict(zip(...)) is last-wins, so with several groundtruth
    # rows per user every target but the latest silently disappears from the
    # labels AND from every coverage number below.
    gt_map: Dict[str, Set[str]] = (
        gt_df.groupby("user_id")["parent_asin"].agg(set).to_dict())
    assert sum(len(v) for v in gt_map.values()) == len(
        gt_df.drop_duplicates(["user_id", "parent_asin"])), "groundtruth rows dropped"
    # n_history_events is constant within a user, so last-wins is harmless here.
    n_hist_map = dict(zip(gt_df["user_id"], gt_df["n_history_events"].astype(int)))
    eval_users = sorted(gt_map)
    n_positives_total = sum(len(v) for v in gt_map.values())

    # ---- extra point-in-time sources (Phase 3A) --------------------------------
    as_of_ms = int(manifest["snapshots"][snapshot]["history_end_ms"])
    extra_prepared = []          # (name, kind, payload)
    if extra_sources:
        from scripts.retrieval.temporal_sources import (
            build_content_profiles,
            build_covisitation,
            decayed_popularity,
            recommend_content_knn,
            recommend_covisitation,
            windowed_popularity,
        )
        for src in extra_sources:
            name, kind = src["name"], src["type"]
            if kind == "windowed_pop":
                series = windowed_popularity(history, as_of_ms,
                                             int(src["window_days"]))
                extra_prepared.append((name, kind, series))
            elif kind == "decayed_pop":
                series = decayed_popularity(history, as_of_ms,
                                            float(src["half_life_days"]))
                extra_prepared.append((name, kind, series))
            elif kind == "covis":
                neighbors = build_covisitation(
                    history, as_of_ms,
                    min_support=int(src.get("min_support", 2)),
                    normalization=src.get("normalization", "cosine"),
                    max_basket_items=int(src.get("max_basket_items", 30)),
                )
                extra_prepared.append((name, kind, {
                    "neighbors": neighbors,
                    "max_seeds": int(src.get("max_seeds", 5)),
                    "seed_decay": float(src.get("seed_decay", 0.7)),
                    "max_neighbors_per_seed":
                        int(src.get("max_neighbors_per_seed", 50)),
                }))
            elif kind == "content_i2i":
                emb_path = (Path(src["embeddings_path"])
                            if src.get("embeddings_path")
                            else Path(processed_dir) / category
                            / "item_text_embeddings.npz")
                if not emb_path.exists():
                    raise FileNotFoundError(
                        f"{emb_path} not found -- content_i2i needs the "
                        f"aligned item text embeddings")
                npz = np.load(emb_path, allow_pickle=True)
                emb_asins = np.asarray(npz["asins"]).astype(str)
                embs = np.asarray(npz["embs"], dtype=np.float32)
                if embs.ndim != 2 or embs.shape[0] != len(emb_asins) \
                        or not len(emb_asins):
                    raise ValueError(
                        f"{emb_path} unusable: asins {emb_asins.shape} vs "
                        f"embs {embs.shape} (text_alignment 'not_present'?)")
                norms = np.linalg.norm(embs, axis=1)
                has_emb = norms > 0.0     # alignment zero-fills missing items
                if not has_emb.any():
                    raise ValueError(f"{emb_path} holds only zero embeddings")
                embs /= np.where(has_emb, norms, 1.0)[:, None]  # unit/zero rows
                emb_index = {a: j for j, (a, ok) in
                             enumerate(zip(emb_asins, has_emb)) if ok}
                assert len(emb_index) == int(has_emb.sum()), \
                    "duplicate parent_asin keys in the embedding npz"
                pool_mode = src.get("pool", "catalog")
                if pool_mode == "catalog":
                    keep = has_emb
                elif pool_mode == "history":
                    hc = history["parent_asin"].value_counts()
                    min_h = int(src.get("min_pool_history", 1))
                    ok_items = set(hc[hc >= min_h].index.astype(str))
                    keep = has_emb & np.array(
                        [a in ok_items for a in emb_asins])
                elif pool_mode == "eligible":
                    keep = has_emb & np.array([a in pool for a in emb_asins])
                else:
                    raise ValueError(
                        f"unknown content_i2i pool {pool_mode!r} "
                        f"(expected catalog|history|eligible)")
                order = np.argsort(emb_asins[keep], kind="stable")
                profiles = build_content_profiles(
                    history, as_of_ms, embs, emb_index,
                    recency_decay=float(src.get("recency_decay", 0.8)),
                    max_events=int(src.get("max_events", 30)),
                )
                prof_users = sorted(profiles)
                extra_prepared.append((name, kind, {
                    "k": int(src.get("top_k", top_k)),
                    "profiles": profiles,
                    "candidate_items": emb_asins[keep][order],
                    "candidate_matrix": embs[keep][order],
                    "norm_embs": embs,
                    "emb_index": emb_index,
                    "prof_index": {u: j for j, u in enumerate(prof_users)},
                    "prof_matrix": (
                        np.stack([profiles[u] for u in prof_users])
                        if prof_users
                        else np.zeros((0, embs.shape[1]), dtype=np.float32)),
                }))
            else:
                raise ValueError(f"unknown extra source type {kind!r}")
    extra_names = [name for name, _, _ in extra_prepared]

    # ---- per-source metric accumulators ----------------------------------------
    all_source_names = ["popularity", "rule", "two_tower", *extra_names]
    # Float accumulators: each user contributes the FRACTION of their positives
    # a source retrieved, so the per-source figures stay macro recalls rather
    # than becoming hit counts once a user has several targets.
    src_hits = {s: {k: 0.0 for k in KS} for s in all_source_names}   # per-user recall sums
    src_prec = {s: {k: 0.0 for k in KS} for s in all_source_names}   # per-user hits@k/k sums
    retrieved_cov_n = 0        # users with >= 1 positive in the union
    retrieved_cov_macro = 0.0  # sum over users of covered-fraction
    retrieved_cov_pos = 0      # positives covered, pooled
    dest_dir = Path(out_dir) if out_dir is not None else tdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"candidates_{snapshot}.parquet"
    writer: Optional[pq.ParquetWriter] = None
    n_rows_total = 0
    n_pos_rows = 0

    for start in range(0, len(eval_users), chunk_users):
        chunk = eval_users[start:start + chunk_users]
        seen = user_seen_from_train(history, users=chunk)

        pop_topk = recommend_popularity(pop_counts, pool, chunk, seen, k=top_k)
        pop_scores = {u: [pop_count_map.get(it, 0.0) for it in items]
                      for u, items in pop_topk.items()}
        rule_topk = recommend_rule_based(
            user_ids=chunk, candidate_items=candidate_items,
            item_store_arr=item_store_arr, item_cat_arr=item_cat_arr,
            item_pop_prior=item_pop_prior, user_store_aff=user_store_aff,
            user_cat_aff=user_cat_aff, user_seen=seen,
            weights=weights, k=top_k,
        )
        # recommend_rule_based's k >= pool_size path (plain argsort) keeps
        # seen items at the -inf tail; strip them so seen items never become
        # candidates when the pool is small.
        for u in chunk:
            s = seen.get(u)
            if s:
                rule_topk[u] = [it for it in rule_topk[u] if it not in s]
        w_store, w_cat, w_pop = weights
        rule_scores = {}
        for u, items in rule_topk.items():
            s_aff = user_store_aff.get(u, {})
            c_aff = user_cat_aff.get(u, {})
            rule_scores[u] = [
                w_store * s_aff.get(item_store_map.get(it, "Unknown"), 0.0)
                + w_cat * c_aff.get(item_cat_map.get(it, "Unknown"), 0.0)
                + w_pop * pop_prior_map.get(it, 0.0)
                for it in items
            ]

        pop_long = _topk_to_long(pop_topk, pop_scores, "popularity")
        rule_long = _topk_to_long(rule_topk, rule_scores, "rule")
        tt_parts = [tt_by_user[u] for u in chunk if u in tt_by_user]
        if tt_parts:
            tt_long = pd.concat(tt_parts, ignore_index=True)[
                ["user_id", "parent_asin", "rank", "score"]
            ].rename(columns={"rank": "two_tower_rank", "score": "two_tower_score"})
        else:
            tt_long = pd.DataFrame(columns=["user_id", "parent_asin",
                                            "two_tower_rank", "two_tower_score"])

        merged = pop_long.merge(rule_long, on=["user_id", "parent_asin"], how="outer")
        merged = merged.merge(tt_long, on=["user_id", "parent_asin"], how="outer")

        # ---- Phase 3A extra sources for this chunk ---------------------------
        extra_chunk_topk: Dict[str, Dict[str, List[str]]] = {}
        for name, kind, payload in extra_prepared:
            if kind in ("windowed_pop", "decayed_pop"):
                topk_e = recommend_popularity(payload, pool, chunk, seen,
                                              k=top_k)
                score_map = payload.to_dict()
                scores_e = {u: [float(score_map.get(it, 0.0)) for it in items]
                            for u, items in topk_e.items()}
            elif kind == "content_i2i":
                topk_e, cos_e = recommend_content_knn(
                    chunk, payload["profiles"], payload["candidate_items"],
                    payload["candidate_matrix"], seen, k=payload["k"],
                    return_scores=True,
                )
                # Stored score = 1 + cosine, shifted into [0, 2]: rank-
                # preserving and non-negative, so resolve_feature_list's
                # log1p(max(x, 0)) never truncates negative cosines. 0.0
                # stays the unserved-user fill (see the fill block below).
                scores_e = {u: [1.0 + s for s in ss]
                            for u, ss in cos_e.items()}
            else:   # covis (the prepare loop rejected every other kind)
                topk_e, scores_e = recommend_covisitation(
                    history, chunk, payload["neighbors"], seen,
                    max_seeds=payload["max_seeds"],
                    seed_decay=payload["seed_decay"],
                    max_neighbors_per_seed=payload["max_neighbors_per_seed"],
                    k=top_k, return_scores=True,
                )
            extra_chunk_topk[name] = topk_e
            e_long = _topk_to_long(topk_e, scores_e, name)
            merged = merged.merge(e_long, on=["user_id", "parent_asin"],
                                  how="outer")

        for s in all_source_names:
            merged[f"source_{s}"] = merged[f"{s}_rank"].notna().astype(np.int8)
            merged[f"{s}_rank"] = merged[f"{s}_rank"].fillna(0).astype(np.int32)
        merged["num_sources"] = sum(
            merged[f"source_{s}"] for s in all_source_names
        ).astype(np.int8)
        rank_cols = [f"{s}_rank" for s in all_source_names]
        rank_arr = merged[rank_cols].to_numpy()
        rank_masked = np.where(rank_arr > 0, rank_arr, np.iinfo(np.int32).max)
        best = rank_masked.min(axis=1)
        best[best == np.iinfo(np.int32).max] = 0
        merged["best_rank"] = best.astype(np.int32)
        for name, kind, payload in extra_prepared:
            if kind == "content_i2i":
                # Exact fill for rows OTHER sources contributed: users with a
                # profile get 1 + cos(profile, item) -- the same shifted scale
                # the source emits, so within-user comparisons stay on one
                # scale. Unserved users (no profile) and items without an
                # embedding fall through to the blanket 0.0 below.
                col = f"{name}_score"
                miss = merged[col].isna().to_numpy()
                if miss.any():
                    rows = np.flatnonzero(miss)
                    prof_index = payload["prof_index"]
                    emb_index = payload["emb_index"]
                    pidx = np.array(
                        [prof_index.get(u, -1)
                         for u in merged["user_id"].to_numpy()[rows]],
                        dtype=np.int64)
                    iidx = np.array(
                        [emb_index.get(it, -1)
                         for it in merged["parent_asin"].to_numpy()[rows]],
                        dtype=np.int64)
                    ok = (pidx >= 0) & (iidx >= 0)
                    if ok.any():
                        cos = np.einsum(
                            "ij,ij->i",
                            payload["prof_matrix"][pidx[ok]],
                            payload["norm_embs"][iidx[ok]],
                        )
                        fill = np.full(len(merged), np.nan, dtype=np.float32)
                        fill[rows[ok]] = 1.0 + cos
                        merged[col] = merged[col].fillna(
                            pd.Series(fill, index=merged.index))
            merged[f"{name}_score"] = (
                merged[f"{name}_score"].fillna(0.0).astype(np.float32))

        # Score fills (missing-source rows).
        merged["popularity_score"] = merged["popularity_score"].fillna(
            merged["parent_asin"].map(pop_count_map)
        ).fillna(0.0).astype(np.float32)
        merged["_store"] = merged["parent_asin"].map(item_store_map).fillna("Unknown")
        merged["_cat"] = merged["parent_asin"].map(item_cat_map).fillna("Unknown")
        aff_s = np.array([
            user_store_aff.get(u, {}).get(st, 0.0)
            for u, st in zip(merged["user_id"], merged["_store"])
        ], dtype=np.float32)
        aff_c = np.array([
            user_cat_aff.get(u, {}).get(ct, 0.0)
            for u, ct in zip(merged["user_id"], merged["_cat"])
        ], dtype=np.float32)
        prior = merged["parent_asin"].map(pop_prior_map).fillna(0.0).to_numpy(np.float32)
        rule_fill = w_store * aff_s + w_cat * aff_c + w_pop * prior
        merged["rule_score"] = merged["rule_score"].fillna(
            pd.Series(rule_fill, index=merged.index)
        ).astype(np.float32)
        merged["two_tower_score"] = merged["two_tower_score"].fillna(
            tt_score_fill
        ).astype(np.float32)

        # Cross features.
        merged["user_store_affinity"] = aff_s
        merged["user_category_affinity"] = aff_c
        merged["same_top_store"] = np.array([
            1 if top_store_map.get(u) == st else 0
            for u, st in zip(merged["user_id"], merged["_store"])
        ], dtype=np.int8)
        merged["same_top_category"] = np.array([
            1 if top_cat_map.get(u) == ct else 0
            for u, ct in zip(merged["user_id"], merged["_cat"])
        ], dtype=np.int8)
        merged = merged.rename(columns={"_store": "store", "_cat": "main_category"})

        # Feature blocks.
        merged = merged.merge(user_feat_block, left_on="user_id",
                              right_index=True, how="left")
        merged["user_has_history"] = merged["user_n_reviews_hist"].notna().astype(np.int8)
        for c in USER_FEATURE_RENAME.values():
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).astype(np.float32)
        merged = merged.merge(item_feat_block, left_on="parent_asin",
                              right_index=True, how="left")
        for c in ITEM_FEATURE_RENAME.values():
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).astype(np.float32)

        # Label + cohort.
        merged["label"] = np.array([
            1 if pa in gt_map.get(u, ()) else 0
            for u, pa in zip(merged["user_id"], merged["parent_asin"])
        ], dtype=np.int8)
        merged["n_history_events"] = (
            merged["user_id"].map(n_hist_map).fillna(0).astype(np.int32)
        )
        merged["snapshot"] = snapshot

        merged = merged.sort_values(["user_id", "parent_asin"],
                                    kind="mergesort").reset_index(drop=True)
        chunk_columns = FINAL_COLUMNS + [
            col for name in extra_names
            for col in (f"source_{name}", f"{name}_rank", f"{name}_score")
        ]
        merged = merged[chunk_columns]

        # ---- metric accumulation (per user, single positive) -----------------
        by_user_items: Dict[str, Set[str]] = {
            u: set(g) for u, g in merged.groupby("user_id")["parent_asin"].agg(set).items()
        }
        for u in chunk:
            targets = gt_map[u]
            n_t = len(targets)
            for src, topk_dict in (("popularity", pop_topk), ("rule", rule_topk)):
                items = topk_dict.get(u, [])
                for k in KS:
                    h = len(targets & set(items[:k]))
                    src_hits[src][k] += h / n_t
                    src_prec[src][k] += h / k
            tt_items = (tt_by_user[u]["parent_asin"].tolist()[:top_k]
                        if u in tt_by_user else [])
            for k in KS:
                h = len(targets & set(tt_items[:k]))
                src_hits["two_tower"][k] += h / n_t
                src_prec["two_tower"][k] += h / k
            for name in extra_names:
                items = extra_chunk_topk[name].get(u, [])
                for k in KS:
                    h = len(targets & set(items[:k]))
                    src_hits[name][k] += h / n_t
                    src_prec[name][k] += h / k
            cand_set = by_user_items.get(u, set())
            hit_n = len(targets & cand_set)
            if hit_n:
                retrieved_cov_n += 1
            retrieved_cov_macro += hit_n / n_t
            retrieved_cov_pos += hit_n

        n_rows_total += len(merged)
        n_pos_rows += int(merged["label"].sum())
        table = pa.Table.from_pandas(merged, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        print(f"[cand-{snapshot}] users {start + len(chunk):,}/{len(eval_users):,} "
              f"rows={n_rows_total:,}", flush=True)

    if writer is not None:
        writer.close()
    else:
        # Zero ground-truth users: still materialize an empty, schema-correct
        # parquet so downstream stages fail informatively instead of on a
        # missing file.
        empty = pd.DataFrame({c: pd.Series(dtype="object" if c in
                              ("snapshot", "user_id", "parent_asin",
                               "main_category", "store") else "float64")
                              for c in FINAL_COLUMNS})
        empty.to_parquet(out_path, index=False)

    n_users = len(eval_users)
    n_cold_users = sum(1 for u in eval_users if n_hist_map.get(u, 0) == 0)
    cohort_counts: Dict[str, int] = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for u in eval_users:
        n = n_hist_map.get(u, 0)
        cohort_counts["0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"] += 1

    def _src_metrics(src: str) -> Dict[str, float]:
        # Only report Ks the retrieval depth actually supports: with
        # top_k < K, "Recall@K" would silently mean Recall@top_k.
        out = {}
        for k in (kk for kk in KS if kk <= top_k):
            out[f"Recall@{k}"] = src_hits[src][k] / n_users if n_users else 0.0
            # True multi-positive precision (mean over users of hits@k / k).
            # The old form divided the recall accumulator by k, mechanically
            # recreating Precision ~= Recall/K (review 2026-08-25, smaller #2).
            out[f"Precision@{k}"] = src_prec[src][k] / n_users if n_users else 0.0
        return out

    report = {
        "snapshot": snapshot,
        "n_positive_eval_users": n_users,
        "candidate_catalog_size": len(pool),
        "coverage": {
            # Macro (mean over users of the fraction of THEIR positives that
            # are covered) -- the form that bounds grouped_eval's macro recall.
            "historical_catalog": float(np.mean(
                [len(gt_map[u] & hist_item_set) / len(gt_map[u])
                 for u in eval_users])) if n_users else 0.0,
            "eligible_pool": float(np.mean(
                [len(gt_map[u] & pool) / len(gt_map[u])
                 for u in eval_users])) if n_users else 0.0,
            "retrieved_union": retrieved_cov_macro / n_users if n_users else 0.0,
            # Any-hit (fraction of users with >=1 positive covered) and micro
            # (fraction of all positives covered). Under one positive per user
            # all three collapse to the same number.
            "retrieved_union_any_hit": retrieved_cov_n / n_users if n_users else 0.0,
            "retrieved_union_per_positive": (
                retrieved_cov_pos / n_positives_total) if n_positives_total else 0.0,
        },
        "cold_users": n_cold_users,
        # Counts POSITIVES whose item never appears in history (this is what
        # the field always meant; under one row per user it equalled a user
        # count by coincidence).
        "n_positives": n_positives_total,
        "cold_items_not_in_history": int(
            (gt_df.drop_duplicates(["user_id", "parent_asin"])["item_in_history"]
             .astype(int) == 0).sum()),
        "history_cohorts": cohort_counts,
        "sources": {s: _src_metrics(s) for s in all_source_names},
        "retrieval_config": {
            "top_k": top_k,
            "min_item_history": min_item_history,
            "tt_k": tt_k,
            "extra_sources": extra_sources,
        },
        "rule_weights": {"w_store": weights[0], "w_cat": weights[1],
                         "w_pop": weights[2]},
        "rule_tuning": tuning_payload,
        "n_rows": n_rows_total,
        "n_positive_rows": n_pos_rows,
        "top_k": top_k,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    print(f"[cand-{snapshot}] done: rows={n_rows_total:,} "
          f"retrieved_coverage={report['coverage']['retrieved_union']:.4f}", flush=True)
    return report


def build_all(category: str, top_k: int = 100, seed: int = 42,
              processed_dir: Path = PROCESSED_DIR,
              results_dir: Path = RESULTS_DIR,
              require_two_tower: bool = True,
              retrieval_config: Optional[dict] = None,
              variant: Optional[str] = None,
              snapshots: Sequence[str] = SNAPSHOT_NAMES,
              label_mode: str = "first_positive") -> Dict:
    """ranker_train first (tunes + freezes rule weights), then the rest.

    `retrieval_config` (Phase 3A): {top_k, min_item_history, tt_k,
    extra_sources}. `variant` writes candidates to
    temporal_ranker/variants/{variant}/ so baseline artifacts stay untouched.
    `snapshots` restricts which snapshots are built (Stage B builds only
    ranker_train + model_selection; the locked test snapshot is built only
    after freezing)."""
    t0 = time.time()
    rc = retrieval_config or {}
    out_dir = None
    if variant:
        out_dir = (Path(processed_dir) / category / "temporal_ranker"
                   / "variants" / variant)
    reports = {}
    for snap in snapshots:
        reports[snap] = build_snapshot_candidates(
            category, snap,
            top_k=int(rc.get("top_k", top_k)),
            seed=seed,
            processed_dir=processed_dir, require_two_tower=require_two_tower,
            min_item_history=rc.get("min_item_history"),
            extra_sources=rc.get("extra_sources"),
            tt_k=rc.get("tt_k"),
            out_dir=out_dir,
            label_mode=label_mode,
        )
    payload = {
        "category": category,
        "variant": variant,
        "label_mode": label_mode,
        "retrieval_config": rc or None,
        "built_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "snapshots": reports,
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    out_path = results_dir / f"{category}_candidates_report{suffix}.json"
    # MERGE, never clobber (review 2026-08-26 #2): a --snapshots test run used
    # to overwrite the whole report, destroying the Stage-A ranker_train /
    # model_selection provenance the freeze's row-count tie-breaker reads.
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            merged_snaps = dict(prev.get("snapshots") or {})
            merged_snaps.update(payload["snapshots"])
            payload["snapshots"] = merged_snaps
            hist = list(prev.get("build_history") or [])
        except Exception:
            hist = []
    else:
        hist = []
    hist.append({"built_utc": payload["built_utc"],
                 "snapshots": sorted(reports)})
    payload["build_history"] = hist
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--snapshot", default="all", choices=["all", *SNAPSHOT_NAMES])
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", default=None,
                   help="Variant name; candidates go to temporal_ranker/"
                        "variants/{variant}/ (baseline untouched).")
    p.add_argument("--config-json", default=None,
                   help="Path to a retrieval-config JSON: {top_k, "
                        "min_item_history, tt_k, extra_sources}.")
    p.add_argument("--label-mode", default="first_positive",
                   choices=["first_positive", "all_positive"],
                   help="Ground-truth frame for labels / coverage / rule tuning. "
                        "all_positive = the project objective (every window "
                        "positive); first_positive only for reproducing frozen "
                        "Phase 0-3A numbers.")
    p.add_argument("--snapshots", default=None,
                   help="Comma-separated subset, e.g. "
                        "'ranker_train,model_selection' for Stage B.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    rc = json.loads(Path(args.config_json).read_text()) if args.config_json else None
    snaps = tuple(args.snapshots.split(",")) if args.snapshots else SNAPSHOT_NAMES
    if args.snapshot == "all":
        build_all(args.category, top_k=args.top_k, seed=args.seed,
                  retrieval_config=rc, variant=args.variant, snapshots=snaps,
                  label_mode=args.label_mode)
    else:
        build_snapshot_candidates(args.category, args.snapshot,
                                  top_k=args.top_k, seed=args.seed)


if __name__ == "__main__":
    main()
