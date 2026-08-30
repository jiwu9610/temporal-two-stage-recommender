"""Periodic two-tower retraining on walk-forward snapshots (Scheme B).

CLI:
    python -m scripts.retrieval.temporal_two_tower --category All_Beauty \
        [--snapshot all|ranker_train|model_selection|test] [--smoke]

Three independent checkpoints, retrained FROM SCRATCH at each cutoff with the
same frozen architecture + hyperparameters:

    A  ranker_train      trains on events ts < T0, predicts snapshot T0.
                         The ONLY checkpoint allowed to early-stop / pick its
                         epoch count -- against the T0->T1 labels (development
                         window). Its best epoch is FROZEN to
                         two_tower_frozen_config.json.
    B  model_selection   trains on events ts < T1 for exactly the frozen epoch
                         count. No selection against T1->T2 labels (those
                         labels drive ranker model selection downstream).
    C  test              trains on events ts < T2, frozen config, no selection.
                         T2->T3 labels influence nothing.

Point-in-time rules enforced here:
  * training pairs / vocabularies / soft negatives / seen-masks come from the
    snapshot's history and eligible catalog only -- no future IDs get
    embeddings;
  * user/item dense features come from the matching snapshot feature store
    (`_hist` aggregates + the three basic temporal features); the dump's
    average_rating / rating_number / price are NEVER used;
  * predictions are emitted only for label users with >= 1 history event.
    Zero-history (future-only) users get NO two-tower row -- their explicit
    fallback is the popularity source inside the candidate union, and they are
    reported in the 0-interaction cohort.

Each checkpoint saves: state dict (.pt), metadata JSON (training boundary,
seed, dataset composition, config + config_hash -- hash must be identical
across A/B/C), and a per-user top-K predictions parquet.

v2 -- all default-off; defaults reproduce v1:
  * selection/reporting labels come from groundtruth_all_{snapshot}.parquet
    (all-positive) with a HARD-WARNING fallback to the single-positive file;
    metrics are dual-reported (all-positive + single-positive gt);
  * taste channel (cfg.use_hist_pool): user_hist_idx/user_hist_w built here
    from the snapshot history (positive, in-pool, most recent L);
  * loss_mode='softmax' dispatches to train_one_epoch_softmax (in-batch
    sampled softmax + exact-unigram logQ + uniform MNS tail), with the
    decoupled hard-negative pool as an auxiliary lambda-weighted BCE term;
  * pair_recency_decay_days weights positive pairs by recency against the
    snapshot boundary (as_of_ms = history_end_ms);
  * deeper_category's raw-metadata read is gated by cfg.use_deeper_cat_emb
    (kept for v1 checkpoint re-export through infer_snapshot_checkpoint);
  * pre-v2 frozen configs (missing loss_mode/optimizer) are refused for
    frozen-schedule retraining (validate_frozen_config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.retrieval.evaluator import build_split_report
from scripts.retrieval.two_tower import (
    ItemTower,
    TwoTower,
    TwoTowerConfig,
    UserTower,
    build_optimizer,
)
from scripts.retrieval.two_tower_dataset import (
    MS_PER_DAY,
    FeatureSpec,
    PairSamplingConfig,
    TwoTowerPairDataset,
    _build_vocab,
    _categories_path,
    _coerce_dense,
)
from scripts.retrieval.train_two_tower import (
    TrainConfig,
    _move_spec_to_tensors,
    encode_all_items,
    encode_users_subset,
    topk_per_user_chunked,
    train_one_epoch,
    train_one_epoch_softmax,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
SNAPSHOT_NAMES = ("ranker_train", "model_selection", "test")
DEFAULT_KS = (10, 50, 100)
DEFAULT_K_RETRIEVE = 100

# Snapshot-store dense columns. `_hist` aggregates + the three basic temporal
# features. price / average_rating / rating_number are deliberately absent
# (dump-time whole-timeline statistics = future information at any cutoff).
USER_DENSE_COLS_HIST = (
    "n_reviews_hist",
    "avg_rating_hist",
    "std_rating_hist",
    "n_unique_items_hist",
    "active_days_hist",
    "verified_rate_hist",
    "days_since_last_interaction",
    "interactions_last_30d",
    "positives_last_30d",
)
LOG1P_USER_COLS_HIST = (
    "n_reviews_hist", "n_unique_items_hist", "active_days_hist",
    "days_since_last_interaction", "interactions_last_30d", "positives_last_30d",
)
ITEM_DENSE_COLS_HIST = (
    "n_reviews_hist",
    "n_positives_hist",
    "avg_rating_hist",
    "n_unique_reviewers_hist",
    "n_features",
    "n_description",
    "n_categories",
)
LOG1P_ITEM_COLS_HIST = (
    "n_reviews_hist", "n_positives_hist", "n_unique_reviewers_hist",
    "n_features", "n_description", "n_categories",
)


def build_temporal_feature_spec(
    history_df: pd.DataFrame,
    user_features_hist: pd.DataFrame,
    item_features_hist: pd.DataFrame,
    raw_metadata: Optional[pd.DataFrame] = None,
    use_deeper_cat: bool = True,
    build_hist_pool: bool = False,
    hist_pool_len: int = 20,
    hist_pool_recency_days: Optional[float] = None,
    boundary_ms: Optional[int] = None,
) -> FeatureSpec:
    """Snapshot-scoped FeatureSpec: vocabularies from the snapshot's history
    users + eligible catalog, dense tables from the snapshot feature stores.
    Future users/items cannot enter any vocabulary because the inputs are all
    history-bounded artifacts.

    v2 taste channel (`build_hist_pool=True`): additionally builds
    user_hist_idx/user_hist_w -- per user, the `hist_pool_len` most recent
    POSITIVE (label==1) history items that are inside the eligible-pool vocab
    (out-of-pool items have no embedding slot and are dropped). Weights are
    uniform 1.0, or exp(-age/`hist_pool_recency_days`) against `boundary_ms`.
    Users with no in-pool positive history keep an all-PAD, all-zero-weight
    row, which the model pools to the exact zero vector."""
    history_df = history_df.copy()
    history_df["user_id"] = history_df["user_id"].astype(str)
    history_df["parent_asin"] = history_df["parent_asin"].astype(str)

    pool_df = item_features_hist[
        item_features_hist["in_eligible_pool"] == 1
    ].copy()
    pool_df["parent_asin"] = pool_df["parent_asin"].astype(str)
    # Future-free deterministic order (stored artifact is already sorted;
    # re-sort defensively so specs never depend on input row order).
    pool_df = pool_df.sort_values("parent_asin", kind="mergesort").reset_index(drop=True)

    user_id_to_idx = _build_vocab(user_features_hist["user_id"].astype(str).tolist())
    item_id_to_idx = _build_vocab(pool_df["parent_asin"].tolist())
    store_vals = pool_df["store"].fillna("Unknown").astype(str)
    main_cat_vals = pool_df["main_category"].fillna("Unknown").astype(str)
    store_to_idx = _build_vocab(store_vals.tolist())
    main_cat_to_idx = _build_vocab(main_cat_vals.tolist())

    if use_deeper_cat and raw_metadata is not None and "categories" in raw_metadata.columns:
        rm = raw_metadata[["parent_asin", "categories"]].copy()
        rm["parent_asin"] = rm["parent_asin"].astype(str)
        rm["_n"] = rm["categories"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        rm = rm.sort_values("_n", ascending=False).drop_duplicates(
            subset=["parent_asin"], keep="first"
        )
        rm["deeper_category"] = rm["categories"].apply(_categories_path)
        deeper_map = dict(zip(rm["parent_asin"], rm["deeper_category"]))
        deeper_values = pool_df["parent_asin"].map(deeper_map).fillna("Unknown").astype(str)
        has_deeper = (deeper_values != "Unknown").any()
        deeper_cat_to_idx = (
            _build_vocab(deeper_values.tolist()) if has_deeper else {"<PAD>": 0}
        )
    else:
        deeper_values = pd.Series(["Unknown"] * len(pool_df))
        has_deeper = False
        deeper_cat_to_idx = {"<PAD>": 0}

    n_users = len(user_id_to_idx)
    ordered_users = [
        u for u, _ in sorted(user_id_to_idx.items(), key=lambda kv: kv[1])
        if u != "<PAD>"
    ]
    user_dense, user_stats = _coerce_dense(
        user_features_hist.set_index("user_id").reindex(ordered_users).reset_index(drop=True),
        USER_DENSE_COLS_HIST, LOG1P_USER_COLS_HIST,
    )
    user_dense_table = np.zeros((n_users, len(USER_DENSE_COLS_HIST)), dtype=np.float32)
    user_dense_table[1:] = user_dense

    n_items = len(item_id_to_idx)
    item_dense, item_stats = _coerce_dense(pool_df, ITEM_DENSE_COLS_HIST, LOG1P_ITEM_COLS_HIST)
    item_dense_table = np.zeros((n_items, len(ITEM_DENSE_COLS_HIST)), dtype=np.float32)
    item_dense_table[1:] = item_dense

    item_store_idx = np.zeros(n_items, dtype=np.int64)
    item_main_cat_idx = np.zeros(n_items, dtype=np.int64)
    item_deeper_cat_idx = np.zeros(n_items, dtype=np.int64)
    pool_stores = store_vals.to_numpy()
    pool_main = main_cat_vals.to_numpy()
    pool_deeper = deeper_values.astype(str).to_numpy()
    pool_pa = pool_df["parent_asin"].to_numpy()
    for k, pa in enumerate(pool_pa):
        ii = item_id_to_idx[pa]
        item_store_idx[ii] = store_to_idx.get(pool_stores[k], 0)
        item_main_cat_idx[ii] = main_cat_to_idx.get(pool_main[k], 0)
        item_deeper_cat_idx[ii] = deeper_cat_to_idx.get(pool_deeper[k], 0)

    candidate_item_idx = np.array([item_id_to_idx[pa] for pa in pool_pa], dtype=np.int64)

    user_hist_idx = None
    user_hist_w = None
    if build_hist_pool:
        assert hist_pool_len > 0, "hist_pool_len must be positive"
        assert "label" in history_df.columns, (
            "build_hist_pool needs the history 'label' column (positives only "
            "enter the taste pool)"
        )
        assert "timestamp" in history_df.columns, (
            "build_hist_pool needs the history 'timestamp' column"
        )
        if hist_pool_recency_days is not None:
            assert hist_pool_recency_days > 0, "hist_pool_recency_days must be positive"
            assert boundary_ms is not None, (
                "hist_pool_recency_days is set but no boundary_ms was passed"
            )
        user_hist_idx = np.zeros((n_users, hist_pool_len), dtype=np.int64)
        user_hist_w = np.zeros((n_users, hist_pool_len), dtype=np.float32)

    user_seen_per_user_idx: Dict[int, set] = {}
    for u, g in history_df.groupby("user_id"):
        u_idx = user_id_to_idx.get(u, 0)
        if u_idx == 0:
            continue
        user_seen_per_user_idx[u_idx] = {
            item_id_to_idx[p] for p in g["parent_asin"].to_numpy()
            if p in item_id_to_idx
        }
        if build_hist_pool:
            pos_g = g[(g["label"] == 1) & g["parent_asin"].isin(item_id_to_idx)]
            if len(pos_g) == 0:
                continue   # zero in-pool positive history -> all-PAD, zero-weight
            # Most recent first; parent_asin ascending breaks timestamp ties
            # deterministically (mergesort = stable).
            pos_g = pos_g.sort_values(
                ["timestamp", "parent_asin"], ascending=[False, True],
                kind="mergesort",
            ).head(hist_pool_len)
            idxs = pos_g["parent_asin"].map(item_id_to_idx).to_numpy(dtype=np.int64)
            if hist_pool_recency_days is not None:
                age_ms = int(boundary_ms) - pos_g["timestamp"].to_numpy(dtype=np.int64)
                assert (age_ms > 0).all(), (
                    "history event at/after the snapshot boundary"
                )
                w = np.exp(
                    -age_ms / (hist_pool_recency_days * MS_PER_DAY)
                ).astype(np.float32)
            else:
                w = np.ones(len(idxs), dtype=np.float32)
            user_hist_idx[u_idx, :len(idxs)] = idxs
            user_hist_w[u_idx, :len(idxs)] = w

    return FeatureSpec(
        n_users=n_users,
        n_items=n_items,
        n_stores=len(store_to_idx),
        n_main_cats=len(main_cat_to_idx),
        n_deeper_cats=len(deeper_cat_to_idx),
        has_deeper_cat=bool(has_deeper),
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        store_to_idx=store_to_idx,
        main_cat_to_idx=main_cat_to_idx,
        deeper_cat_to_idx=deeper_cat_to_idx,
        user_dense=user_dense_table,
        item_dense=item_dense_table,
        item_store_idx=item_store_idx,
        item_main_cat_idx=item_main_cat_idx,
        item_deeper_cat_idx=item_deeper_cat_idx,
        candidate_item_idx=candidate_item_idx,
        user_seen_per_user_idx=user_seen_per_user_idx,
        user_dense_cols=USER_DENSE_COLS_HIST,
        item_dense_cols=ITEM_DENSE_COLS_HIST,
        user_dense_norm_stats=user_stats,
        item_dense_norm_stats=item_stats,
        user_hist_idx=user_hist_idx,
        user_hist_w=user_hist_w,
    )


def _config_hash(cfg: TwoTowerConfig, pair_cfg: PairSamplingConfig,
                 train_cfg: TrainConfig, frozen_epochs: Optional[int],
                 seed: int) -> str:
    """Hash of everything that defines the model + training procedure,
    EXCLUDING the training boundary. Must be identical across A/B/C."""
    payload = {
        "two_tower": asdict(cfg),
        "pair_sampling": asdict(pair_cfg),
        "train": {
            "batch_size": train_cfg.batch_size,
            "lr": train_cfg.lr,
            "weight_decay": train_cfg.weight_decay,
            "grad_clip": train_cfg.grad_clip,
            "optimizer": train_cfg.optimizer,
        },
        "frozen_epochs": frozen_epochs,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def validate_frozen_config(frozen: Dict) -> None:
    """Refuse to load a pre-v2 frozen config. v1-era files lack
    `pair_sampling.loss_mode` (and `train.optimizer`); silently defaulting
    them would retrain the B/C snapshots under a different objective than the
    one selected on A. v1 CHECKPOINTS remain loadable -- via
    `infer_snapshot_checkpoint`, which reads the per-snapshot meta, not this
    file -- only frozen-schedule RETRAINING is refused."""
    cfg = frozen.get("config", {})
    missing = []
    if "loss_mode" not in cfg.get("pair_sampling", {}):
        missing.append("pair_sampling.loss_mode")
    if "optimizer" not in cfg.get("train", {}):
        missing.append("train.optimizer")
    if missing:
        raise ValueError(
            f"two_tower_frozen_config.json predates the v2 fields {missing}; "
            "refusing to load. Re-run the ranker_train development snapshot "
            "to regenerate the frozen config under the v2 code."
        )


def _cohort(n_hist: int) -> str:
    return {0: "0", 1: "1", 2: "2"}.get(int(n_hist), "3+")


def train_snapshot_checkpoint(
    category: str,
    snapshot: str,
    cfg: TwoTowerConfig,
    train_cfg: TrainConfig,
    pair_cfg: PairSamplingConfig,
    seed: int = 42,
    frozen_epochs: Optional[int] = None,
    smoke: bool = False,
    processed_dir: Path = PROCESSED_DIR,
    raw_dir: Path = RAW_DIR,
) -> Dict:
    """Train ONE checkpoint from scratch on its snapshot history and emit
    predictions for that snapshot's label users (>=1 history event).

    frozen_epochs=None -> development mode (snapshot must be 'ranker_train'):
    early-stop on this snapshot's own labels and RETURN the chosen epoch.
    frozen_epochs=int -> train exactly that many epochs, no selection.
    """
    if frozen_epochs is None and snapshot != "ranker_train":
        raise ValueError(
            f"epoch selection is only allowed on the ranker_train snapshot; "
            f"snapshot={snapshot!r} must receive frozen_epochs"
        )
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    tdir = Path(processed_dir) / category / "temporal_ranker"
    history = pd.read_parquet(tdir / f"history_{snapshot}.parquet")
    # v2: selection and reporting run on the all-positive label set. Fall back
    # (loudly) to the single-positive file where groundtruth_all is not on
    # disk -- e.g. synthetic test universes.
    gt_all_path = tdir / f"groundtruth_all_{snapshot}.parquet"
    if gt_all_path.exists():
        labels = pd.read_parquet(gt_all_path)
        label_mode = "all_positive"
    else:
        print(f"[tt-{snapshot}] HARD WARNING: {gt_all_path.name} not found -- "
              f"falling back to single-positive groundtruth_{snapshot}.parquet. "
              f"v2 selection is specified on the all-positive label set.",
              flush=True)
        labels = pd.read_parquet(tdir / f"groundtruth_{snapshot}.parquet")
        label_mode = "single_positive"
    user_features = pd.read_parquet(tdir / f"user_features_{snapshot}.parquet")
    item_features = pd.read_parquet(tdir / f"item_features_{snapshot}.parquet")
    manifest = json.loads((tdir / "snapshot_manifest.json").read_text())
    boundary_ms = int(manifest["snapshots"][snapshot]["history_end_ms"])
    assert int(history["timestamp"].max()) < boundary_ms, (
        "history file violates its own snapshot boundary"
    )

    # v2: the deeper_category raw-metadata read is GATED, not deleted -- v1
    # checkpoints (use_deeper_cat_emb=True in their meta) still re-export
    # through infer_snapshot_checkpoint unchanged.
    raw_meta = None
    if cfg.use_deeper_cat_emb:
        raw_meta_path = Path(raw_dir) / category / "metadata.parquet"
        if raw_meta_path.exists():
            try:
                raw_meta = pd.read_parquet(raw_meta_path, columns=["parent_asin", "categories"])
            except Exception:
                raw_meta = None

    if smoke:
        cfg.embedding_dim = min(cfg.embedding_dim, 16)
        cfg.hidden_dim = min(cfg.hidden_dim, 32)
        cfg.id_emb_dim = min(cfg.id_emb_dim, 16)
        cfg.cat_emb_dim = min(cfg.cat_emb_dim, 8)
        train_cfg.epochs = min(train_cfg.epochs, 2)
        train_cfg.batch_size = min(train_cfg.batch_size, 2048)
        train_cfg.max_train_pairs = 60_000
        train_cfg.max_eval_users = 5_000
        pair_cfg.n_soft_neg = min(pair_cfg.n_soft_neg, 2)

    spec = build_temporal_feature_spec(
        history, user_features, item_features, raw_meta, cfg.use_deeper_cat_emb,
        build_hist_pool=cfg.use_hist_pool,
        hist_pool_len=cfg.hist_pool_len,
        hist_pool_recency_days=cfg.hist_pool_recency_days,
        boundary_ms=boundary_ms,
    )
    print(f"[tt-{snapshot}] vocab users={spec.n_users} items={spec.n_items} "
          f"stores={spec.n_stores} cats={spec.n_main_cats} "
          f"deeper={spec.n_deeper_cats} hist_pool={cfg.use_hist_pool}", flush=True)

    dataset = TwoTowerPairDataset(history, spec, pair_cfg, as_of_ms=boundary_ms)
    composition = dataset.composition()
    print(f"[tt-{snapshot}] pairs = {composition} "
          f"(hard pool = {dataset.n_hard_pool})", flush=True)
    if train_cfg.max_train_pairs and len(dataset) > train_cfg.max_train_pairs:
        # Smoke cap: subsample the INTERNAL pos/hard/soft arrays (not just the
        # flat tensors) so per-epoch soft-negative resampling stays capped.
        ratio = train_cfg.max_train_pairs / len(dataset)
        rng_np = np.random.default_rng(seed)
        for kind in ("pos", "hard", "soft"):
            u = getattr(dataset, f"_{kind}_uidx")
            if len(u) == 0:
                continue
            keep = np.sort(rng_np.choice(
                len(u), size=max(1, int(len(u) * ratio)), replace=False,
            ))
            setattr(dataset, f"_{kind}_uidx", u[keep])
            setattr(dataset, f"_{kind}_iidx",
                    getattr(dataset, f"_{kind}_iidx")[keep])
            if kind == "pos":
                dataset._pos_recency_w = dataset._pos_recency_w[keep]
        dataset._rebuild_flat()
        print(f"[tt-{snapshot}] smoke-capped pairs to {len(dataset):,}", flush=True)

    # Item tower first: the v2 taste channel shares its item_id_emb table.
    item_tower = ItemTower(
        n_items=spec.n_items, n_stores=spec.n_stores,
        n_main_cats=spec.n_main_cats, n_deeper_cats=spec.n_deeper_cats,
        n_dense=spec.n_item_dense, cfg=cfg, has_deeper_cat=spec.has_deeper_cat,
    )
    user_tower = UserTower(
        spec.n_users, spec.n_user_dense, cfg,
        shared_item_emb=item_tower.item_id_emb if cfg.use_hist_pool else None,
    )
    model = TwoTower(user_tower, item_tower).to(train_cfg.device)
    optimizer = build_optimizer(model, train_cfg.optimizer, train_cfg.lr,
                                train_cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    train_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(seed)
    spec_t = _move_spec_to_tensors(spec, train_cfg.device)

    # Ground truth + eval users (>=1 history event only; zero-history users'
    # fallback is the popularity source of the candidate union). gt[u] is a
    # SET -- one item in single-positive mode, all window positives in
    # all-positive mode.
    labels["user_id"] = labels["user_id"].astype(str)
    labels["parent_asin"] = labels["parent_asin"].astype(str)
    # Set-valued per user. The old `{u: {pa} for u, pa in zip(...)}` reads like
    # it builds sets but is a per-ROW comprehension: with several groundtruth
    # rows per user, later keys overwrite earlier ones and each user silently
    # keeps one item.
    gt = {u: set(g) for u, g in
          labels.groupby("user_id")["parent_asin"].agg(set).items()}
    assert sum(len(v) for v in gt.values()) == len(labels), "groundtruth rows dropped"
    # n_history_events must be a per-user constant; a zip would silently keep
    # the last row's value if an artifact ever violated that, so assert it.
    he = labels.groupby("user_id")["n_history_events"]
    he_min, he_max = he.min(), he.max()
    assert (he_min == he_max).all(), (
        "n_history_events differs across a user's label rows -- corrupt "
        "groundtruth artifact"
    )
    hist_events = he_max.astype(int).to_dict()
    eval_users = sorted(u for u in gt if hist_events.get(u, 0) >= 1)
    n_zero_history = len(gt) - len(eval_users)
    eval_users_capped = bool(
        train_cfg.max_eval_users and len(eval_users) > train_cfg.max_eval_users
    )
    if eval_users_capped:
        rng_eval = np.random.default_rng(seed + 1)
        eval_users = sorted(rng_eval.choice(
            eval_users, size=train_cfg.max_eval_users, replace=False,
        ))
    # Reporting scope: with the smoke cap active, users outside the sample
    # would register as guaranteed misses and poison the diagnostics -- score
    # metrics over the sampled users only and flag it. Without the cap, the
    # FULL gt (including zero-history users, counted as misses) is the
    # denominator, per the no-silent-drop contract.
    gt_for_metrics = {u: gt[u] for u in eval_users} if eval_users_capped else gt
    eval_user_idx_np = np.array(
        [spec.user_id_to_idx.get(u, 0) for u in eval_users], dtype=np.int64,
    )
    eval_user_idx = torch.from_numpy(eval_user_idx_np)
    candidate_pa = np.array(
        [pa for pa, i in sorted(spec.item_id_to_idx.items(), key=lambda kv: kv[1])
         if pa != "<PAD>"], dtype=object,
    )
    pool_set = set(candidate_pa.tolist())

    def _current_topk(k=DEFAULT_K_RETRIEVE, return_scores=False):
        if not eval_users:
            return ({}, {}) if return_scores else {}
        item_vecs = encode_all_items(model, spec, spec_t, device=train_cfg.device)
        user_vecs = encode_users_subset(model, eval_user_idx, spec_t,
                                        device=train_cfg.device)
        return topk_per_user_chunked(
            user_vecs, item_vecs, candidate_pa, eval_users, eval_user_idx_np,
            spec.user_seen_per_user_idx, k=k, return_scores=return_scores,
        )

    # ---- training -------------------------------------------------------------
    history_log: List[Dict] = []
    development_mode = frozen_epochs is None
    max_epochs = train_cfg.epochs if development_mode else frozen_epochs
    best_recall, best_state, best_epoch = -1.0, None, 0
    patience_left = train_cfg.early_stopping_patience

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        if pair_cfg.loss_mode == "softmax":
            train_metrics = train_one_epoch_softmax(
                model, dataset, spec_t, optimizer, train_cfg.device,
                train_cfg.grad_clip, train_cfg.batch_size, train_rng,
                tail_size=pair_cfg.softmax_tail_size,
                hard_negative_lambda=pair_cfg.hard_negative_lambda,
            )
        else:
            train_metrics = train_one_epoch(
                model, dataset, spec_t, optimizer, bce, train_cfg.device,
                train_cfg.grad_clip, train_cfg.batch_size, train_rng,
            )
        entry = {"epoch": epoch, "epoch_seconds": round(time.time() - ep_t0, 2),
                 "train": train_metrics}
        if development_mode:
            # Selection against THIS snapshot's own (development) labels.
            topk = _current_topk()
            rep = build_split_report(snapshot, topk, gt_for_metrics, pool_set,
                                     "eligible_pool", DEFAULT_KS)
            entry["dev_metrics"] = dict(rep.metrics)
            cur = rep.metrics.get("Recall@100", -1.0)
            print(f"[tt-{snapshot}] ep{epoch:02d} loss={train_metrics['loss']:.4f} "
                  f"dev_R@100={cur:.4f}", flush=True)
            if cur > best_recall:
                best_recall, best_epoch = cur, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_left = train_cfg.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    history_log.append(entry)
                    print(f"[tt-{snapshot}] early stop at ep{epoch}", flush=True)
                    break
        else:
            print(f"[tt-{snapshot}] ep{epoch:02d}/{max_epochs} "
                  f"loss={train_metrics['loss']:.4f} (frozen schedule)", flush=True)
        history_log.append(entry)
        dataset.resample_soft_negatives(epoch=epoch)

    if development_mode and best_state is not None:
        model.load_state_dict(best_state)
        chosen_epochs = best_epoch
    else:
        chosen_epochs = max_epochs

    # ---- final inference + report ---------------------------------------------
    topk_items, topk_scores = _current_topk(return_scores=True)
    rep = build_split_report(snapshot, topk_items, gt_for_metrics, pool_set,
                             "eligible_pool", DEFAULT_KS)

    # Dual-gt reporting: alongside the
    # all-positive metrics, score the SAME top-K against the single-positive
    # groundtruth so numbers stay comparable across the migration. Selection
    # above used the all-positive set only.
    metrics_single_gt = None
    if label_mode == "all_positive":
        single_path = tdir / f"groundtruth_{snapshot}.parquet"
        if single_path.exists():
            sl = pd.read_parquet(single_path)
            sl["user_id"] = sl["user_id"].astype(str)
            sl["parent_asin"] = sl["parent_asin"].astype(str)
            gt_single = {u: set(g) for u, g in sl.groupby("user_id")["parent_asin"]}
            if eval_users_capped:
                gt_single = {u: gt_single[u] for u in eval_users if u in gt_single}
            rep_single = build_split_report(
                f"{snapshot}_single_gt", topk_items, gt_single, pool_set,
                "eligible_pool", DEFAULT_KS,
            )
            metrics_single_gt = dict(rep_single.metrics)

    # Cohort breakdown (0 cohort has no predictions by design -> recall 0).
    cohort_metrics: Dict[str, Dict] = {}
    for cohort_name in ("0", "1", "2", "3+"):
        users_c = {u for u in gt_for_metrics
                   if _cohort(hist_events.get(u, 0)) == cohort_name}
        sub_gt = {u: g for u, g in gt_for_metrics.items() if u in users_c}
        sub_topk = {u: topk_items.get(u, []) for u in users_c}
        crep = build_split_report(f"{snapshot}_h{cohort_name}", sub_topk, sub_gt,
                                  pool_set, "eligible_pool", DEFAULT_KS)
        cohort_metrics[cohort_name] = {
            "n_users": len(users_c), **crep.metrics,
        }

    pred_rows = []
    for uid in eval_users:
        rank = 0
        for pa, sc in zip(topk_items[uid], topk_scores[uid]):
            # When the pool is smaller than K, torch.topk keeps the -inf
            # seen-masked slots; they are not real recommendations.
            if not np.isfinite(sc):
                continue
            rank += 1
            pred_rows.append((uid, pa, float(sc), rank))
    pred_df = pd.DataFrame(
        pred_rows, columns=["user_id", "parent_asin", "score", "rank"],
    )
    pred_path = tdir / f"two_tower_predictions_{snapshot}.parquet"
    pred_df.to_parquet(pred_path, index=False)

    ckpt_path = tdir / f"two_tower_{snapshot}.pt"
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "category": category,
        "snapshot": snapshot,
        "training_boundary_ms": boundary_ms,
        "training_boundary_iso": manifest["snapshots"][snapshot]["history_end_iso"],
        "n_training_events": int(len(history)),
        "dataset_composition": dataset.composition(),
        "seed": seed,
        "mode": "development_early_stop" if development_mode else "frozen_schedule",
        "epochs_trained": chosen_epochs,
        "frozen_epochs_input": frozen_epochs,
        "config": {
            "two_tower": asdict(cfg),
            "pair_sampling": asdict(pair_cfg),
            "train": {"batch_size": train_cfg.batch_size, "lr": train_cfg.lr,
                      "weight_decay": train_cfg.weight_decay,
                      "grad_clip": train_cfg.grad_clip,
                      "optimizer": train_cfg.optimizer},
        },
        "config_hash": _config_hash(cfg, pair_cfg, train_cfg,
                                    frozen_epochs=chosen_epochs, seed=seed),
        "vocab_sizes": {"users": spec.n_users, "items": spec.n_items},
        "label_mode": label_mode,
        "n_label_users": int(len(gt)),
        "n_label_rows": int(len(labels)),
        "n_zero_history_label_users": int(n_zero_history),
        "empty_pool_user_share": round(n_zero_history / max(1, len(gt)), 6),
        "n_hard_neg_pool": dataset.n_hard_pool,
        "metrics_over_sampled_users_only": eval_users_capped,
        "n_users_in_metrics": int(len(gt_for_metrics)),
        "metrics": dict(rep.metrics),
        "metrics_single_gt": metrics_single_gt,
        "coverage_eligible_pool": rep.heldout_positive_coverage_by_candidate_pool,
        "cohorts": cohort_metrics,
        "history_log": history_log,
        "elapsed_seconds": round(time.time() - t0, 2),
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(tdir / f"two_tower_{snapshot}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[tt-{snapshot}] done: epochs={chosen_epochs} "
          f"R@100={rep.metrics['Recall@100']:.4f} -> {ckpt_path.name}", flush=True)
    return meta


@torch.no_grad()
def infer_snapshot_checkpoint(
    category: str,
    snapshot: str,
    k: int = DEFAULT_K_RETRIEVE,
    processed_dir: Path = PROCESSED_DIR,
    raw_dir: Path = RAW_DIR,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Inference-only: load the frozen checkpoint for `snapshot` and re-emit
    per-user top-K predictions at an arbitrary K. NO training, NO selection --
    safe for the Phase 3A candidate-K sweep (larger K from the same frozen
    model is a pure widening of the returned list).

    Rebuilds the FeatureSpec deterministically from the snapshot artifacts
    (same inputs as training), restores weights, and scores label users with
    >= 1 history event exactly like train_snapshot_checkpoint's final pass.
    """
    tdir = Path(processed_dir) / category / "temporal_ranker"
    meta = json.loads((tdir / f"two_tower_{snapshot}_meta.json").read_text())
    cfg = TwoTowerConfig(**meta["config"]["two_tower"])

    history = pd.read_parquet(tdir / f"history_{snapshot}.parquet")
    labels = pd.read_parquet(tdir / f"groundtruth_{snapshot}.parquet")
    user_features = pd.read_parquet(tdir / f"user_features_{snapshot}.parquet")
    item_features = pd.read_parquet(tdir / f"item_features_{snapshot}.parquet")
    # v2: gated raw-metadata read (kept, not deleted) -- a v1 checkpoint's meta
    # has use_deeper_cat_emb=True and takes this path exactly as before.
    raw_meta = None
    if cfg.use_deeper_cat_emb:
        raw_meta_path = Path(raw_dir) / category / "metadata.parquet"
        if raw_meta_path.exists():
            try:
                raw_meta = pd.read_parquet(raw_meta_path,
                                           columns=["parent_asin", "categories"])
            except Exception:
                raw_meta = None

    spec = build_temporal_feature_spec(
        history, user_features, item_features, raw_meta, cfg.use_deeper_cat_emb,
        build_hist_pool=cfg.use_hist_pool,
        hist_pool_len=cfg.hist_pool_len,
        hist_pool_recency_days=cfg.hist_pool_recency_days,
        boundary_ms=int(meta["training_boundary_ms"]),
    )
    item_tower = ItemTower(
        n_items=spec.n_items, n_stores=spec.n_stores,
        n_main_cats=spec.n_main_cats, n_deeper_cats=spec.n_deeper_cats,
        n_dense=spec.n_item_dense, cfg=cfg, has_deeper_cat=spec.has_deeper_cat,
    )
    user_tower = UserTower(
        spec.n_users, spec.n_user_dense, cfg,
        shared_item_emb=item_tower.item_id_emb if cfg.use_hist_pool else None,
    )
    model = TwoTower(user_tower, item_tower)
    state = torch.load(tdir / f"two_tower_{snapshot}.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    spec_t = _move_spec_to_tensors(spec, device)

    labels["user_id"] = labels["user_id"].astype(str)
    hist_events = dict(zip(labels["user_id"],
                           labels["n_history_events"].astype(int)))
    # unique(): several groundtruth rows per user must not yield duplicate
    # eval users, which would double-count them in the emitted predictions.
    eval_users = sorted({u for u in labels["user_id"]
                         if hist_events.get(u, 0) >= 1})
    if not eval_users:
        pred_df = pd.DataFrame(columns=["user_id", "parent_asin", "score", "rank"])
    else:
        eval_user_idx_np = np.array(
            [spec.user_id_to_idx.get(u, 0) for u in eval_users], dtype=np.int64,
        )
        candidate_pa = np.array(
            [pa for pa, i in sorted(spec.item_id_to_idx.items(),
                                    key=lambda kv: kv[1]) if pa != "<PAD>"],
            dtype=object,
        )
        item_vecs = encode_all_items(model, spec, spec_t, device=device)
        user_vecs = encode_users_subset(
            model, torch.from_numpy(eval_user_idx_np), spec_t, device=device,
        )
        topk_items, topk_scores = topk_per_user_chunked(
            user_vecs, item_vecs, candidate_pa, eval_users, eval_user_idx_np,
            spec.user_seen_per_user_idx, k=k, return_scores=True,
        )
        rows = []
        for uid in eval_users:
            rank = 0
            for pa, sc in zip(topk_items[uid], topk_scores[uid]):
                if not np.isfinite(sc):
                    continue
                rank += 1
                rows.append((uid, pa, float(sc), rank))
        pred_df = pd.DataFrame(rows, columns=["user_id", "parent_asin",
                                              "score", "rank"])
    if out_path is not None:
        # Atomic write: concurrent variant jobs share this cache file, and a
        # reader must never see a half-written parquet (observed as a 4-byte
        # file in parallel Stage B runs).
        import os
        tmp = Path(str(out_path) + f".tmp.{os.getpid()}")
        pred_df.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
    return pred_df


def run_all_snapshots(
    category: str,
    cfg: TwoTowerConfig,
    train_cfg: TrainConfig,
    pair_cfg: PairSamplingConfig,
    seed: int = 42,
    smoke: bool = False,
    processed_dir: Path = PROCESSED_DIR,
    raw_dir: Path = RAW_DIR,
) -> Dict:
    """A (develop + freeze) -> B, C (frozen schedule). Returns per-snapshot meta."""
    tdir = Path(processed_dir) / category / "temporal_ranker"
    frozen_path = tdir / "two_tower_frozen_config.json"

    import copy
    meta_a = train_snapshot_checkpoint(
        category, "ranker_train", copy.deepcopy(cfg), copy.deepcopy(train_cfg),
        copy.deepcopy(pair_cfg), seed=seed, frozen_epochs=None, smoke=smoke,
        processed_dir=processed_dir, raw_dir=raw_dir,
    )
    frozen = {
        "frozen_epochs": meta_a["epochs_trained"],
        "config_hash": meta_a["config_hash"],
        "selected_on": "ranker_train (T0->T1 development labels)",
        "config": meta_a["config"],
        "seed": seed,
    }
    with open(frozen_path, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"[tt] frozen config saved: epochs={frozen['frozen_epochs']} "
          f"hash={frozen['config_hash']}", flush=True)

    metas = {"ranker_train": meta_a}
    for snap in ("model_selection", "test"):
        metas[snap] = train_snapshot_checkpoint(
            category, snap, copy.deepcopy(cfg), copy.deepcopy(train_cfg),
            copy.deepcopy(pair_cfg), seed=seed,
            frozen_epochs=frozen["frozen_epochs"], smoke=smoke,
            processed_dir=processed_dir, raw_dir=raw_dir,
        )
        assert metas[snap]["config_hash"] == frozen["config_hash"], (
            "config hash drifted between checkpoints -- hyperparameters must "
            "be frozen after the ranker_train development snapshot"
        )
    return metas


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--snapshot", default="all",
                   choices=["all", *SNAPSHOT_NAMES])
    p.add_argument("--variant", default="metadata_and_ids",
                   choices=["metadata_only", "metadata_and_ids"])
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--id-emb-dim", type=int, default=32)
    p.add_argument("--cat-emb-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=8,
                   help="Max epochs for the development checkpoint (A).")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-soft-negatives", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    # ---- v2 knobs (defaults reproduce v1 exactly) ----------------------------
    p.add_argument("--loss-mode", default="bce", choices=["bce", "softmax"],
                   help="bce = v1 explicit-pair objective; softmax = in-batch "
                        "sampled softmax with exact-unigram logQ + MNS tail.")
    p.add_argument("--hard-negative-lambda", type=float, default=0.0,
                   help="softmax mode: weight of the auxiliary BCE term over "
                        "explicit label==0 pairs.")
    p.add_argument("--softmax-tail-size", type=int, default=4096,
                   help="B' uniform MNS tail items per batch (0 disables).")
    p.add_argument("--pair-recency-decay-days", type=float, default=None,
                   help="exp(-age/days) weight on positive pairs; None = off.")
    p.add_argument("--use-hist-pool", action="store_true", default=False,
                   help="v2 taste channel: pooled positive-history item "
                        "embeddings (shares the item tower's id table).")
    p.add_argument("--hist-pool-len", type=int, default=20)
    p.add_argument("--hist-pool-recency-days", type=float, default=None)
    p.add_argument("--no-deeper-cat-emb", action="store_true", default=False,
                   help="Disable the deeper_category embedding AND skip the "
                        "raw-metadata read that feeds it.")
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                   help="adam = v1; adamw = decoupled decay, wd=0 on all "
                        "embedding groups + logit_scale.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    use_ids = args.variant == "metadata_and_ids"
    cfg = TwoTowerConfig(
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        id_emb_dim=args.id_emb_dim, cat_emb_dim=args.cat_emb_dim,
        dropout=args.dropout, use_user_id_emb=use_ids, use_item_id_emb=use_ids,
        use_deeper_cat_emb=not args.no_deeper_cat_emb,
        use_hist_pool=args.use_hist_pool,
        hist_pool_len=args.hist_pool_len,
        hist_pool_recency_days=args.hist_pool_recency_days,
    )
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, weight_decay=args.weight_decay,
                            early_stopping_patience=2,
                            optimizer=args.optimizer)
    pair_cfg = PairSamplingConfig(n_soft_neg=args.num_soft_negatives,
                                  seed=args.seed,
                                  loss_mode=args.loss_mode,
                                  hard_negative_lambda=args.hard_negative_lambda,
                                  softmax_tail_size=args.softmax_tail_size,
                                  pair_recency_decay_days=args.pair_recency_decay_days)
    if args.snapshot == "all":
        run_all_snapshots(args.category, cfg, train_cfg, pair_cfg,
                          seed=args.seed, smoke=args.smoke)
        return

    frozen_path = (PROCESSED_DIR / args.category / "temporal_ranker"
                   / "two_tower_frozen_config.json")
    if args.snapshot == "ranker_train":
        meta = train_snapshot_checkpoint(
            args.category, args.snapshot, cfg, train_cfg, pair_cfg,
            seed=args.seed, frozen_epochs=None, smoke=args.smoke,
        )
        with open(frozen_path, "w") as f:
            json.dump({
                "frozen_epochs": meta["epochs_trained"],
                "config_hash": meta["config_hash"],
                "selected_on": "ranker_train (T0->T1 development labels)",
                "config": meta["config"],
                "seed": args.seed,
                "smoke": args.smoke,
            }, f, indent=2)
        print(f"[tt] frozen config saved: epochs={meta['epochs_trained']} "
              f"hash={meta['config_hash']}", flush=True)
        return

    # model_selection / test: the ENTIRE configuration comes from the frozen
    # file -- CLI model/training flags are ignored so hyperparameters cannot
    # drift from checkpoint A.
    frozen = json.loads(frozen_path.read_text())
    validate_frozen_config(frozen)   # v1-era files (no loss_mode): refuse.
    print("[tt] single-snapshot mode: rebuilding config from "
          "two_tower_frozen_config.json (CLI model flags ignored)", flush=True)
    cfg = TwoTowerConfig(**frozen["config"]["two_tower"])
    pair_cfg = PairSamplingConfig(**frozen["config"]["pair_sampling"])
    tr = frozen["config"]["train"]
    train_cfg = TrainConfig(batch_size=tr["batch_size"], lr=tr["lr"],
                            weight_decay=tr["weight_decay"],
                            grad_clip=tr["grad_clip"],
                            optimizer=tr["optimizer"])
    meta = train_snapshot_checkpoint(
        args.category, args.snapshot, cfg, train_cfg, pair_cfg,
        seed=frozen["seed"], frozen_epochs=frozen["frozen_epochs"],
        smoke=frozen.get("smoke", False),
    )
    assert meta["config_hash"] == frozen["config_hash"], (
        "config hash drifted from the frozen configuration -- refusing to "
        "accept this checkpoint"
    )


if __name__ == "__main__":
    main()
