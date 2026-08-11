"""Two-tower retrieval — feature preparation + explicit pair sampler.

Spec-locked design choices, in service of a clean v1 (see the Phase 1 plan in
project memory and the user's two-tower spec):

* v1 default (loss_mode="bce"): no in-batch sampled softmax. Negatives are
  explicit:
    - **Positives**:        train rows with label==1.
    - **Hard negatives**    train rows with label==0 (rating<3 interactions).
                            The pool is ALWAYS built (v2); use_hard_negatives
                            only gates their inclusion in the BCE flat tensors.
    - **Soft negatives**    items sampled from the in_train_catalog pool MINUS
                            anything the user has seen in train (positives or
                            hard negs). num_soft_negatives per positive.
* v2 (loss_mode="softmax"): the trainer draws negatives in-batch + uniform
  MNS tail (see train_two_tower.train_one_epoch_softmax); the soft-negative
  sampler is skipped and only the positive/hard arrays are consumed.
* No shared id_emb table between user-history encoder and item tower
  (we don't even build a history encoder in v1; user-tower input is dense + an
  optional standalone user_id embedding).
* Train-only feature substrate: user_features.parquet + item_features.parquet
  carry only train-derived aggregates. The dataset never reads val/test rows.
* Candidate universe = item_features[in_train_catalog == 1]. We never score
  cold items in v1.

The dataset returns flat tensors of (user_idx, item_idx, label, weight). The
model indexes into precomputed feature tables (FeatureSpec) using these idxs;
this avoids per-row numpy work in the DataLoader hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Indexing convention: 0 is reserved as the "unknown / padding" slot for every
# categorical embedding, so out-of-vocabulary keys at inference time map there.
PAD_IDX = 0

# Keep in sync with scripts.data.temporal_split.MS_PER_DAY (this module stays
# stdlib+np+pd+torch-only on purpose; the anchor scripts monkeypatch around it).
MS_PER_DAY = 86_400_000


# ---- Dense feature normalization helpers -------------------------------------

def _log1p_clip(x: np.ndarray, clip_max: Optional[float] = None) -> np.ndarray:
    """log1p with optional pre-clip — guards against the long tails Amazon's
    rating_number / n_reviews_train / etc. carry."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0)
    if clip_max is not None:
        x = np.minimum(x, clip_max)
    return np.log1p(x)


def _zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Z-score; returns (normalized, mean, std). Falls back to no-op when std==0
    so we don't divide by zero on degenerate columns (e.g., a category with
    only one value in user_features.std_rating_train)."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
    std = float(np.nanstd(x)) if np.isfinite(x).any() else 0.0
    out = x - mean
    if std > 1e-9:
        out = out / std
    return out.astype(np.float32), mean, std


def _categories_path(value) -> str:
    """Join a `categories` list to a single deeper-category string. Empty/missing
    lists return 'Unknown' so they collapse to the OOV slot."""
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        return " | ".join(str(x) for x in value)
    return "Unknown"


# ---- Vocabularies + feature tables -------------------------------------------

USER_DENSE_COLS = (
    "n_reviews_train",
    "avg_rating_train",
    "std_rating_train",
    "n_unique_items_train",
    "active_days_train",
    "verified_rate_train",
)

ITEM_DENSE_COLS = (
    "price",
    "average_rating",
    "rating_number",
    "n_reviews_train",
    "avg_rating_train",
    "n_features",
    "n_description",
    "n_categories",
)

# Columns we apply log1p to (long-tailed counts/ratings_number).
LOG1P_USER_COLS = ("n_reviews_train", "n_unique_items_train", "active_days_train")
LOG1P_ITEM_COLS = ("price", "rating_number", "n_reviews_train",
                   "n_features", "n_description", "n_categories")


@dataclass
class FeatureSpec:
    """Frozen feature tables + vocabularies for one category. Built once,
    indexed by user_idx/item_idx during training and inference."""
    # Vocab sizes (effective = max idx + 1, including PAD_IDX at 0).
    n_users: int
    n_items: int
    n_stores: int
    n_main_cats: int
    n_deeper_cats: int
    has_deeper_cat: bool

    # raw_id -> idx (PAD_IDX==0 is reserved for unknowns).
    user_id_to_idx: Dict[str, int]
    item_id_to_idx: Dict[str, int]
    store_to_idx: Dict[str, int]
    main_cat_to_idx: Dict[str, int]
    deeper_cat_to_idx: Dict[str, int]

    # Dense feature tensors; row 0 corresponds to PAD_IDX (zeros).
    user_dense: np.ndarray   # [n_users, n_user_dense] float32
    item_dense: np.ndarray   # [n_items, n_item_dense] float32

    # Per-item categorical idxs for the candidate pool slots in `item_id_to_idx`.
    # Row 0 corresponds to PAD_IDX (idx 0 in each cat vocab).
    item_store_idx: np.ndarray         # [n_items] int64
    item_main_cat_idx: np.ndarray      # [n_items] int64
    item_deeper_cat_idx: np.ndarray    # [n_items] int64

    # Candidate item idxs (every in_train_catalog item). Soft negatives draw from here.
    candidate_item_idx: np.ndarray     # int64

    # Train-seen items per user_idx (soft-neg sampler excludes these).
    user_seen_per_user_idx: Dict[int, Set[int]]

    # Names for downstream reporting.
    user_dense_cols: Tuple[str, ...] = USER_DENSE_COLS
    item_dense_cols: Tuple[str, ...] = ITEM_DENSE_COLS

    # Normalization stats (saved in the JSON report so behavior is auditable).
    user_dense_norm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    item_dense_norm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ---- v2 taste channel (Track A). Tail-appended with None defaults so the
    # phase1 path (`build_feature_spec`), which never builds history arrays,
    # keeps its field order and construction untouched. Populated only by
    # build_temporal_feature_spec(build_hist_pool=True).
    user_hist_idx: Optional[np.ndarray] = None   # [n_users, L] int64, PAD-padded
    user_hist_w: Optional[np.ndarray] = None     # [n_users, L] float32, 0 on pad

    @property
    def n_user_dense(self) -> int:
        return self.user_dense.shape[1]

    @property
    def n_item_dense(self) -> int:
        return self.item_dense.shape[1]


def _build_vocab(values: Sequence[str], pad: str = "<PAD>") -> Dict[str, int]:
    """Sorted-unique vocab; PAD_IDX==0 reserved. Real entries start at 1."""
    seen: Dict[str, int] = {pad: PAD_IDX}
    for v in sorted(set(str(x) for x in values if x is not None)):
        if v == pad:
            continue
        seen[v] = len(seen)
    return seen


def _coerce_dense(df: pd.DataFrame, cols: Sequence[str],
                  log1p_cols: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Apply log1p to long-tailed cols, then z-score every col. Row order is
    preserved; the caller is responsible for prepending the PAD row of zeros."""
    log1p_set = set(log1p_cols)
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    stats: Dict[str, Dict[str, float]] = {}
    for j, c in enumerate(cols):
        col = df[c].to_numpy() if c in df.columns else np.zeros(len(df))
        col = pd.to_numeric(pd.Series(col), errors="coerce").fillna(0.0).to_numpy()
        if c in log1p_set:
            col = _log1p_clip(col)
        norm, mean, std = _zscore(col)
        out[:, j] = norm
        stats[c] = {"mean": float(mean), "std": float(std), "log1p": (c in log1p_set)}
    return out, stats


def build_feature_spec(
    train_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    raw_metadata: Optional[pd.DataFrame] = None,
    use_deeper_cat: bool = True,
) -> FeatureSpec:
    """Build the FeatureSpec that the trainer + model index into.

    `raw_metadata`, when supplied, is data/raw/{cat}/metadata.parquet (post the
    2026-04-26 refresh) restricted to at least [parent_asin, categories]. We use
    it to derive deeper_category for items in the candidate pool. If omitted
    or `use_deeper_cat=False`, deeper_cat is degenerate (single OOV slot).
    """
    train_df = train_df.copy()
    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["parent_asin"] = train_df["parent_asin"].astype(str)

    # Candidate pool = in_train_catalog items.
    pool_df = item_features[item_features["in_train_catalog"] == 1].reset_index(drop=True)
    pool_df["parent_asin"] = pool_df["parent_asin"].astype(str)

    # ---- vocabularies ---------------------------------------------------------
    user_id_to_idx = _build_vocab(user_features["user_id"].astype(str).tolist())
    item_id_to_idx = _build_vocab(pool_df["parent_asin"].tolist())
    store_to_idx = _build_vocab(pool_df["store"].astype(str).fillna("Unknown").tolist())
    main_cat_to_idx = _build_vocab(pool_df["main_category"].astype(str).fillna("Unknown").tolist())

    # deeper category from raw metadata
    if use_deeper_cat and raw_metadata is not None and "categories" in raw_metadata.columns:
        rm = raw_metadata[["parent_asin", "categories"]].copy()
        rm["parent_asin"] = rm["parent_asin"].astype(str)
        # canonicalize.py keeps the highest-rating-number row per parent_asin;
        # for deeper-category we just keep the longest list (most specific path).
        rm["_n"] = rm["categories"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        rm = rm.sort_values("_n", ascending=False).drop_duplicates(
            subset=["parent_asin"], keep="first"
        )
        rm["deeper_category"] = rm["categories"].apply(_categories_path)
        deeper_map = dict(zip(rm["parent_asin"], rm["deeper_category"]))
        deeper_values = pool_df["parent_asin"].map(deeper_map).fillna("Unknown").astype(str)
        # If every row is "Unknown" the feature is dead; flag and skip.
        has_deeper = (deeper_values != "Unknown").any()
        deeper_cat_to_idx = (
            _build_vocab(deeper_values.tolist()) if has_deeper else {"<PAD>": PAD_IDX}
        )
    else:
        deeper_values = pd.Series(["Unknown"] * len(pool_df))
        has_deeper = False
        deeper_cat_to_idx = {"<PAD>": PAD_IDX}

    # ---- user feature table (one row per user_idx, row 0 = PAD) --------------
    n_users = len(user_id_to_idx)
    user_dense, user_stats = _coerce_dense(
        user_features.set_index("user_id").reindex(
            [u for u, _ in sorted(user_id_to_idx.items(), key=lambda kv: kv[1]) if u != "<PAD>"]
        ).reset_index(drop=True),
        USER_DENSE_COLS, LOG1P_USER_COLS,
    )
    user_dense_table = np.zeros((n_users, len(USER_DENSE_COLS)), dtype=np.float32)
    user_dense_table[1:] = user_dense  # row 0 stays zero for PAD_IDX

    # ---- item feature table ---------------------------------------------------
    n_items = len(item_id_to_idx)
    item_dense, item_stats = _coerce_dense(pool_df, ITEM_DENSE_COLS, LOG1P_ITEM_COLS)
    item_dense_table = np.zeros((n_items, len(ITEM_DENSE_COLS)), dtype=np.float32)
    item_dense_table[1:] = item_dense

    # Categorical idxs (0=PAD, real items at 1..n_items-1, aligned to pool_df order).
    item_store_idx = np.zeros(n_items, dtype=np.int64)
    item_main_cat_idx = np.zeros(n_items, dtype=np.int64)
    item_deeper_cat_idx = np.zeros(n_items, dtype=np.int64)

    pool_stores = pool_df["store"].astype(str).fillna("Unknown").to_numpy()
    pool_main = pool_df["main_category"].astype(str).fillna("Unknown").to_numpy()
    pool_deeper = deeper_values.astype(str).to_numpy()
    pool_pa = pool_df["parent_asin"].to_numpy()

    for k, pa in enumerate(pool_pa):
        ii = item_id_to_idx[pa]
        item_store_idx[ii] = store_to_idx.get(pool_stores[k], PAD_IDX)
        item_main_cat_idx[ii] = main_cat_to_idx.get(pool_main[k], PAD_IDX)
        item_deeper_cat_idx[ii] = deeper_cat_to_idx.get(pool_deeper[k], PAD_IDX)

    # ---- candidate idxs + per-user train-seen sets ---------------------------
    candidate_item_idx = np.array(
        [item_id_to_idx[pa] for pa in pool_pa], dtype=np.int64
    )
    user_seen_per_user_idx: Dict[int, Set[int]] = {}
    for u, g in train_df.groupby("user_id"):
        u_idx = user_id_to_idx.get(u, PAD_IDX)
        if u_idx == PAD_IDX:
            continue
        seen_pa = g["parent_asin"].to_numpy()
        seen_iidx = {item_id_to_idx[p] for p in seen_pa if p in item_id_to_idx}
        user_seen_per_user_idx[u_idx] = seen_iidx

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
        user_dense_norm_stats=user_stats,
        item_dense_norm_stats=item_stats,
    )


# ---- Pair sampler dataset ----------------------------------------------------

@dataclass
class PairSamplingConfig:
    n_soft_neg: int = 4
    use_hard_negatives: bool = True
    positive_weight: float = 1.0
    hard_negative_weight: float = 1.0
    soft_negative_weight: float = 1.0
    seed: int = 42
    # ---- v2 (rebuild wave, Track A). Tail-appended; defaults preserve v1.
    # These are hashed via asdict() in _config_hash and land in the frozen
    # config, so they cannot drift between the A/B/C snapshot checkpoints.
    loss_mode: str = "bce"                  # "bce" (v1) | "softmax" (in-batch + logQ)
    hard_negative_lambda: float = 0.0       # softmax mode: weight of the auxiliary
                                            # BCE term over explicit label==0 pairs
    softmax_tail_size: int = 4096           # B' uniform MNS tail per batch (0 = off)
    pair_recency_decay_days: Optional[float] = None  # positive-pair recency weight
                                            # exp(-age/days); None = v1 uniform


class TwoTowerPairDataset(Dataset):
    """Flat (user_idx, item_idx, label, weight) row sequence.

    Built once at __init__ for positives + hard negatives. Soft negatives are
    resampled each call to ``resample_soft_negatives(epoch)`` so the model
    doesn't see a fixed soft-negative set across epochs.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        spec: FeatureSpec,
        config: PairSamplingConfig,
        as_of_ms: Optional[int] = None,
    ):
        assert config.loss_mode in ("bce", "softmax"), (
            f"unknown loss_mode {config.loss_mode!r}; expected 'bce' or 'softmax'"
        )
        self.spec = spec
        self.config = config

        train_df = train_df.copy()
        train_df["user_id"] = train_df["user_id"].astype(str)
        train_df["parent_asin"] = train_df["parent_asin"].astype(str)

        # Drop rows whose item isn't in the candidate vocab (shouldn't happen in
        # well-formed data, but be defensive against orphan rows).
        train_df = train_df[train_df["parent_asin"].isin(spec.item_id_to_idx)]

        train_df["_uidx"] = train_df["user_id"].map(spec.user_id_to_idx).fillna(PAD_IDX).astype(np.int64)
        train_df["_iidx"] = train_df["parent_asin"].map(spec.item_id_to_idx).fillna(PAD_IDX).astype(np.int64)
        train_df = train_df[(train_df["_uidx"] != PAD_IDX) & (train_df["_iidx"] != PAD_IDX)]

        pos = train_df[train_df["label"] == 1]
        # v2: the hard-negative pool is ALWAYS built from label==0 rows,
        # decoupled from use_hard_negatives (which now only gates their
        # inclusion in the BCE flat tensors). The softmax auxiliary term
        # (hard_negative_lambda) draws from this pool regardless of the flag.
        hardneg = train_df[train_df["label"] == 0]

        self._pos_uidx = pos["_uidx"].to_numpy()
        self._pos_iidx = pos["_iidx"].to_numpy()
        self._hard_uidx = hardneg["_uidx"].to_numpy()
        self._hard_iidx = hardneg["_iidx"].to_numpy()

        # v2 recency weight on positive pairs: exp(-age / decay_days). All-ones
        # when disabled, so the flat weight tensor is byte-identical to v1.
        if config.pair_recency_decay_days is not None:
            assert config.pair_recency_decay_days > 0, (
                "pair_recency_decay_days must be positive"
            )
            assert as_of_ms is not None, (
                "pair_recency_decay_days is set but no as_of_ms was passed -- "
                "the temporal driver must supply the snapshot boundary"
            )
            assert "timestamp" in train_df.columns, (
                "pair_recency_decay_days needs a 'timestamp' column on the "
                "training frame"
            )
            age_ms = int(as_of_ms) - pos["timestamp"].to_numpy(dtype=np.int64)
            assert (age_ms > 0).all(), (
                "positive pair at/after as_of_ms -- history violates its boundary"
            )
            self._pos_recency_w = np.exp(
                -age_ms / (config.pair_recency_decay_days * MS_PER_DAY)
            ).astype(np.float32)
        else:
            self._pos_recency_w = np.ones(len(self._pos_uidx), dtype=np.float32)

        # Soft negatives are resampled each epoch.
        self._soft_uidx: np.ndarray = np.empty(0, dtype=np.int64)
        self._soft_iidx: np.ndarray = np.empty(0, dtype=np.int64)

        # Cached flat tensors built each epoch.
        self._user_t: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._item_t: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._label_t: torch.Tensor = torch.empty(0, dtype=torch.float32)
        self._weight_t: torch.Tensor = torch.empty(0, dtype=torch.float32)

        self.resample_soft_negatives(epoch=0)

    def resample_soft_negatives(self, epoch: int) -> None:
        """Draw num_soft_neg soft negatives per positive, excluding train-seen
        items per user. Deterministic in (config.seed, epoch)."""
        if self.config.loss_mode == "softmax":
            # Softmax mode draws its negatives in-batch + uniform tail inside
            # train_one_epoch_softmax; the explicit soft-negative pool is
            # unused, so skip the O(n_pos) per-user sampling loop entirely.
            self._soft_uidx = np.empty(0, dtype=np.int64)
            self._soft_iidx = np.empty(0, dtype=np.int64)
            self._rebuild_flat()
            return
        rng = np.random.default_rng(self.config.seed + epoch * 7919)
        cand = self.spec.candidate_item_idx
        n_cand = len(cand)
        n_neg = self.config.n_soft_neg
        if n_neg <= 0 or n_cand == 0:
            self._soft_uidx = np.empty(0, dtype=np.int64)
            self._soft_iidx = np.empty(0, dtype=np.int64)
        else:
            # Vectorized rejection sampling: draw a 2x oversample, filter by
            # not-seen, take the first n_neg per row. Falls back to an explicit
            # per-user rejection loop only for the rows that didn't fill.
            n_pos = len(self._pos_uidx)
            soft_uidx = np.repeat(self._pos_uidx, n_neg)
            soft_iidx = np.empty(n_pos * n_neg, dtype=np.int64)
            # Bulk draw with oversample factor; refine for stragglers.
            oversample = max(2, int(n_neg * 2))
            picks = cand[rng.integers(0, n_cand, size=n_pos * oversample)]
            picks = picks.reshape(n_pos, oversample)
            for r in range(n_pos):
                u = self._pos_uidx[r]
                seen = self.spec.user_seen_per_user_idx.get(int(u), set())
                row_picks = picks[r]
                # Filter out seen items, dedupe, then top n_neg.
                mask = np.array([p not in seen for p in row_picks], dtype=bool)
                kept = row_picks[mask]
                if len(kept) >= n_neg:
                    soft_iidx[r * n_neg:(r + 1) * n_neg] = kept[:n_neg]
                    continue
                # Straggler: keep drawing until we have n_neg distinct unseen items.
                acc: List[int] = list(kept[:n_neg])
                acc_set = set(acc)
                while len(acc) < n_neg:
                    extra = int(cand[rng.integers(0, n_cand)])
                    if extra in seen or extra in acc_set:
                        continue
                    acc.append(extra)
                    acc_set.add(extra)
                soft_iidx[r * n_neg:(r + 1) * n_neg] = acc
            self._soft_uidx = soft_uidx
            self._soft_iidx = soft_iidx

        self._rebuild_flat()

    def _rebuild_flat(self) -> None:
        cfg = self.config
        # The pool is always built (v2); the old flag now only gates whether
        # hard negatives enter the BCE flat tensors -- v1 semantics preserved.
        hard_u = self._hard_uidx if cfg.use_hard_negatives else self._hard_uidx[:0]
        hard_i = self._hard_iidx if cfg.use_hard_negatives else self._hard_iidx[:0]
        u = np.concatenate([self._pos_uidx, hard_u, self._soft_uidx])
        i = np.concatenate([self._pos_iidx, hard_i, self._soft_iidx])
        label = np.concatenate([
            np.ones(len(self._pos_uidx), dtype=np.float32),
            np.zeros(len(hard_u), dtype=np.float32),
            np.zeros(len(self._soft_uidx), dtype=np.float32),
        ])
        weight = np.concatenate([
            # positive_weight * all-ones recency == np.full(..) bit-for-bit
            # when pair_recency_decay_days is None.
            (cfg.positive_weight * self._pos_recency_w).astype(np.float32),
            np.full(len(hard_u), cfg.hard_negative_weight, dtype=np.float32),
            np.full(len(self._soft_uidx), cfg.soft_negative_weight, dtype=np.float32),
        ])
        self._user_t = torch.from_numpy(u.astype(np.int64))
        self._item_t = torch.from_numpy(i.astype(np.int64))
        self._label_t = torch.from_numpy(label)
        self._weight_t = torch.from_numpy(weight)

    def __len__(self) -> int:
        return self._user_t.numel()

    def __getitem__(self, idx: int):
        return (
            self._user_t[idx],
            self._item_t[idx],
            self._label_t[idx],
            self._weight_t[idx],
        )

    @property
    def n_hard_pool(self) -> int:
        """Size of the always-built label==0 pool (independent of the
        use_hard_negatives flag; the softmax auxiliary term draws from it)."""
        return int(len(self._hard_uidx))

    @property
    def positive_pair_weights(self) -> np.ndarray:
        """positive_weight * recency weight per positive pair (float32).
        All equal to positive_weight when pair_recency_decay_days is None."""
        return (self.config.positive_weight * self._pos_recency_w).astype(np.float32)

    # Diagnostics for the JSON report.
    def composition(self) -> Dict[str, int]:
        # n_hard_neg counts hard rows IN THE BCE FLAT TENSORS (0 when the flag
        # is off) -- v1 report semantics. The decoupled pool is n_hard_pool.
        n_hard_flat = int(len(self._hard_uidx)) if self.config.use_hard_negatives else 0
        return {
            "n_positive": int(len(self._pos_uidx)),
            "n_hard_neg": n_hard_flat,
            "n_soft_neg": int(len(self._soft_uidx)),
        }
