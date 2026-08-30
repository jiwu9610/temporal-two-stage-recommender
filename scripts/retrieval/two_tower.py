"""Two-tower retrieval model — v1 core + v2 extensions.

User tower:
    optional user_id embedding  +  user dense features (USER_DENSE_COLS)
    [+ v2 taste channel: pooled positive-history item embeddings, sharing the
     item tower's item_id_emb table (Covington et al., RecSys 2016), gated by
     use_hist_pool -- default OFF, v1 behavior untouched]
    -> 2-layer MLP -> embedding_dim user vector

Item tower:
    optional parent_asin embedding  +  store_emb  +  main_category_emb
    +  optional deeper_category_emb  +  item dense features (ITEM_DENSE_COLS)
    -> 2-layer MLP -> embedding_dim item vector

Score:
    score(u, i) = dot(user_vec, item_vec) * logit_scale (clamped to [1, 100])

Toggleable id embeddings (`use_user_id_emb`, `use_item_id_emb`) let us A/B
metadata-only vs metadata+ids to demonstrate whether the model is generalizing
on metadata or memorizing frequent ids.

v2 additions (all default-off / identity so v1 checkpoints and v1 training
runs reproduce bit-for-bit):
  * taste channel (`use_hist_pool`): weighted mean-pool of the user's most
    recent L positive-history item embeddings, target-masked during training;
  * `build_optimizer` helper: 'adam' == v1 construction; 'adamw' = decoupled
    weight decay with wd=0 on every embedding table + logit_scale
    (Loshchilov & Hutter, ICLR 2019);
  * logit_scale clamp [1, 100] -- exact identity (value and gradient) while
    the parameter stays in range, which it does from the v1 init of 10.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# v2: clamp bounds for the learnable dot-product temperature. Identity for the
# v1 init (10.0); only binds if training drives the scale to a degenerate value.
LOGIT_SCALE_MIN = 1.0
LOGIT_SCALE_MAX = 100.0


@dataclass
class TwoTowerConfig:
    embedding_dim: int = 64                 # output of each tower / dot-product dim
    hidden_dim: int = 128                   # hidden layer of the per-tower MLP
    id_emb_dim: int = 32                    # categorical id embedding dim (user/item ids)
    cat_emb_dim: int = 16                   # categorical metadata embedding dim (store, cat, deeper_cat)
    dropout: float = 0.1
    use_user_id_emb: bool = True
    use_item_id_emb: bool = True
    use_deeper_cat_emb: bool = True
    # ---- v2. Tail-appended with defaults that keep
    # v1 behavior byte-identical; asdict() -> _config_hash covers them.
    use_hist_pool: bool = False             # taste channel; temporal driver only
    hist_pool_len: int = 20                 # L most recent positive history items
    hist_pool_recency_days: Optional[float] = None  # None = uniform mean-pool


class _TowerMLP(nn.Module):
    """Shared 2-layer MLP head. Output is L2-normalized so the eventual dot
    product behaves like cosine similarity (helps optimization stability when
    the user can also include an unnormalized id embedding alongside dense
    features whose magnitudes vary widely)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class UserTower(nn.Module):
    def __init__(self, n_users: int, n_dense: int, cfg: TwoTowerConfig,
                 shared_item_emb: Optional[nn.Embedding] = None):
        super().__init__()
        self.cfg = cfg
        self.use_hist_pool = bool(cfg.use_hist_pool)
        self.user_id_emb: Optional[nn.Embedding] = None
        if cfg.use_user_id_emb:
            self.user_id_emb = nn.Embedding(n_users, cfg.id_emb_dim, padding_idx=0)
        # v2 taste channel: the pooled history embeddings come from the ITEM
        # tower's item_id_emb table (shared module, not a copy) so user taste
        # and item identity live in the same space.
        self.shared_item_emb: Optional[nn.Embedding] = None
        if self.use_hist_pool:
            assert shared_item_emb is not None, (
                "use_hist_pool=True requires the item tower's item_id_emb table "
                "(use_item_id_emb must be True; construct ItemTower first and "
                "pass item_tower.item_id_emb)"
            )
            self.shared_item_emb = shared_item_emb
        in_dim = (
            (cfg.id_emb_dim if cfg.use_user_id_emb else 0)
            + (self.shared_item_emb.embedding_dim if self.use_hist_pool else 0)
            + n_dense
        )
        self.head = _TowerMLP(in_dim, cfg.hidden_dim, cfg.embedding_dim, cfg.dropout)

    def pool_history(
        self,
        hist_idx: torch.Tensor,                     # [B, L] item idxs, PAD_IDX-padded
        hist_w: torch.Tensor,                       # [B, L] weights, 0 on padding
        exclude_item_idx: Optional[torch.Tensor] = None,  # [B] target masking
    ) -> torch.Tensor:
        """Weighted mean-pool of history item embeddings.

        Target masking: slots equal to `exclude_item_idx` get weight 0 so the
        model never sees the item being scored inside its own input (training
        pairs are drawn FROM the history). Zero-history rows (all weights 0,
        possibly after masking) return the EXACT zero vector: the weighted sum
        is 0 and the denominator is clamped, so nothing leaks from the PAD row
        (which is itself zeros via padding_idx=0)."""
        assert self.shared_item_emb is not None, "pool_history requires use_hist_pool"
        w = hist_w
        if exclude_item_idx is not None:
            w = w * (hist_idx != exclude_item_idx.unsqueeze(1)).to(w.dtype)
        emb = self.shared_item_emb(hist_idx)         # [B, L, D]
        pooled = (emb * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return pooled / denom

    def forward(
        self,
        user_idx: torch.Tensor,
        dense: torch.Tensor,
        hist_idx: Optional[torch.Tensor] = None,
        hist_w: Optional[torch.Tensor] = None,
        hist_exclude_item_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = []
        if self.user_id_emb is not None:
            feats.append(self.user_id_emb(user_idx))
        if self.use_hist_pool:
            assert hist_idx is not None and hist_w is not None, (
                "use_hist_pool=True but no history tensors were passed -- the "
                "FeatureSpec must be built with build_hist_pool=True (temporal "
                "driver only) and spec_t must carry user_hist_idx/user_hist_w"
            )
            feats.append(self.pool_history(hist_idx, hist_w, hist_exclude_item_idx))
        feats.append(dense)
        return self.head(torch.cat(feats, dim=-1))


class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_stores: int,
        n_main_cats: int,
        n_deeper_cats: int,
        n_dense: int,
        cfg: TwoTowerConfig,
        has_deeper_cat: bool,
    ):
        super().__init__()
        self.cfg = cfg
        self.has_deeper_cat = has_deeper_cat and cfg.use_deeper_cat_emb

        self.item_id_emb: Optional[nn.Embedding] = None
        if cfg.use_item_id_emb:
            self.item_id_emb = nn.Embedding(n_items, cfg.id_emb_dim, padding_idx=0)
        self.store_emb = nn.Embedding(n_stores, cfg.cat_emb_dim, padding_idx=0)
        self.main_cat_emb = nn.Embedding(n_main_cats, cfg.cat_emb_dim, padding_idx=0)
        self.deeper_cat_emb: Optional[nn.Embedding] = None
        if self.has_deeper_cat:
            self.deeper_cat_emb = nn.Embedding(n_deeper_cats, cfg.cat_emb_dim, padding_idx=0)

        in_dim = (
            (cfg.id_emb_dim if cfg.use_item_id_emb else 0)
            + cfg.cat_emb_dim                          # store
            + cfg.cat_emb_dim                          # main_category
            + (cfg.cat_emb_dim if self.has_deeper_cat else 0)
            + n_dense
        )
        self.head = _TowerMLP(in_dim, cfg.hidden_dim, cfg.embedding_dim, cfg.dropout)

    def forward(
        self,
        item_idx: torch.Tensor,
        store_idx: torch.Tensor,
        main_cat_idx: torch.Tensor,
        deeper_cat_idx: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        feats = []
        if self.item_id_emb is not None:
            feats.append(self.item_id_emb(item_idx))
        feats.append(self.store_emb(store_idx))
        feats.append(self.main_cat_emb(main_cat_idx))
        if self.deeper_cat_emb is not None:
            feats.append(self.deeper_cat_emb(deeper_cat_idx))
        feats.append(dense)
        return self.head(torch.cat(feats, dim=-1))


class TwoTower(nn.Module):
    """Wrap user + item towers; forward returns a logit (un-sigmoided dot product
    optionally rescaled by a learnable temperature)."""

    def __init__(self, user_tower: UserTower, item_tower: ItemTower, init_logit_scale: float = 10.0):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        # Both towers L2-normalize their output, so dot product is in [-1, 1].
        # A learnable temperature lets BCE actually saturate; without it the
        # logits stay tiny and BCE ignores label structure.
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def clamped_logit_scale(self) -> torch.Tensor:
        """v2: clamp to [LOGIT_SCALE_MIN, LOGIT_SCALE_MAX]. Exact identity in
        value AND gradient while the parameter stays in range (it starts at
        10.0), so v1 trajectories are untouched."""
        return self.logit_scale.clamp(min=LOGIT_SCALE_MIN, max=LOGIT_SCALE_MAX)

    def encode_users(self, user_idx, user_dense, hist_idx=None, hist_w=None,
                     hist_exclude_item_idx=None) -> torch.Tensor:
        return self.user_tower(user_idx, user_dense, hist_idx, hist_w,
                               hist_exclude_item_idx)

    def encode_items(self, item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense) -> torch.Tensor:
        return self.item_tower(item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense)

    def forward(
        self,
        user_idx, user_dense,
        item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense,
        hist_idx=None, hist_w=None, hist_exclude_item_idx=None,
    ) -> torch.Tensor:
        u = self.encode_users(user_idx, user_dense, hist_idx, hist_w,
                              hist_exclude_item_idx)
        i = self.encode_items(item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense)
        # element-wise product summed -> dot product per row
        return (u * i).sum(dim=-1) * self.clamped_logit_scale()


# ---- v2 optimizer helpers ----------------------------------------------------

def split_decay_param_groups(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[str]]:
    """Two-group AdamW split: every nn.Embedding weight plus the logit_scale
    temperature goes in the no-decay group; everything else (tower MLP weights
    and biases) decays. Shared parameters (the taste channel's shared_item_emb
    is the item tower's table) are deduplicated by identity.

    Returns (decay_params, no_decay_params, no_decay_names)."""
    no_decay_ids = {
        id(mod.weight)
        for mod in model.modules()
        if isinstance(mod, nn.Embedding)
    }
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    no_decay_names: List[str] = []
    seen = set()
    for name, p in model.named_parameters():   # remove_duplicate=True by default
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in no_decay_ids or name.split(".")[-1] == "logit_scale":
            no_decay.append(p)
            no_decay_names.append(name)
        else:
            decay.append(p)
    assert len(decay) + len(no_decay) == len(seen), (
        "param-group split lost or duplicated parameters"
    )
    return decay, no_decay, no_decay_names


def build_optimizer(model: nn.Module, optimizer: str, lr: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    """'adam' reproduces the v1 construction exactly (single group, L2-in-
    gradient decay). 'adamw' is the v2 path: decoupled weight decay
    (Loshchilov & Hutter, ICLR 2019) with weight_decay=0 on all embedding
    parameter groups + logit_scale."""
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    if optimizer == "adamw":
        decay, no_decay, _ = split_decay_param_groups(model)
        assert no_decay, "adamw split found no embedding/logit_scale parameters"
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
        )
    raise ValueError(f"unknown optimizer {optimizer!r}; expected 'adam' or 'adamw'")
