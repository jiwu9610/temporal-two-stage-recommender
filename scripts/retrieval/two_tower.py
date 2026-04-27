"""Two-tower retrieval model — clean v1.

User tower:
    optional user_id embedding  +  user dense features (USER_DENSE_COLS)
    -> 2-layer MLP -> embedding_dim user vector

Item tower:
    optional parent_asin embedding  +  store_emb  +  main_category_emb
    +  optional deeper_category_emb  +  item dense features (ITEM_DENSE_COLS)
    -> 2-layer MLP -> embedding_dim item vector

Score:
    score(u, i) = dot(user_vec, item_vec)

Toggleable id embeddings (`use_user_id_emb`, `use_item_id_emb`) let us A/B
metadata-only vs metadata+ids to demonstrate whether the model is generalizing
on metadata or memorizing frequent ids — the design concern flagged in review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, n_users: int, n_dense: int, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.user_id_emb: Optional[nn.Embedding] = None
        if cfg.use_user_id_emb:
            self.user_id_emb = nn.Embedding(n_users, cfg.id_emb_dim, padding_idx=0)
        in_dim = (cfg.id_emb_dim if cfg.use_user_id_emb else 0) + n_dense
        self.head = _TowerMLP(in_dim, cfg.hidden_dim, cfg.embedding_dim, cfg.dropout)

    def forward(self, user_idx: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.user_id_emb is not None:
            feats.append(self.user_id_emb(user_idx))
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

    def encode_users(self, user_idx, user_dense) -> torch.Tensor:
        return self.user_tower(user_idx, user_dense)

    def encode_items(self, item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense) -> torch.Tensor:
        return self.item_tower(item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense)

    def forward(
        self,
        user_idx, user_dense,
        item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense,
    ) -> torch.Tensor:
        u = self.encode_users(user_idx, user_dense)
        i = self.encode_items(item_idx, store_idx, main_cat_idx, deeper_cat_idx, item_dense)
        # element-wise product summed -> dot product per row
        return (u * i).sum(dim=-1) * self.logit_scale
