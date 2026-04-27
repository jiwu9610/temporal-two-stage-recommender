"""Phase 2B — Deep & Cross ranker (complex, 2nd-order + deep).

Architecture (matches the spec's CTR ranking diagram):

    user/item/context sparse embeddings + dense features
        |
        v   (concat into x0)
        |
   +----+----+--------------+
   |         |              |
   |    CrossNet (Branch A)         Deep MLP (Branch B)
   |   x_{l+1} = x0 * (x_l . w_l) + b_l + x_l
   |         |              |
   |         v              v
   |   cross_out      deep_out
   |         \\            /
   |          concat -> prediction MLP -> logit -> BCEWithLogitsLoss
   |
   +-- shared input layer

CrossNet implements explicit second-order (and higher) feature interactions
in O(d) parameters per layer. The deep tower captures implicit higher-order
patterns. Their concatenation feeds a small head that produces a single
logit, trained with pointwise BCE — the same training loop the MLP ranker
uses, so a fair A/B is just a matter of swapping the model.

Deliberately NOT included in v1 (per the user's spec):
  - DIN attention over user histories (requires padded sequence inputs)
  - target-aware item attention
  - multi-head transformer encoders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn

from scripts.ranker.ranker_features import RankerFeatureSpec


@dataclass
class DeepCrossConfig:
    cat_emb_dim: int = 16
    n_cross_layers: int = 3
    deep_hidden_dims: tuple = (256, 128)
    head_hidden_dim: int = 64
    dropout: float = 0.2


class CrossNet(nn.Module):
    """DCN-v1 cross network. Each layer:

        x_{l+1} = x_0 * (x_l . w_l) + b_l + x_l

    where w_l is a [d] vector and b_l is [d]. Per-layer params: 2d (very cheap).
    Captures explicit polynomial-order feature interactions; output dim == input.
    """

    def __init__(self, in_dim: int, n_layers: int):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim).normal_(std=1.0 / max(1.0, in_dim ** 0.5)))
            for _ in range(n_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(in_dim)) for _ in range(n_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            xw = (x * w).sum(dim=-1, keepdim=True)        # [B, 1]
            x = x0 * xw + b + x
        return x


class DeepTower(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepCrossRanker(nn.Module):
    def __init__(self, spec: RankerFeatureSpec, cfg: DeepCrossConfig):
        super().__init__()
        self.spec = spec
        self.cfg = cfg

        self.cat_embs = nn.ModuleDict({
            name: nn.Embedding(spec.cat_vocab_size(name), cfg.cat_emb_dim, padding_idx=0)
            for name in spec.cat_vocabs
        })
        cat_concat_dim = cfg.cat_emb_dim * spec.n_cat
        in_dim = spec.n_dense + cat_concat_dim

        self.cross = CrossNet(in_dim, cfg.n_cross_layers)
        self.deep = DeepTower(in_dim, cfg.deep_hidden_dims, cfg.dropout)
        head_in = in_dim + self.deep.out_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(self, dense: torch.Tensor, **kwargs) -> torch.Tensor:
        feats = [dense]
        for name in self.spec.cat_vocabs:
            idx = kwargs[f"cat__{name}"]
            feats.append(self.cat_embs[name](idx))
        x0 = torch.cat(feats, dim=-1)
        cross_out = self.cross(x0)
        deep_out = self.deep(x0)
        return self.head(torch.cat([cross_out, deep_out], dim=-1)).squeeze(-1)
