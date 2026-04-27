"""Phase 2A — MLP ranker (simple baseline).

dense + sparse-embedded features  ->  3-layer MLP  ->  logit  ->  BCE.

Designed to be the cheap baseline against the Deep+Cross ranker. Same input
tensors (built by ranker_features.build_tensors), same training loop scaffold,
same evaluator (rerank Recall@K / Precision@K).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from scripts.ranker.ranker_features import RankerFeatureSpec


@dataclass
class MLPRankerConfig:
    cat_emb_dim: int = 16
    hidden_dims: tuple = (256, 128, 64)
    dropout: float = 0.2


class MLPRanker(nn.Module):
    def __init__(self, spec: RankerFeatureSpec, cfg: MLPRankerConfig):
        super().__init__()
        self.spec = spec
        self.cfg = cfg

        self.cat_embs = nn.ModuleDict({
            name: nn.Embedding(spec.cat_vocab_size(name), cfg.cat_emb_dim, padding_idx=0)
            for name in spec.cat_vocabs
        })
        cat_concat_dim = cfg.cat_emb_dim * spec.n_cat
        in_dim = spec.n_dense + cat_concat_dim

        layers: List[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, dense: torch.Tensor, **kwargs) -> torch.Tensor:
        feats = [dense]
        for name in self.spec.cat_vocabs:
            idx = kwargs[f"cat__{name}"]
            feats.append(self.cat_embs[name](idx))
        x = torch.cat(feats, dim=-1)
        return self.mlp(x).squeeze(-1)
