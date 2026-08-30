"""Sequence module at the RANKING position.

        logit = DCN(x0) + alpha * g(cand_emb, seq_out)        (RESIDUAL)

    DCN         = the frozen DeepCrossRanker, unchanged (CrossNet(3) +
                  DeepTower(256,128) + head 64 over x0=[dense, cat embs])
    Seq module  = engagement encoder + one of four sequence variants -> [B,d]
    g           = small MLP over [cand_emb, seq_out, cand_emb*seq_out]
    alpha       = learnable scalar, init 0  =>  alpha=0 IS the DCN baseline

Engagement encoder (per position): item_id emb + main_category emb + store emb
+ rating emb (engagement = every interaction, rating kept as a sparse id per
the spec's "minimal set of sparse ids"; item index 1 = <unk> for items first
seen after T0 so OOV events stay valid positions) + position emb, where
position emb is one of
    abs    monthly absolute-time bucket -> fixed sin/cos table -> linear
    delta  log-spaced gap-to-next bucket -> learned emb
    both   sum of the two
Sequence module variants (`SeqConfig.variant`):
    vanilla  bidirectional self-attention encoder, masked mean pool
    mh_pool  multi-head target-aware weighted pooling (DIN generalised:
             query = candidate embedding, keys/values = encoded history)
    causal   causal self-attention encoder, take the LAST valid position
             (SASRec-style user state)
    hstu     Hierarchical Sequential Transduction Unit block (Zhai et al.
             2024): pointwise SiLU-gated attention (no softmax) with a learned
             relative-position bias, then take the last valid position
The module output vector, the candidate embedding and their product are
concatenated with the dense + categorical inputs of the frozen ranker and fed
to the same MLP overarch head. Everything the sequence sees is history_{snap}
(ts < cutoff) -- see train_temporal_ranker._SeqContext.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ranker.ranker_features import RankerFeatureSpec


@dataclass
class SeqConfig:
    variant: str = "causal"          # vanilla | mh_pool | causal | hstu
    pos: str = "delta"               # abs | delta | both
    d: int = 32                      # engagement embedding dim
    layers: int = 1
    heads: int = 2
    ffn_mult: int = 2                # transformer FFN width = ffn_mult * d
    dropout: float = 0.1
    # DCN branch -- identical to the frozen DeepCrossConfig
    cat_emb_dim: int = 16
    n_cross_layers: int = 3
    deep_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dcn_dropout: float = 0.2
    # overarch
    head_hidden_dims: List[int] = field(default_factory=lambda: [64])

    @classmethod
    def parse(cls, spec: str) -> "SeqConfig":
        """'seq:<variant>[:pos=..][:d=..][:L=..][:H=..]'"""
        parts = spec.split(":")
        assert parts[0] == "seq" and len(parts) >= 2, spec
        cfg = cls(variant=parts[1])
        for kv in parts[2:]:
            k, v = kv.split("=")
            if k == "pos": cfg.pos = v
            elif k == "d": cfg.d = int(v)
            elif k == "L": cfg.layers = int(v)
            elif k == "H": cfg.heads = int(v)
            elif k == "ffn": cfg.ffn_mult = int(v)
            elif k == "head": cfg.head_hidden_dims = [int(t) for t in v.split("-")]
            else: raise ValueError(f"unknown seq option {kv!r} in {spec!r}")
        assert cfg.variant in ("vanilla", "mh_pool", "causal", "hstu"), spec
        assert cfg.pos in ("abs", "delta", "both"), spec
        return cfg


def _sincos_table(n: int, d: int) -> torch.Tensor:
    pe = torch.zeros(n, d)
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    pe[0] = 0.0                                   # pad row
    return pe


class EngagementEncoder(nn.Module):
    """<item, cat, store, abs_b, delta_b, rating_b> -> d-dim vector; index 0 = pad."""

    def __init__(self, n_items: int, n_side: Dict[str, int], n_abs: int,
                 n_delta: int, cfg: SeqConfig):
        super().__init__()
        d = cfg.d
        self.item = nn.Embedding(n_items, d, padding_idx=0)
        self.cat = nn.Embedding(n_side["main_category"], d, padding_idx=0)
        self.store = nn.Embedding(n_side["store"], d, padding_idx=0)
        self.rating = nn.Embedding(7, d, padding_idx=0)   # 0 pad, 1..5 stars, 6 unknown
        self.pos = cfg.pos
        if cfg.pos in ("abs", "both"):
            self.register_buffer("abs_table", _sincos_table(n_abs, d))
            self.abs_proj = nn.Linear(d, d, bias=False)
        if cfg.pos in ("delta", "both"):
            self.delta = nn.Embedding(n_delta + 1, d, padding_idx=0)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def item_vec(self, ids3: torch.Tensor) -> torch.Tensor:      # [.., 3]
        return (self.item(ids3[..., 0]) + self.cat(ids3[..., 1])
                + self.store(ids3[..., 2]))

    def forward(self, hist: torch.Tensor) -> torch.Tensor:        # [B, L, 6]
        x = self.item_vec(hist[..., :3]) + self.rating(hist[..., 5])
        if self.pos in ("abs", "both"):
            x = x + self.abs_proj(self.abs_table[hist[..., 3]])
        if self.pos in ("delta", "both"):
            x = x + self.delta(hist[..., 4])
        return self.drop(self.norm(x))


class _HSTUBlock(nn.Module):
    """Sequential Transduction Unit, checked line-by-line against Meta's
    reference (generative_recommenders/research/modeling/sequential/hstu.py):
        u,v,q,k = split(silu(W_uvqk · LN(x)))
        A = silu(q k^T + rab_pos + rab_time) / n   (pointwise, NO softmax)
        A = A ⊙ mask (causal ∧ valid)
        y = o(dropout(u ⊙ LN(A v))) + x
    rab_time = learned bias over the bucketized timespan between the two
    positions (RelativeBucketedTimeAndPositionBasedBias); here the span is
    taken in months from the absolute-month bucket carried per position and
    bucketized log-spaced (0..N_TB)."""

    N_TB = 12                       # timespan buckets: 0,1,2,3-4,5-8,... months

    def __init__(self, d: int, heads: int, max_len: int, dropout: float):
        super().__init__()
        self.h, self.dh, self.n = heads, d // heads, max_len
        self.norm = nn.LayerNorm(d)
        self.uvqk = nn.Linear(d, 4 * d, bias=False)
        self.pos_w = nn.Parameter(torch.zeros(heads, 2 * max_len - 1))
        self.ts_w = nn.Parameter(torch.zeros(heads, self.N_TB + 1))
        nn.init.normal_(self.pos_w, std=0.02); nn.init.normal_(self.ts_w, std=0.02)
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        idx = torch.arange(max_len)
        self.register_buffer("rel_idx", (idx[None, :] - idx[:, None]) + max_len - 1)
        self.register_buffer("causal", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)))

    def _ts_bucket(self, months: torch.Tensor) -> torch.Tensor:  # [B,L,L] >= 0
        # log2-spaced: 0->0, 1->1, 2->2, 3-4->3, 5-8->4, ...
        b = torch.where(months <= 0, torch.zeros_like(months),
                        torch.floor(torch.log2(months.clamp_min(1).float())).long() + 1)
        return b.clamp(max=self.N_TB)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                abs_month: torch.Tensor) -> torch.Tensor:
        # Same math as before; restructured so the two bias terms are single
        # gathers (the [B,h,L,L] advanced-index + permute version was ~6x
        # slower than the transformer variants and timed out at 4h).
        B, L, d = x.shape
        u, v, q, k = F.silu(self.uvqk(self.norm(x))).chunk(4, dim=-1)
        q = q.view(B, L, self.h, self.dh).transpose(1, 2)        # [B,h,L,dh]
        k = k.view(B, L, self.h, self.dh).transpose(1, 2)
        v = v.view(B, L, self.h, self.dh).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2))                # [B,h,L,L]
        pos_b = self.pos_w[:, self.rel_idx[:L, :L]]               # [h,L,L] (tiny)
        span = (abs_month[:, :, None] - abs_month[:, None, :]).abs()   # [B,L,L]
        tb = self._ts_bucket(span)                                # [B,L,L] long
        ts_b = F.embedding(tb, self.ts_w.t())                     # [B,L,L,h]
        att = att + pos_b.unsqueeze(0) + ts_b.permute(0, 3, 1, 2)
        att = F.silu(att) / self.n
        keep = self.causal[:L, :L][None, None] & valid[:, None, None, :]
        att = att.masked_fill(~keep, 0.0)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, L, d)
        y = self.out(self.drop(u * self.out_norm(y)))
        return x + y


class SeqModule(nn.Module):
    def __init__(self, cfg: SeqConfig, max_len: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        if cfg.variant in ("vanilla", "causal"):
            layer = nn.TransformerEncoderLayer(
                d, cfg.heads, dim_feedforward=cfg.ffn_mult * d,
                dropout=cfg.dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg.layers)
        elif cfg.variant == "hstu":
            self.blocks = nn.ModuleList(
                [_HSTUBlock(d, cfg.heads, max_len, cfg.dropout) for _ in range(cfg.layers)])
        elif cfg.variant == "mh_pool":
            # optional L bidirectional encoder layers before the pooling
            # readout (L=0 -> pool the raw engagement embeddings, the DIN
            # generalisation; L>=1 -> transformer + weighted-pooling readout)
            self.enc = None
            if cfg.layers > 0:
                layer = nn.TransformerEncoderLayer(
                    d, cfg.heads, dim_feedforward=cfg.ffn_mult * d,
                    dropout=cfg.dropout, batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, cfg.layers)
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, valid: torch.Tensor,
                cand: torch.Tensor, abs_month: torch.Tensor = None) -> torch.Tensor:
        """x [B,L,d] encoded history (right-aligned, oldest first), valid
        [B,L] bool, cand [B,d] candidate embedding, abs_month [B,L] month
        bucket per position (hstu time bias) -> [B,d]."""
        B, L, d = x.shape
        empty = ~valid.any(dim=1)                                 # no history
        # keep at least one key per row so attention never sees all-masked
        valid_safe = valid.clone(); valid_safe[empty, -1] = True
        v = self.cfg.variant
        if v == "vanilla":
            h = self.enc(x, src_key_padding_mask=~valid_safe)
            w = valid_safe.float().unsqueeze(-1)
            out = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
        elif v == "causal":
            # Right-aligned sequences: a padded query position under a causal
            # mask sees only padded keys -> softmax over -inf -> NaN, which
            # then leaks into later layers as a key. Give every position its
            # own diagonal so no query is fully masked; padded positions are
            # never read (we take the last, valid position).
            H = self.cfg.heads
            allow = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            allow = allow[None] & valid_safe[:, None, :]          # [B,L,L]
            allow = allow | torch.eye(L, dtype=torch.bool, device=x.device)[None]
            am = (~allow).repeat_interleave(H, dim=0)             # [B*H,L,L]
            h = self.enc(x, mask=am)
            out = h[:, -1]                                        # last = newest
        elif v == "hstu":
            h = x
            for blk in self.blocks:
                h = blk(h, valid_safe, abs_month)
            out = h[:, -1]
        else:  # mh_pool: target-aware multi-head weighted pooling
            if self.enc is not None:
                x = self.enc(x, src_key_padding_mask=~valid_safe)
            H, dh = self.cfg.heads, d // self.cfg.heads
            q = self.q_proj(cand).view(B, H, 1, dh)
            k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
            vv = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)       # [B,H,1,L]
            att = att.masked_fill(~valid_safe[:, None, None, :], float("-inf"))
            att = torch.softmax(att, dim=-1)
            out = self.o_proj((att @ vv).transpose(1, 2).reshape(B, d))
        out = self.final_norm(out)
        return out.masked_fill(empty.unsqueeze(-1), 0.0)          # empty -> 0


class SeqRanker(nn.Module):
    """RESIDUAL design:

        logit = DCN(x0)  +  alpha * g([cand_emb, seq_out, cand_emb*seq_out])

    DCN is the frozen DeepCrossRanker layout (same config, same head); alpha
    is a learnable scalar initialised at 0, so at initialisation the model IS
    the DCN baseline exactly (asserted by a structural test) and the sequence branch can
    only learn a residual correction. The previous design fed candidate-id
    embedding + sequence + a NEW head into one MLP, which changed three things
    at once and made the seq-vs-DCN comparison unattributable.
    """

    def __init__(self, spec: RankerFeatureSpec, n_items: int,
                 n_side: Dict[str, int], n_abs: int, n_delta: int,
                 cfg: SeqConfig, hist_table: torch.Tensor, max_len: int = 20):
        super().__init__()
        from scripts.ranker.complex_ranker import DeepCrossRanker, DeepCrossConfig
        self.spec, self.cfg = spec, cfg
        self.register_buffer("hist_table", hist_table.to(torch.int32),
                             persistent=False)
        self.dcn = DeepCrossRanker(spec, DeepCrossConfig())      # exact baseline
        self.encoder = EngagementEncoder(n_items, n_side, n_abs, n_delta, cfg)
        self.seq = SeqModule(cfg, max_len)
        d = cfg.d
        layers: List[nn.Module] = []
        prev = 3 * d
        for h in cfg.head_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.residual = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(()))

    def seq_vector(self, seq__uix: torch.Tensor, seq__cand: torch.Tensor):
        cand = self.encoder.item_vec(seq__cand.long())            # [B,d]
        if self.cfg.variant == "mh_pool":
            seq__hist = self.hist_table.index_select(0, seq__uix.long()).long()
            x = self.encoder(seq__hist)
            valid = seq__hist[..., 0] > 0
            s = self.seq(x, valid, cand, abs_month=seq__hist[..., 3])
        else:
            uniq, inv = torch.unique(seq__uix, return_inverse=True)
            hist_u = self.hist_table.index_select(0, uniq.long()).long()
            x_u = self.encoder(hist_u)
            valid_u = hist_u[..., 0] > 0
            s_u = self.seq(x_u, valid_u, None, abs_month=hist_u[..., 3])
            s = s_u.index_select(0, inv)
        return cand, s

    def forward(self, dense: torch.Tensor, *, seq__uix: torch.Tensor,
                seq__cand: torch.Tensor, **kwargs) -> torch.Tensor:
        base = self.dcn(dense, **kwargs)                          # [B]
        cand, s = self.seq_vector(seq__uix, seq__cand)
        r = self.residual(torch.cat([cand, s, cand * s], dim=-1)).squeeze(-1)
        return base + self.alpha * r
