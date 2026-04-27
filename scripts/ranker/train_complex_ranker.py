"""Phase 2B — train the Deep+Cross ranker (complex, 2nd-order + deep).

CLI:
    python -m scripts.ranker.train_complex_ranker
        --category Video_Games
        [--epochs 20] [--batch-size 8192] [--lr 1e-3]
        [--cat-emb-dim 16] [--n-cross-layers 3]
        [--deep-hidden-dims 256,128] [--head-hidden-dim 64] [--dropout 0.2]

Reads:
    data/processed/{category}/candidates.parquet  (built by candidate_builder)

Writes:
    results/phase2/{category}_complex_ranker.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Tuple

from scripts.ranker.complex_ranker import DeepCrossConfig, DeepCrossRanker
from scripts.ranker.ranker_features import RankerFeatureSpec
from scripts.ranker.train_runner import run


def _parse_hidden_dims(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   choices=["All_Beauty", "Video_Games", "Books", "Electronics"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--cat-emb-dim", type=int, default=16)
    p.add_argument("--n-cross-layers", type=int, default=3)
    p.add_argument("--deep-hidden-dims", type=str, default="256,128")
    p.add_argument("--head-hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cfg = DeepCrossConfig(
        cat_emb_dim=args.cat_emb_dim,
        n_cross_layers=args.n_cross_layers,
        deep_hidden_dims=_parse_hidden_dims(args.deep_hidden_dims),
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    )

    def _build(spec: RankerFeatureSpec):
        return DeepCrossRanker(spec, cfg)

    run(
        category=args.category,
        ranker_name="complex_ranker",
        build_model=_build,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        extra_config=asdict(cfg),
    )


if __name__ == "__main__":
    main()
