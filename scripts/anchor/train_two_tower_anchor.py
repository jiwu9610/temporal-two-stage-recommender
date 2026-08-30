"""Leg D: our two-tower under the anchor protocol.

CLI (production conda env `tae`, NOT the recbole venv):
    python -m scripts.anchor.train_two_tower_anchor --category All_Beauty
    [--seed 42] [--epochs N] [--device cpu|cuda] [--smoke]

What this wrapper does (and deliberately nothing more):
    * monkeypatches scripts.retrieval.train_two_tower module globals:
        - PROCESSED_DIR -> data/anchor              (anchor exports use the
          phase1 file layout: train/val/test.parquet + user/item_features.parquet)
        - RESULTS_DIR   -> results/anchor/{cat}     (rule: NEVER
          results/phase1 — the phase1 lineage JSONs live there and the variant
          label change alone would not protect the *_predictions.parquet)
        - DEFAULT_K_RETRIEVE -> 101                 (100+1 so score_predictions'
          valid-item removal still leaves 100 ranked items; Recall@100 in the
          training loop only reads the first 100, so selection is unchanged)
      RAW_DIR stays data/raw: the deeper_category lookup joins raw metadata on
      raw parent_asin — which IS the anchor identity, so no shim is needed.
    * runs t2t.run_category with variant="anchor_v1" and the phase1-v1
      hyperparameters (the exact config recorded in
      results/phase1/{cat}_two_tower_metadata_and_ids.json):
        TwoTowerConfig(64/128/32/16, dropout 0.1, user+item id embs, deeper cat)
        TrainConfig(epochs=100, batch=4096, lr=1e-3, wd=1e-5, patience=8,
                    early stop on val Recall@100)
        PairSamplingConfig(n_soft_neg=4, use_hard_negatives=False  <- hard
                    negatives explicitly OFF, all weights 1.0)
    * post-annotates the emitted JSON with the anchor protocol block: isolation
      declaration + the two spec-required memo disclosures (selection-metric
      asymmetry; lifetime-catalog item-feature aggregates).

Outputs (results/anchor/{cat}/):
    {cat}_two_tower_anchor_v1.json                     report (annotated)
    {cat}_two_tower_anchor_v1_predictions.parquet      per-user top-101 dump
                                                       (train-seen masked ONLY —
                                                       score_predictions removes
                                                       the valid item before
                                                       test scoring)
"""

from __future__ import annotations

import argparse
import json
import sys

from scripts.anchor.anchor_common import (
    ANCHOR_DATA_ROOT,
    ANCHOR_RESULTS_ROOT,
    CATEGORIES,
    ISOLATION_DECLARATION,
    MEMO_LIFETIME_ITEM_FEATURES,
    MEMO_SELECTION_ASYMMETRY,
    MEMO_SLICE_CAVEAT,
    SEEN_MASK_CONVENTIONS,
    write_anchor_memo,
)

VARIANT = "anchor_v1"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100,
                   help="max epochs (phase1-v1 default 100; early stop "
                        "patience 8 on val Recall@100)")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                   help="override TrainConfig.device (default: cuda if available)")
    p.add_argument("--smoke", action="store_true",
                   help="t2t smoke mode (tiny dims, 1 epoch, capped pairs)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    import scripts.retrieval.train_two_tower as t2t
    from scripts.retrieval.two_tower import TwoTowerConfig
    from scripts.retrieval.two_tower_dataset import PairSamplingConfig

    # ---- monkeypatch: anchor in, anchor out -----------------
    t2t.PROCESSED_DIR = ANCHOR_DATA_ROOT
    t2t.RESULTS_DIR = ANCHOR_RESULTS_ROOT / args.category
    t2t.DEFAULT_K_RETRIEVE = 101
    assert "phase1" not in str(t2t.RESULTS_DIR), "leg D must never write results/phase1"
    assert str(t2t.RESULTS_DIR).startswith(str(ANCHOR_RESULTS_ROOT))
    t2t.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_anchor_memo()

    # ---- phase1-v1 hyperparameters, hard negatives OFF -----------------------
    cfg = TwoTowerConfig(
        embedding_dim=64,
        hidden_dim=128,
        id_emb_dim=32,
        cat_emb_dim=16,
        dropout=0.1,
        use_user_id_emb=True,
        use_item_id_emb=True,
        use_deeper_cat_emb=True,
    )
    train_cfg = t2t.TrainConfig(
        epochs=args.epochs,
        batch_size=4096,
        lr=1e-3,
        weight_decay=1e-5,
        early_stopping_patience=8,
    )
    if args.device:
        train_cfg.device = args.device
    pair_cfg = PairSamplingConfig(
        n_soft_neg=4,
        use_hard_negatives=False,   # explicitly OFF (spec P1.2)
        positive_weight=1.0,
        hard_negative_weight=1.0,
        soft_negative_weight=1.0,
        seed=args.seed,
    )

    print(f"[anchor-leg-D] {args.category}: PROCESSED_DIR={t2t.PROCESSED_DIR} "
          f"RESULTS_DIR={t2t.RESULTS_DIR} k_retrieve={t2t.DEFAULT_K_RETRIEVE} "
          f"device={train_cfg.device}", flush=True)

    t2t.run_category(
        category=args.category,
        variant=VARIANT,
        cfg=cfg,
        train_cfg=train_cfg,
        pair_cfg=pair_cfg,
        seed=args.seed,
        smoke=args.smoke,
    )

    # ---- post-annotate the report with the anchor protocol block -------------
    report_path = t2t.RESULTS_DIR / f"{args.category}_two_tower_{VARIANT}.json"
    with open(report_path) as f:
        report = json.load(f)
    report["anchor_protocol"] = {
        "leg": "two_tower_anchor_v1",
        "leg_letter": "D",
        "identity": "raw parent_asin (data/anchor exports; canonicalize.py bypassed)",
        "split": "per-user leave-last-two (data/anchor/{cat}/, P1.1 gated)",
        "hyperparameters": "phase1-v1 (results/phase1/*_two_tower_metadata_and_ids.json "
                           "config), hard negatives explicitly OFF",
        "selection_metric": "val Recall@100 early stopping (phase1 behavior)",
        "selection_asymmetry_memo": MEMO_SELECTION_ASYMMETRY,
        "item_feature_leakage_memo": MEMO_LIFETIME_ITEM_FEATURES,
        "slice_caveat": MEMO_SLICE_CAVEAT,
        "seen_mask": SEEN_MASK_CONVENTIONS["two_tower_dump"],
        "groundtruth_note": (
            "this JSON's val/test metrics use the phase1 evaluator gt "
            "(label==1 positives only) WITHOUT valid-item removal; the "
            "RecBole-comparable numbers (rating-blind gt, valid item removed "
            "from the dump) live in score_predictions' "
            "score_two_tower_anchor_v1.json"
        ),
        "k_retrieve": 101,
        "raw_metadata_note": "deeper_category joined from data/raw metadata on raw "
                             "parent_asin (read-only; identical to anchor identity)",
        "isolation_declaration": ISOLATION_DECLARATION,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[anchor-leg-D] annotated {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
