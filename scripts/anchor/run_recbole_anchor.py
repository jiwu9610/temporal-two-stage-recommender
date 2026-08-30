"""RecBole anchor legs.

CLI (run inside the recbole-anchor venv):
    python -m scripts.anchor.run_recbole_anchor --category All_Beauty --legs pop,itemknn
    python -m scripts.anchor.run_recbole_anchor --category Video_Games \
        --legs pop,itemknn,sasrec,sasrec_len20,official_sasrec,official_sasrec_len20

Leg registry (letter = spec P1.2 nomenclature):
    pop                   A     RecBole Pop        on our anchor slice
    itemknn               B     RecBole ItemKNN    on our anchor slice
    sasrec                C     RecBole SASRec (max_len 50) on our anchor slice
    sasrec_len20          C-20  RecBole SASRec (max_len 20) on our anchor slice
    official_sasrec       E-50  SASRec (max_len 50), OFFICIAL 5-core VG benchmark files
    official_sasrec_len20 E-20  SASRec (max_len 20), OFFICIAL files  <- LEG E GATE:
                                Recall@10 in [0.070, 0.097] => pipeline/protocol healthy
                                (band anchors use max_len 20, so E-20 carries the gate)
    official_pop          F-pop  \  optional leg F (spec §6-D9: +official Pop/ItemKNN);
    official_itemknn      F-knn  /  registered but not in the default batch leg list
    (leg D, our two-tower, lives in scripts/anchor/train_two_tower_anchor.py)

Protocol per leg:
    * Slice legs: data/anchor/{cat}/{cat}.inter (rows pre-sorted so RecBole's
      stable ``order: TO`` + ``LS: valid_and_test`` reproduces the exported
      leave-last-two split — hard-gated at export time in P1.1).
    * Official legs: RecBole benchmark_filename mode. Sequential benchmark
      files need ``item_id_list:token_seq`` columns, so this script first
      builds them (idempotent) from the official last_out CSVs already on disk
      under data/anchor/_official/benchmark/5core/last_out/ — prefix expansion
      identical to RecBole's own ``data_augmentation`` (all prefixes, truncated
      to the last MAX_ITEM_LIST_LENGTH items). Compute nodes have no internet;
      only the on-disk CSVs are read.
    * Full-catalogue ranking (eval mode 'full'), metrics Recall/NDCG @10/50/100,
      checkpoint selection on val NDCG@10 (RecBole valid_metric) — the
      selection asymmetry vs leg D's val Recall@100 is recorded in every JSON
      and in results/anchor/ANCHOR_MEMO.md.
    * Fixed seed (42), reproducibility=True.

Outputs, per leg:
    results/anchor/{cat}/{leg}.json           protocol + valid/test metrics (+ gate)
    results/anchor/{cat}/{leg}_top101.csv.gz  per-user top-101 dump for score_predictions
    results/anchor/{cat}/_ckpt/, log/, ...    RecBole clutter, kept inside results/anchor

ISOLATION: reads data/anchor only, writes results/anchor (+ derived seq
benchmark files under data/anchor/_official/recbole_seq/). Nothing here may
enter a production pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.anchor.anchor_common import (
    ANCHOR_DATA_ROOT,
    ANCHOR_RESULTS_ROOT,
    CATEGORIES,
    ISOLATION_DECLARATION,
    KS,
    LEG_E_BAND,
    MEMO_SELECTION_ASYMMETRY,
    MEMO_SLICE_CAVEAT,
    OFFICIAL_ROOT,
    SEEN_MASK_CONVENTIONS,
    TOPK_DUMP,
    write_anchor_memo,
)

# Official 5-core VG triple (double-anchored; also gated at fetch time).
OFFICIAL_TRIPLE = {"users": 94_762, "items": 25_612, "interactions": 814_586}

LAST_OUT_DIR = OFFICIAL_ROOT / "benchmark" / "5core" / "last_out"
SEQ_BENCH_ROOT = OFFICIAL_ROOT / "recbole_seq"

LEGS = {
    # name: (letter, model, kind, max_item_list_length)
    "pop": ("A", "Pop", "slice_general", None),
    "itemknn": ("B", "ItemKNN", "slice_general", None),
    "sasrec": ("C", "SASRec", "slice_seq", 50),
    "sasrec_len20": ("C-20", "SASRec", "slice_seq", 20),
    "official_sasrec": ("E-50", "SASRec", "official_seq", 50),
    "official_sasrec_len20": ("E-20", "SASRec", "official_seq", 20),
    "official_pop": ("F-pop", "Pop", "official_general", None),
    "official_itemknn": ("F-knn", "ItemKNN", "official_general", None),
}

GATE_LEGS = {"official_sasrec", "official_sasrec_len20"}  # E-20 is the anchor-comparable one


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---- Official sequential benchmark files (built locally, idempotent) ---------

def build_official_seq_files(max_len: int, rebuild: bool = False) -> Path:
    """Build RecBole *sequential* benchmark atomic files from the official
    last_out CSVs (already on disk; no network).

    Format (hyp1231/AmazonReviews2023 benchmark_scripts seq convention):
        user_id:token \t item_id_list:token_seq \t item_id:token
    train  = all per-user prefixes (>=1 history item), i.e. exactly RecBole's
             own ``data_augmentation`` output, truncated to the last `max_len`;
    valid  = one row/user, history = all train items (last `max_len`);
    test   = one row/user, history = train items + valid item (last `max_len`).
    """
    import pandas as pd

    out_dir = SEQ_BENCH_ROOT / f"len{max_len}" / "Video_Games"
    marker = out_dir / "seq_build_manifest.json"
    expected = [out_dir / f"Video_Games.{p}.inter" for p in ("train", "valid", "test")]
    if not rebuild and marker.exists() and all(p.exists() for p in expected):
        return out_dir

    print(f"[official_seq] building max_len={max_len} benchmark files "
          f"from {LAST_OUT_DIR}", flush=True)
    dtypes = {"user_id": str, "parent_asin": str, "rating": float, "timestamp": "int64"}
    splits = {}
    for name in ("train", "valid", "test"):
        p = LAST_OUT_DIR / f"Video_Games.{name}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run scripts.anchor.fetch_official_vg on the "
                f"login node first (compute nodes have no internet)."
            )
        splits[name] = pd.read_csv(p, dtype=dtypes)

    n_union = sum(len(d) for d in splits.values())
    assert n_union == OFFICIAL_TRIPLE["interactions"], (
        f"official last_out union rows {n_union} != {OFFICIAL_TRIPLE['interactions']}"
    )
    for name in ("valid", "test"):
        assert len(splits[name]) == OFFICIAL_TRIPLE["users"], (
            f"official {name} rows {len(splits[name])} != one per user"
        )
        assert splits[name]["user_id"].is_unique

    # Stable order: official CSV row order is the protocol's tie-break; a
    # mergesort by (user, timestamp) preserves it within equal timestamps.
    tr = splits["train"].sort_values(["user_id", "timestamp"], kind="mergesort")
    train_items = tr.groupby("user_id", sort=True)["parent_asin"].agg(list)
    valid_item = splits["valid"].set_index("user_id")["parent_asin"]
    test_item = splits["test"].set_index("user_id")["parent_asin"]
    assert set(train_items.index) == set(valid_item.index) == set(test_item.index)

    def rows_train():
        for uid, items in train_items.items():
            for j in range(1, len(items)):
                yield uid, items[max(0, j - max_len):j], items[j]

    def rows_valid():
        for uid, items in train_items.items():
            yield uid, items[-max_len:], valid_item[uid]

    def rows_test():
        for uid, items in train_items.items():
            hist = items + [valid_item[uid]]
            yield uid, hist[-max_len:], test_item[uid]

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, gen in (("train", rows_train), ("valid", rows_valid), ("test", rows_test)):
        path = out_dir / f"Video_Games.{name}.inter"
        n = 0
        with open(path, "w") as f:
            f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
            for uid, hist, target in gen():
                assert 0 < len(hist) <= max_len
                f.write(f"{uid}\t{' '.join(hist)}\t{target}\n")
                n += 1
        counts[name] = n
        print(f"[official_seq] wrote {path.name}: {n:,} rows", flush=True)

    expected_train_rows = int((train_items.str.len() - 1).sum())
    assert counts["train"] == expected_train_rows
    assert counts["valid"] == counts["test"] == OFFICIAL_TRIPLE["users"]

    with open(marker, "w") as f:
        json.dump({
            "created_utc": _now_iso(),
            "max_item_list_length": max_len,
            "source": str(LAST_OUT_DIR),
            "row_counts": counts,
            "construction": (
                "prefix expansion identical to RecBole data_augmentation "
                "(all prefixes with >=1 history item), history truncated to the "
                "last max_len items; valid history = train items; test history "
                "= train + valid item. Stable (user, timestamp) mergesort "
                "preserves official CSV row order under timestamp ties."
            ),
            "official_triple": OFFICIAL_TRIPLE,
            "isolation_declaration": ISOLATION_DECLARATION,
        }, f, indent=2)
    return out_dir


# ---- Config assembly ---------------------------------------------------------

def make_config_dict(leg: str, category: str, args) -> tuple[dict, str]:
    """Return (config_dict, dataset_name) for one leg."""
    import torch

    letter, model, kind, max_len = LEGS[leg]
    out_dir = ANCHOR_RESULTS_ROOT / category

    cd = {
        "checkpoint_dir": str(out_dir / "_ckpt"),
        "seed": args.seed,
        "reproducibility": True,
        "use_gpu": (not args.cpu) and torch.cuda.is_available(),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "metrics": ["Recall", "NDCG"],
        "topk": list(KS),
        "valid_metric": "NDCG@10",
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "group_by": "user",
            "mode": "full",
        },
        "epochs": args.epochs,
        "stopping_step": args.stopping_step,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "show_progress": False,
        "log_wandb": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "benchmark_filename": None,
    }

    if kind == "slice_general":
        cd["data_path"] = str(ANCHOR_DATA_ROOT)
        cd["load_col"] = {"inter": ["user_id", "item_id", "rating", "timestamp"]}
        dataset = category
    elif kind == "slice_seq":
        cd["data_path"] = str(ANCHOR_DATA_ROOT)
        cd["load_col"] = {"inter": ["user_id", "item_id", "rating", "timestamp"]}
        cd["MAX_ITEM_LIST_LENGTH"] = max_len
        cd["train_neg_sample_args"] = None  # CE loss, full softmax
        dataset = category
    elif kind == "official_general":
        assert category == "Video_Games", "official legs are Video_Games-only"
        cd["data_path"] = str(OFFICIAL_ROOT)
        cd["load_col"] = {"inter": ["user_id", "item_id", "rating", "timestamp"]}
        cd["benchmark_filename"] = ["train", "valid", "test"]
        dataset = "Video_Games"
    elif kind == "official_seq":
        assert category == "Video_Games", "official legs are Video_Games-only"
        seq_dir = build_official_seq_files(max_len, rebuild=args.rebuild_seq)
        cd["data_path"] = str(seq_dir.parent)  # .../recbole_seq/len{L}
        cd["load_col"] = {"inter": ["user_id", "item_id_list", "item_id"]}
        # CRITICAL: without this alias, item_id_list tokens get a vocabulary
        # separate from item_id — item_num drops (25,610 vs 25,612 measured)
        # and list indices silently mis-address the item embedding table.
        cd["alias_of_item_id"] = ["item_id_list"]
        cd["benchmark_filename"] = ["train", "valid", "test"]
        cd["MAX_ITEM_LIST_LENGTH"] = max_len
        cd["train_neg_sample_args"] = None
        dataset = "Video_Games"
    else:  # pragma: no cover
        raise ValueError(kind)
    return cd, dataset


# ---- Top-K dump --------------------------------------------------------------

def dump_topk(model, dataset, test_data, k: int, device, out_path: Path) -> dict:
    """Per-user top-k over the full catalogue, mirroring RecBole's own
    ``Trainer._full_sort_batch_eval`` masking exactly:
      * [pad] column -inf always;
      * general models: history_index (train+valid at test) -inf;
      * sequential models: no history masking (history_index is None).
    Ties: torch.topk resolves by ascending internal item id — deterministic
    for a fixed .inter file + seed.
    """
    import numpy as np
    import pandas as pd
    import torch

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    tot_item_num = dataset.item_num
    k_eff = min(k, tot_item_num - 1)

    model.eval()
    users, items, ranks, scores_out = [], [], [], []
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, _pos_u, _pos_i = batched_data
            scores = model.full_sort_predict(interaction.to(device))
            scores = scores.view(-1, tot_item_num)
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf
            topv, topi = torch.topk(scores, k_eff, dim=1)
            uid_tokens = dataset.id2token(uid_field, interaction[uid_field].cpu().numpy())
            item_tokens = dataset.id2token(iid_field, topi.cpu().numpy())
            n_rows = len(uid_tokens)
            users.append(np.repeat(np.asarray(uid_tokens, dtype=object), k_eff))
            items.append(np.asarray(item_tokens, dtype=object).reshape(-1))
            ranks.append(np.tile(np.arange(1, k_eff + 1), n_rows))
            scores_out.append(topv.cpu().numpy().reshape(-1))

    df = pd.DataFrame({
        "user_id": np.concatenate(users),
        "parent_asin": np.concatenate(items),
        "rank": np.concatenate(ranks),
        "score": np.concatenate(scores_out),
    })
    df.to_csv(out_path, index=False)  # .csv.gz suffix => gzip inferred
    return {"path": str(out_path), "k": int(k_eff), "rows": int(len(df)),
            "n_users": int(df["user_id"].nunique())}


# ---- One leg -----------------------------------------------------------------

def run_leg(leg: str, category: str, args) -> dict:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    letter, model_name, kind, max_len = LEGS[leg]
    out_dir = ANCHOR_RESULTS_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cd, dataset_name = make_config_dict(leg, category, args)
    config = Config(model=model_name, dataset=dataset_name, config_dict=cd)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    data_stats = {
        "n_users_incl_pad": int(dataset.user_num),
        "n_items_incl_pad": int(dataset.item_num),
        "n_interactions": int(dataset.inter_num),
    }
    print(f"[{category}/{leg}] {model_name} kind={kind} "
          f"users={dataset.user_num - 1} items={dataset.item_num - 1} "
          f"inter={dataset.inter_num} device={'gpu' if config['use_gpu'] else 'cpu'}",
          flush=True)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])  # RecBole quickstart does this too

    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=False,
    )
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    test_metrics = {k: float(v) for k, v in dict(test_result).items()}
    valid_metrics = {k: float(v) for k, v in dict(best_valid_result).items()}
    print(f"[{category}/{leg}] valid={valid_metrics}", flush=True)
    print(f"[{category}/{leg}] test ={test_metrics}", flush=True)

    dump_info = None
    if args.topk_dump > 0:
        dump_path = out_dir / f"{leg}_top{args.topk_dump}.csv.gz"
        try:
            dump_info = dump_topk(model, dataset, test_data, args.topk_dump,
                                  config["device"], dump_path)
            print(f"[{category}/{leg}] dump -> {dump_path} "
                  f"({dump_info['rows']:,} rows)", flush=True)
        except NotImplementedError:
            dump_info = {"skipped": f"{model_name} has no full_sort_predict"}
            print(f"[{category}/{leg}] dump SKIPPED: no full_sort_predict", flush=True)

    gate = None
    if leg in GATE_LEGS:
        r10 = test_metrics.get("recall@10")
        gate = {
            "gate": "leg_E_official_sasrec_recall@10_band",
            "band": list(LEG_E_BAND),
            "measured_recall@10": r10,
            "passed": (r10 is not None and LEG_E_BAND[0] <= r10 <= LEG_E_BAND[1]),
            "anchor_comparable": leg == "official_sasrec_len20",
            "note": "band anchors (Pctx / LLaDA-Rec) use max_len 20; the len20 "
                    "leg carries the verdict, len50 is informative.",
        }
        print(f"[{category}/{leg}] LEG E GATE: R@10={r10} band={LEG_E_BAND} "
              f"passed={gate['passed']}", flush=True)

    seen_mask = (SEEN_MASK_CONVENTIONS["recbole_sequential_full_sort"]
                 if "seq" in kind else
                 SEEN_MASK_CONVENTIONS["recbole_general_full_sort"])

    def cfg_get(key):
        try:
            v = config[key]
        except Exception:
            return None
        return v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)

    report = {
        "category": category,
        "leg": leg,
        "leg_letter": letter,
        "model": model_name,
        "started_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "recbole": __import__("recbole").__version__,
            "torch": __import__("torch").__version__,
            "device": str(config["device"]),
            "venv_hint": sys.prefix,
        },
        "protocol": {
            "identity": "raw parent_asin (canonicalize.py bypassed — anchor identity)",
            "dataset_kind": kind,
            "split": ("official last_out benchmark files (RecBole benchmark_filename "
                      "mode; no re-splitting)" if kind.startswith("official")
                      else "per-user leave-one-out via RecBole LS:valid_and_test, "
                           "order TO on the pre-sorted anchor .inter (export gate "
                           "asserts identity with leave_last_two parquet)"),
            "ranking": "full catalogue (eval mode 'full')",
            "seen_mask": seen_mask,
            "selection_metric": "val NDCG@10 (RecBole valid_metric)",
            "selection_asymmetry_memo": MEMO_SELECTION_ASYMMETRY,
            "slice_caveat": None if kind.startswith("official") else MEMO_SLICE_CAVEAT,
            "max_item_list_length": max_len,
            "seed": args.seed,
            "isolation_declaration": ISOLATION_DECLARATION,
        },
        "config_snapshot": {
            key: cfg_get(key) for key in [
                "epochs", "stopping_step", "learning_rate", "train_batch_size",
                "eval_batch_size", "train_neg_sample_args", "MAX_ITEM_LIST_LENGTH",
                "n_layers", "n_heads", "hidden_size", "inner_size", "loss_type",
                "k", "shrink", "data_path", "benchmark_filename",
            ]
        },
        "data_stats": data_stats,
        "best_valid_score": (None if best_valid_score is None else float(best_valid_score)),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "topk_dump": dump_info,
        "gate": gate,
        "saved_model_file": getattr(trainer, "saved_model_file", None),
    }

    out_path = out_dir / f"{leg}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{category}/{leg}] wrote {out_path} "
          f"({report['elapsed_seconds']}s)", flush=True)
    return report


# ---- CLI ---------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--legs", required=True,
                   help=f"comma-separated from {sorted(LEGS)}")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip any leg whose results JSON already exists "
                        "(resubmission economy after a tail failure)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200,
                   help="max epochs (early stop on val NDCG@10, stopping_step)")
    p.add_argument("--stopping-step", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--topk-dump", type=int, default=TOPK_DUMP,
                   help="per-user top-K dump size (0 disables); default 101 so "
                        "score_predictions' valid-item removal leaves >=100")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is up")
    p.add_argument("--rebuild-seq", action="store_true",
                   help="rebuild the derived official sequential benchmark files")
    return p.parse_args(argv)


def main(argv=None):
    try:
        import recbole  # noqa: F401
    except ImportError:
        print("recbole not importable — activate the anchor venv first:\n"
              "  source <recbole-anchor-venv>/bin/activate",
              file=sys.stderr)
        return 2

    args = _parse_args(argv)
    legs = [x.strip() for x in args.legs.split(",") if x.strip()]
    for leg in legs:
        if leg not in LEGS:
            print(f"unknown leg {leg!r}; choose from {sorted(LEGS)}", file=sys.stderr)
            return 2
        if LEGS[leg][2].startswith("official") and args.category != "Video_Games":
            print(f"leg {leg} is Video_Games-only", file=sys.stderr)
            return 2

    out_dir = ANCHOR_RESULTS_ROOT / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    # RecBole writes ./log and ./log_tensorboard under cwd — keep the clutter
    # inside results/anchor/{cat}/ (results/anchor is anchor-owned).
    os.chdir(out_dir)
    write_anchor_memo()

    for leg in legs:
        leg_json = out_dir / f"{leg}.json"
        if args.skip_existing and leg_json.exists():
            print(f"[{args.category}/{leg}] skip: {leg_json.name} already exists "
                  f"(--skip-existing; delete the JSON to force a rerun)", flush=True)
            continue
        run_leg(leg, args.category, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
