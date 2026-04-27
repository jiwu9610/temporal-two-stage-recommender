"""Per-category two-tower training + retrieval evaluation.

CLI:
    python -m scripts.retrieval.train_two_tower
        --category Video_Games
        [--variant metadata_only|metadata_and_ids]
        [--embedding-dim 64] [--hidden-dim 128]
        [--epochs 5] [--batch-size 4096] [--lr 1e-3]
        [--num-soft-negatives 4] [--use-hard-negatives]
        [--smoke]

Pipeline (spec):
    1. Load Phase 0 artifacts + post-2026-04-26 raw metadata for deeper_cat.
    2. Preflight asserts: refreshed metadata is in use (n_features /
       n_description / n_categories non-zero for VG/Books/Electronics).
    3. Build FeatureSpec (vocabs, dense feature tables, train-seen sets).
    4. Build TwoTowerPairDataset (positives + optional hard-negs + per-epoch
       resampled soft-negs).
    5. Train with explicit pointwise BCE-with-logits, per-pair weights,
       val Recall@100 early stopping.
    6. Run retrieval inference (chunked matmul over candidate pool, exclude
       train-seen, top-K), evaluate on val + test, write JSON.

Output:
    results/phase1/{category}_two_tower_{variant}.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.retrieval.evaluator import build_groundtruth, build_split_report
from scripts.retrieval.two_tower import (
    ItemTower,
    TwoTower,
    TwoTowerConfig,
    UserTower,
)
from scripts.retrieval.two_tower_dataset import (
    FeatureSpec,
    PairSamplingConfig,
    TwoTowerPairDataset,
    build_feature_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results" / "phase1"
DEFAULT_KS = (10, 50, 100)
DEFAULT_K_RETRIEVE = 100


# Categories where the post-2026-04-26 metadata refresh should populate the
# list-feature columns. All_Beauty's source `categories` field is empty in the
# upstream HF dump, so we can't require it there.
RICH_METADATA_CATEGORIES = {"Video_Games", "Books", "Electronics"}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---- Preflight ---------------------------------------------------------------

def metadata_preflight(category: str, item_features: pd.DataFrame) -> Dict[str, bool]:
    """Assert the refreshed metadata is actually populated. Catches the case
    where someone runs against the pre-2026-04-26 8-col raw metadata and the
    list features are all zero (all the regret of training a metadata-first
    model against zeroed signals).
    """
    checks = {}
    checks["n_features_nonzero"] = bool((item_features["n_features"] > 0).any())
    checks["n_description_nonzero"] = bool((item_features["n_description"] > 0).any())
    checks["n_categories_nonzero"] = bool((item_features["n_categories"] > 0).any())

    must = ["n_features_nonzero", "n_description_nonzero"]
    if category in RICH_METADATA_CATEGORIES:
        must.append("n_categories_nonzero")
    failed = [k for k in must if not checks[k]]
    if failed:
        raise RuntimeError(
            f"[preflight] {category}: refreshed metadata is missing -- {failed}. "
            f"Run scripts/data/refresh_metadata.py before training."
        )
    return checks


# ---- Training ----------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    early_stopping_patience: int = 8
    eval_every_epoch: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_n_batches: int = 100
    # Smoke / dev caps. None = no cap. The smoke flag in run_category wires
    # these to small numbers so a smoke loop completes in seconds, not minutes.
    max_train_pairs: int = 0          # 0 = no cap
    max_eval_users: int = 0           # 0 = no cap


def _move_spec_to_tensors(spec: FeatureSpec, device: str) -> Dict[str, torch.Tensor]:
    """Pre-stage feature tables as device tensors so per-batch indexing has no
    host->device copy in the hot path."""
    return {
        "user_dense": torch.from_numpy(spec.user_dense).to(device),
        "item_dense": torch.from_numpy(spec.item_dense).to(device),
        "item_store_idx": torch.from_numpy(spec.item_store_idx).to(device),
        "item_main_cat_idx": torch.from_numpy(spec.item_main_cat_idx).to(device),
        "item_deeper_cat_idx": torch.from_numpy(spec.item_deeper_cat_idx).to(device),
    }


def train_one_epoch(
    model: TwoTower,
    dataset: TwoTowerPairDataset,
    spec_t: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    bce: nn.BCEWithLogitsLoss,
    device: str,
    grad_clip: float,
    batch_size: int,
    rng: torch.Generator,
) -> Dict[str, float]:
    """Manual batched indexing into the dataset's flat tensors. Skips
    `DataLoader.__getitem__` -- per-row scalar fetch + collation cost ~ms per
    row, which on 700K-row VG epochs adds up to half an hour even though the
    actual forward/backward is sub-ms.
    """
    model.train()
    # Pull the four flat tensors directly off the dataset and stage them on
    # device (small ints/floats; total ~few hundred MB even for Books).
    user_all = dataset._user_t.to(device)
    item_all = dataset._item_t.to(device)
    label_all = dataset._label_t.to(device)
    weight_all = dataset._weight_t.to(device)

    n_total = user_all.numel()
    perm = torch.randperm(n_total, generator=rng, device="cpu").to(device)
    n = n_total

    n_examples = 0
    sum_loss = 0.0
    sum_pos = 0.0
    sum_neg = 0.0
    n_pos = 0
    n_neg = 0

    n_batches_total = (n + batch_size - 1) // batch_size

    for batch_i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        idx = perm[start:end]
        u_idx = user_all.index_select(0, idx)
        i_idx = item_all.index_select(0, idx)
        label = label_all.index_select(0, idx)
        weight = weight_all.index_select(0, idx)

        u_dense = spec_t["user_dense"].index_select(0, u_idx)
        i_dense = spec_t["item_dense"].index_select(0, i_idx)
        store_idx = spec_t["item_store_idx"].index_select(0, i_idx)
        maincat_idx = spec_t["item_main_cat_idx"].index_select(0, i_idx)
        deeper_idx = spec_t["item_deeper_cat_idx"].index_select(0, i_idx)

        logits = model(
            u_idx, u_dense,
            i_idx, store_idx, maincat_idx, deeper_idx, i_dense,
        )
        per_row = bce(logits, label) * weight
        loss = per_row.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = label.numel()
        n_examples += bs
        loss_val = float(loss.detach())
        sum_loss += loss_val * bs
        with torch.no_grad():
            pos_mask = label > 0.5
            neg_mask = ~pos_mask
            sum_pos += float(logits[pos_mask].sum().detach()) if pos_mask.any() else 0.0
            sum_neg += float(logits[neg_mask].sum().detach()) if neg_mask.any() else 0.0
            n_pos += int(pos_mask.sum().item())
            n_neg += int(neg_mask.sum().item())

        if (batch_i + 1) % 100 == 0 or batch_i == 0 or batch_i + 1 == n_batches_total:
            print(f"    batch {batch_i+1}/{n_batches_total}  loss={loss_val:.4f}  "
                  f"running_mean={sum_loss/max(1,n_examples):.4f}", flush=True)

    return {
        "loss": sum_loss / max(1, n_examples),
        "mean_logit_pos": sum_pos / max(1, n_pos),
        "mean_logit_neg": sum_neg / max(1, n_neg),
        "n_examples": int(n_examples),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ---- Inference ---------------------------------------------------------------

@torch.no_grad()
def encode_all_items(model: TwoTower, spec: FeatureSpec, spec_t: Dict[str, torch.Tensor],
                     batch_size: int = 8192, device: str = "cpu") -> torch.Tensor:
    """Compute encoded vectors for every item idx in [1, n_items).

    Iterates idx order (1, 2, ..., n_items-1) so the returned [n_items-1, dim]
    tensor's row j is the encoding of item idx (j+1). This matches the
    `candidate_pa` array `train_two_tower.run_category` builds (also in
    idx-sorted order), which lets the top-K mapping be a plain row->parent_asin
    lookup. Encoding in pool-row order would silently mis-align top-K.
    """
    model.eval()
    out = []
    for start in range(1, spec.n_items, batch_size):
        end = min(start + batch_size, spec.n_items)
        idx = torch.arange(start, end, dtype=torch.long, device=device)
        store_idx = spec_t["item_store_idx"][idx]
        main_cat_idx = spec_t["item_main_cat_idx"][idx]
        deeper_idx = spec_t["item_deeper_cat_idx"][idx]
        i_dense = spec_t["item_dense"][idx]
        out.append(model.encode_items(idx, store_idx, main_cat_idx, deeper_idx, i_dense).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_users_subset(
    model: TwoTower,
    user_idxs: torch.Tensor,
    spec_t: Dict[str, torch.Tensor],
    batch_size: int = 8192,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode a list of user_idxs into [n, embedding_dim] CPU tensor."""
    model.eval()
    out = []
    for start in range(0, len(user_idxs), batch_size):
        idx = user_idxs[start:start + batch_size].to(device)
        u_dense = spec_t["user_dense"][idx]
        out.append(model.encode_users(idx, u_dense).cpu())
    return torch.cat(out, dim=0)


def topk_per_user_chunked(
    user_vecs: torch.Tensor,                       # [U, d] CPU, row r = encoding of user_idx_in_order[r]
    item_vecs: torch.Tensor,                       # [I, d] CPU, row j = encoding of item idx (j+1)
    candidate_item_ids: np.ndarray,                # parent_asin str, length I, candidate_item_ids[j] = item idx (j+1)
    user_ids_in_order: List[str],
    user_idx_in_order: np.ndarray,                 # int64 array aligned with user_vecs rows
    user_seen_per_user_idx: Dict[int, Set[int]],   # user_idx -> set of item idxs seen in train
    k: int,
    chunk_size: int = 1024,
    return_scores: bool = False,
):
    """Score U×I in chunks of users; mask train-seen items per user; take top-K.

    Item idxs in `user_seen_per_user_idx` are 1..n_items-1 (the FeatureSpec
    convention with PAD_IDX==0). Column j of `item_vecs`/`scores` corresponds
    to idx (j+1), so seen-mask cols are simply `[i - 1 for i in seen]` —
    a constant-time vectorized operation per user, NOT the O(n_items) string
    membership scan an earlier draft did.

    Returns a dict user_id -> ranked list of parent_asin (length k).
    """
    n_items_minus_pad = item_vecs.shape[0]   # = spec.n_items - 1
    out: Dict[str, List[str]] = {}
    out_scores: Dict[str, List[float]] = {} if return_scores else None
    for start in range(0, len(user_ids_in_order), chunk_size):
        end = min(start + chunk_size, len(user_ids_in_order))
        uv = user_vecs[start:end]
        scores = uv @ item_vecs.T            # [chunk, n_items - 1]
        for r in range(end - start):
            uidx = int(user_idx_in_order[start + r])
            seen = user_seen_per_user_idx.get(uidx)
            if not seen:
                continue
            mask_cols = [i - 1 for i in seen if 1 <= i <= n_items_minus_pad]
            if mask_cols:
                scores[r, mask_cols] = -float("inf")
        topk = torch.topk(scores, k=min(k, n_items_minus_pad), dim=1)
        idx_rows = topk.indices.tolist()
        if return_scores:
            val_rows = topk.values.tolist()
        for r, uid in enumerate(user_ids_in_order[start:end]):
            out[uid] = [candidate_item_ids[j] for j in idx_rows[r]]
            if return_scores:
                out_scores[uid] = [float(v) for v in val_rows[r]]
    if return_scores:
        return out, out_scores
    return out


# ---- End-to-end run ----------------------------------------------------------

def run_category(
    category: str,
    variant: str,
    cfg: TwoTowerConfig,
    train_cfg: TrainConfig,
    pair_cfg: PairSamplingConfig,
    seed: int = 42,
    smoke: bool = False,
) -> Dict:
    """Train + eval two-tower on one category. Writes the report JSON.

    `variant` is just a label for the JSON / filename; the actual feature
    toggles live in `cfg` (use_user_id_emb, use_item_id_emb, ...).
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cat_dir = PROCESSED_DIR / category
    train_df = pd.read_parquet(cat_dir / "train.parquet")
    val_df = pd.read_parquet(cat_dir / "val.parquet")
    test_df = pd.read_parquet(cat_dir / "test.parquet")
    item_features = pd.read_parquet(cat_dir / "item_features.parquet")
    user_features = pd.read_parquet(cat_dir / "user_features.parquet")

    preflight = metadata_preflight(category, item_features)

    # Raw metadata for deeper_category. Only [parent_asin, categories] needed.
    raw_meta_path = RAW_DIR / category / "metadata.parquet"
    raw_meta = None
    if raw_meta_path.exists():
        raw_meta = pd.read_parquet(raw_meta_path, columns=["parent_asin", "categories"])

    spec = build_feature_spec(
        train_df=train_df,
        user_features=user_features,
        item_features=item_features,
        raw_metadata=raw_meta,
        use_deeper_cat=cfg.use_deeper_cat_emb,
    )
    print(f"[two-tower] {category}/{variant}: vocab "
          f"users={spec.n_users} items={spec.n_items} stores={spec.n_stores} "
          f"main_cats={spec.n_main_cats} deeper_cats={spec.n_deeper_cats} "
          f"(has_deeper={spec.has_deeper_cat})", flush=True)

    # Smoke override: tiny everything. The aim is "completes in <60 s and all
    # the moving pieces ran end-to-end", not "produces a meaningful model".
    if smoke:
        cfg.embedding_dim = min(cfg.embedding_dim, 16)
        cfg.hidden_dim = min(cfg.hidden_dim, 32)
        cfg.id_emb_dim = min(cfg.id_emb_dim, 16)
        cfg.cat_emb_dim = min(cfg.cat_emb_dim, 8)
        train_cfg.epochs = 1
        train_cfg.batch_size = 1024
        train_cfg.max_train_pairs = 50_000
        train_cfg.max_eval_users = 5_000
        pair_cfg.n_soft_neg = min(pair_cfg.n_soft_neg, 2)

    dataset = TwoTowerPairDataset(train_df, spec, pair_cfg)
    print(f"[two-tower] {category}/{variant}: pairs initial composition = "
          f"{dataset.composition()}", flush=True)

    # Apply --smoke pair cap by truncating the dataset's flat tensors. We trim
    # AFTER resample_soft_negatives ran in __init__, so the cap respects the
    # real positive : hard-neg : soft-neg ratio (positives are first, hard
    # negs second, soft negs last in the flat layout); a head-cap mostly
    # keeps positives + hard negs and slices into the soft-neg tail.
    if train_cfg.max_train_pairs and len(dataset) > train_cfg.max_train_pairs:
        cap = train_cfg.max_train_pairs
        # Stratified head: keep all positives (cheap), then fill with shuffled
        # mixture of hard + soft negs. We reuse the dataset internals because
        # rewriting the flat tensors after a permutation is cleanest.
        composition_before = dataset.composition()
        n_pos = composition_before["n_positive"]
        n_hard = composition_before["n_hard_neg"]
        n_soft = composition_before["n_soft_neg"]
        keep_pos = min(n_pos, max(1, cap // 3))
        remaining = cap - keep_pos
        # Maintain the (1 positive : n_soft soft-neg) shape by capping soft-neg
        # to keep_pos * n_soft, then any remainder fills with hard negs.
        keep_soft = min(n_soft, keep_pos * pair_cfg.n_soft_neg)
        keep_hard = min(n_hard, max(0, remaining - keep_soft))
        rng_np = np.random.default_rng(seed)
        pos_pick = rng_np.choice(n_pos, size=keep_pos, replace=False)
        hard_pick = (rng_np.choice(n_hard, size=keep_hard, replace=False)
                     if keep_hard > 0 else np.empty(0, dtype=np.int64))
        soft_pick = (rng_np.choice(n_soft, size=keep_soft, replace=False)
                     if keep_soft > 0 else np.empty(0, dtype=np.int64))
        # Rebuild flat tensors directly off the dataset's internals.
        u = np.concatenate([
            dataset._pos_uidx[pos_pick],
            dataset._hard_uidx[hard_pick],
            dataset._soft_uidx[soft_pick],
        ])
        i = np.concatenate([
            dataset._pos_iidx[pos_pick],
            dataset._hard_iidx[hard_pick],
            dataset._soft_iidx[soft_pick],
        ])
        label = np.concatenate([
            np.ones(keep_pos, dtype=np.float32),
            np.zeros(keep_hard, dtype=np.float32),
            np.zeros(keep_soft, dtype=np.float32),
        ])
        weight = np.concatenate([
            np.full(keep_pos, pair_cfg.positive_weight, dtype=np.float32),
            np.full(keep_hard, pair_cfg.hard_negative_weight, dtype=np.float32),
            np.full(keep_soft, pair_cfg.soft_negative_weight, dtype=np.float32),
        ])
        dataset._user_t = torch.from_numpy(u.astype(np.int64))
        dataset._item_t = torch.from_numpy(i.astype(np.int64))
        dataset._label_t = torch.from_numpy(label)
        dataset._weight_t = torch.from_numpy(weight)
        # Update internal pos/hard/soft idx arrays so future resamples
        # (no-op in smoke since epochs=1) stay consistent.
        dataset._pos_uidx = dataset._pos_uidx[pos_pick]
        dataset._pos_iidx = dataset._pos_iidx[pos_pick]
        dataset._hard_uidx = dataset._hard_uidx[hard_pick]
        dataset._hard_iidx = dataset._hard_iidx[hard_pick]
        dataset._soft_uidx = dataset._soft_uidx[soft_pick]
        dataset._soft_iidx = dataset._soft_iidx[soft_pick]
        print(f"[two-tower] {category}/{variant}: capped pairs to "
              f"{len(dataset):,} (was {sum(composition_before.values()):,}); "
              f"new composition = {dataset.composition()}", flush=True)

    user_tower = UserTower(spec.n_users, spec.n_user_dense, cfg)
    item_tower = ItemTower(
        n_items=spec.n_items,
        n_stores=spec.n_stores,
        n_main_cats=spec.n_main_cats,
        n_deeper_cats=spec.n_deeper_cats,
        n_dense=spec.n_item_dense,
        cfg=cfg,
        has_deeper_cat=spec.has_deeper_cat,
    )
    model = TwoTower(user_tower, item_tower).to(train_cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    train_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(seed)

    spec_t = _move_spec_to_tensors(spec, train_cfg.device)

    # ---- Preprare eval prerequisites once ------------------------------------
    val_gt = build_groundtruth(val_df)
    test_gt = build_groundtruth(test_df)
    pool_set = {pa for pa, idx in spec.item_id_to_idx.items() if idx != 0}
    eval_users = sorted(set(val_df["user_id"].astype(str)) | set(test_df["user_id"].astype(str)))
    if train_cfg.max_eval_users and len(eval_users) > train_cfg.max_eval_users:
        rng_eval = np.random.default_rng(seed + 1)
        # Bias toward users with at least one positive in val OR test, so the
        # smoke recall denominator isn't all zeros. Falls back to a uniform
        # sample if we don't have enough positive users.
        pos_users = sorted(set(val_gt) | set(test_gt))
        cap = train_cfg.max_eval_users
        keep_pos = min(len(pos_users), int(cap * 0.7))
        keep_extra = cap - keep_pos
        chosen_pos = list(rng_eval.choice(pos_users, size=keep_pos, replace=False)) if keep_pos else []
        chosen_pos_set = set(chosen_pos)
        rest = [u for u in eval_users if u not in chosen_pos_set]
        chosen_extra = list(rng_eval.choice(rest, size=min(keep_extra, len(rest)), replace=False)) if rest else []
        eval_users = sorted(set(chosen_pos) | set(chosen_extra))
        print(f"[two-tower] {category}/{variant}: capped eval users to "
              f"{len(eval_users)} (smoke)", flush=True)

    # Map eval users to FeatureSpec idx; cold users (not in train -- shouldn't
    # occur given Phase 0 leave-last-two with min_user>=3) fall to PAD_IDX.
    eval_user_idx_np = np.array(
        [spec.user_id_to_idx.get(u, 0) for u in eval_users], dtype=np.int64,
    )
    eval_user_idx = torch.from_numpy(eval_user_idx_np)
    candidate_pa = np.array(
        [pa for pa, _ in sorted(spec.item_id_to_idx.items(), key=lambda kv: kv[1])
         if pa != "<PAD>"], dtype=object,
    )

    # ---- Training loop --------------------------------------------------------
    history: List[Dict] = []
    best_val_recall = -1.0
    best_state = None
    patience_left = train_cfg.early_stopping_patience
    for epoch in range(1, train_cfg.epochs + 1):
        ep_t0 = time.time()
        train_metrics = train_one_epoch(
            model, dataset, spec_t, optimizer, bce,
            train_cfg.device, train_cfg.grad_clip,
            train_cfg.batch_size, train_rng,
        )
        ep_elapsed = time.time() - ep_t0

        eval_metrics: Dict[str, float] = {}
        if train_cfg.eval_every_epoch:
            item_vecs = encode_all_items(model, spec, spec_t, device=train_cfg.device)
            user_vecs = encode_users_subset(
                model, eval_user_idx, spec_t, device=train_cfg.device,
            )
            topk_val = topk_per_user_chunked(
                user_vecs, item_vecs, candidate_pa,
                eval_users, eval_user_idx_np,
                spec.user_seen_per_user_idx,
                k=DEFAULT_K_RETRIEVE,
            )
            val_rep = build_split_report(
                "val", topk_val, val_gt, pool_set, "in_train_catalog", DEFAULT_KS,
            )
            eval_metrics = dict(val_rep.metrics)
            eval_metrics["heldout_positive_coverage_by_candidate_pool"] = (
                val_rep.heldout_positive_coverage_by_candidate_pool
            )

        history.append({
            "epoch": epoch,
            "epoch_seconds": round(ep_elapsed, 2),
            "train": train_metrics,
            "val_metrics": eval_metrics,
        })
        print(
            f"[two-tower] {category}/{variant} ep{epoch:02d}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"logit_pos={train_metrics['mean_logit_pos']:.3f}  "
            f"logit_neg={train_metrics['mean_logit_neg']:.3f}  "
            f"val_R@100={eval_metrics.get('Recall@100', float('nan')):.4f}",
            flush=True,
        )

        # Early stopping on val Recall@100.
        cur = eval_metrics.get("Recall@100", -1.0)
        if cur > best_val_recall:
            best_val_recall = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = train_cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[two-tower] {category}/{variant}: early stop at ep{epoch}",
                      flush=True)
                break

        # Resample soft negs each epoch so the model doesn't overfit a fixed set.
        dataset.resample_soft_negatives(epoch=epoch)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final eval (val + test) on best checkpoint --------------------------
    item_vecs = encode_all_items(model, spec, spec_t, device=train_cfg.device)
    user_vecs = encode_users_subset(
        model, eval_user_idx, spec_t, device=train_cfg.device,
    )
    topk_full, topk_scores = topk_per_user_chunked(
        user_vecs, item_vecs, candidate_pa,
        eval_users, eval_user_idx_np,
        spec.user_seen_per_user_idx,
        k=DEFAULT_K_RETRIEVE,
        return_scores=True,
    )
    val_rep = build_split_report(
        "val", topk_full, val_gt, pool_set, "in_train_catalog", DEFAULT_KS,
    )
    test_rep = build_split_report(
        "test", topk_full, test_gt, pool_set, "in_train_catalog", DEFAULT_KS,
    )

    # Save per-user top-K (items + scores) so Phase 2 candidate_builder can
    # consume the two-tower retrieval source without re-running the model.
    # Parquet rather than JSON: ~78K users x 100 items = 7.8M rows for VG;
    # JSON would be hundreds of MB, parquet is tens.
    pred_rows = []
    for uid in eval_users:
        items_u = topk_full[uid]
        scores_u = topk_scores[uid]
        for r, (pa, sc) in enumerate(zip(items_u, scores_u)):
            pred_rows.append((uid, pa, "two_tower", float(sc), r + 1, category, variant))
    pred_df = pd.DataFrame(
        pred_rows,
        columns=["user_id", "parent_asin", "source", "score", "rank",
                 "category", "model_variant"],
    )
    predictions_path = RESULTS_DIR / f"{category}_two_tower_{variant}_predictions.parquet"
    pred_df.to_parquet(predictions_path, index=False)
    print(f"[two-tower] saved per-user predictions -> {predictions_path} "
          f"({len(pred_df):,} rows)", flush=True)

    report = {
        "category": category,
        "variant": variant,
        "started_utc": _now_iso(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "preflight": preflight,
        "config": {
            "two_tower": asdict(cfg),
            "training": asdict(train_cfg),
            "pair_sampling": asdict(pair_cfg),
            "seed": seed,
        },
        "feature_spec_summary": {
            "n_users": spec.n_users,
            "n_items": spec.n_items,
            "n_stores": spec.n_stores,
            "n_main_cats": spec.n_main_cats,
            "n_deeper_cats": spec.n_deeper_cats,
            "has_deeper_cat": spec.has_deeper_cat,
            "user_dense_cols": list(spec.user_dense_cols),
            "item_dense_cols": list(spec.item_dense_cols),
            "user_dense_norm_stats": spec.user_dense_norm_stats,
            "item_dense_norm_stats": spec.item_dense_norm_stats,
        },
        "dataset_composition": dataset.composition(),
        "candidate_pool": {
            "type": "in_train_catalog",
            "size": len(pool_set),
        },
        "history": history,
        "best_val_recall@100": best_val_recall,
        "splits": {
            "val": val_rep.to_dict(),
            "test": test_rep.to_dict(),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{category}_two_tower_{variant}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[two-tower] wrote {out} (elapsed {report['elapsed_seconds']}s)", flush=True)
    return report


# ---- CLI ---------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True,
                   choices=["All_Beauty", "Video_Games", "Books", "Electronics"])
    p.add_argument("--variant", default="metadata_and_ids",
                   choices=["metadata_only", "metadata_and_ids"],
                   help=("metadata_only: ablate user_id + parent_asin embeddings off. "
                         "metadata_and_ids: enable both id embeddings (v1 default)."))
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--id-emb-dim", type=int, default=32)
    p.add_argument("--cat-emb-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-soft-negatives", type=int, default=4)
    p.add_argument("--use-hard-negatives", action="store_true", default=False)
    p.add_argument("--positive-weight", type=float, default=1.0)
    p.add_argument("--hard-negative-weight", type=float, default=1.0)
    p.add_argument("--soft-negative-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: 1 epoch, embedding_dim<=16, num_soft_negatives=2.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    use_user_id = args.variant == "metadata_and_ids"
    use_item_id = args.variant == "metadata_and_ids"

    cfg = TwoTowerConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        id_emb_dim=args.id_emb_dim,
        cat_emb_dim=args.cat_emb_dim,
        dropout=args.dropout,
        use_user_id_emb=use_user_id,
        use_item_id_emb=use_item_id,
        use_deeper_cat_emb=True,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    pair_cfg = PairSamplingConfig(
        n_soft_neg=args.num_soft_negatives,
        use_hard_negatives=args.use_hard_negatives,
        positive_weight=args.positive_weight,
        hard_negative_weight=args.hard_negative_weight,
        soft_negative_weight=args.soft_negative_weight,
        seed=args.seed,
    )
    run_category(
        category=args.category,
        variant=args.variant,
        cfg=cfg,
        train_cfg=train_cfg,
        pair_cfg=pair_cfg,
        seed=args.seed,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
