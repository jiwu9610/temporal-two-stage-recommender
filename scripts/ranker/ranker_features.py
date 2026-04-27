"""Shared feature prep + train/eval helpers for Phase 2 rankers.

Both the MLP ranker and the Deep+Cross ranker consume
`data/processed/{category}/candidates.parquet` and must build identical
feature tensors so a fair model A/B is possible. Anything model-specific
(architecture, hidden dims) lives in mlp_ranker.py / complex_ranker.py;
feature prep, vocab building, normalization, the training loop scaffold,
and the post-train rerank evaluator live here.

Train / val / test contract (spec, fixes the earlier "val == train"
bug that would have over-stated Recall):

  - candidates.parquet's `split == "val"` rows are split USER-WISE 80/20:
        80% val users -> ranker_train  (loss + grad)
        20% val users -> ranker_val    (early stop + best-ckpt selection)
  - candidates.parquet's `split == "test"` rows are TEST ONLY -- never used
    for normalization, vocab, training, or early stopping.
  - Vocabularies and dense-feature normalization stats are built from
    ranker_train rows only.
  - Class imbalance is handled with `pos_weight = n_neg / n_pos` in
    BCEWithLogitsLoss (computed on ranker_train labels).

Reporting / coverage:

  - `candidate_pool_type` in ranker reports = "candidate_union_top{K}"
    (NOT in_train_catalog), because the ranker's reachable set per user is
    the union of that user's retrieval candidates, not the global catalog.
  - We report `heldout_positive_in_candidate_union_rate`: the fraction of
    eval users whose held-out positive item appears in their candidate rows
    -- the true upper-bound of Recall the ranker can achieve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.retrieval.evaluator import build_split_report


# Numeric features for the dense vector. Order is fixed so checkpoints can be
# replayed against a freshly-built candidates table.
DENSE_FEATURES = (
    # retrieval scores + ranks + flags
    "popularity_score",
    "rule_score",
    "two_tower_score",
    "popularity_rank",
    "rule_rank",
    "two_tower_rank",
    "best_rank",
    "num_sources",
    "source_popularity",
    "source_rule",
    "source_two_tower",
    # user dense
    "n_reviews_train",
    "avg_rating_train",
    "std_rating_train",
    "n_unique_items_train",
    "active_days_train",
    "verified_rate_train",
    # item dense
    "price",
    "average_rating",
    "rating_number",
    "n_features",
    "n_description",
    "n_categories",
    # cross
    "user_store_affinity",
    "user_category_affinity",
    "same_top_store",
    "same_top_category",
)

LOG1P_FEATURES = {
    "popularity_score", "rule_score",
    "popularity_rank", "rule_rank", "two_tower_rank", "best_rank",
    "n_reviews_train", "n_unique_items_train", "active_days_train",
    "price", "rating_number",
    "n_features", "n_description", "n_categories",
}

CATEGORICAL_FEATURES = ("main_category", "store")


@dataclass
class RankerFeatureSpec:
    n_dense: int
    dense_mean: np.ndarray
    dense_std: np.ndarray
    cat_vocabs: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def n_cat(self) -> int:
        return len(self.cat_vocabs)

    def cat_vocab_size(self, name: str) -> int:
        return len(self.cat_vocabs[name])


def _build_cat_vocab(values: pd.Series, pad: str = "<PAD>") -> Dict[str, int]:
    out: Dict[str, int] = {pad: 0}
    for v in sorted(values.dropna().astype(str).unique()):
        if v == pad:
            continue
        out[v] = len(out)
    return out


def split_val_users_for_ranker(
    candidates_df: pd.DataFrame,
    ranker_val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Deterministic user-wise split of candidates.parquet's val rows into
    ranker_train_users (80%) and ranker_val_users (20%). Returns two sets of
    user_id strings so callers can mask candidates by user.
    """
    val_users = (
        candidates_df.loc[candidates_df["split"] == "val", "user_id"]
        .astype(str).unique()
    )
    val_users = np.array(sorted(val_users))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(val_users))
    val_users = val_users[perm]
    n_val = max(1, int(len(val_users) * ranker_val_frac))
    ranker_val_users = set(val_users[:n_val].tolist())
    ranker_train_users = set(val_users[n_val:].tolist())
    return ranker_train_users, ranker_val_users


def build_feature_spec(ranker_train_df: pd.DataFrame) -> RankerFeatureSpec:
    """Compute dense normalization stats + categorical vocabs from
    ranker_train rows only. Test rows MUST NOT be in this DataFrame."""
    dense = np.zeros((len(ranker_train_df), len(DENSE_FEATURES)), dtype=np.float64)
    for j, col in enumerate(DENSE_FEATURES):
        x = pd.to_numeric(ranker_train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if col in LOG1P_FEATURES:
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.maximum(x, 0.0)
            x = np.log1p(x)
        dense[:, j] = x

    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)

    cat_vocabs = {
        c: _build_cat_vocab(ranker_train_df[c]) for c in CATEGORICAL_FEATURES
    }
    return RankerFeatureSpec(
        n_dense=len(DENSE_FEATURES),
        dense_mean=mean,
        dense_std=std,
        cat_vocabs=cat_vocabs,
    )


def build_tensors(
    df: pd.DataFrame,
    spec: RankerFeatureSpec,
) -> Dict[str, torch.Tensor]:
    n = len(df)
    dense = np.zeros((n, len(DENSE_FEATURES)), dtype=np.float32)
    for j, col in enumerate(DENSE_FEATURES):
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if col in LOG1P_FEATURES:
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.maximum(x, 0.0)
            x = np.log1p(x)
        x = (x - spec.dense_mean[j]) / spec.dense_std[j]
        dense[:, j] = x.astype(np.float32)

    out = {
        "dense": torch.from_numpy(dense),
        "label": torch.from_numpy(df["label"].to_numpy(dtype=np.float32)),
    }
    for c, vocab in spec.cat_vocabs.items():
        s = df[c].astype(str).map(vocab).fillna(0).to_numpy(dtype=np.int64)
        out[f"cat__{c}"] = torch.from_numpy(s)
    return out


def heldout_positive_coverage_in_candidate_union(
    df: pd.DataFrame,
    groundtruth: Dict[str, Set[str]],
) -> float:
    """Per-user fraction: does the user's held-out positive appear in their
    candidate rows? Average over users with |gt[u]| > 0. This is the true
    Recall upper-bound for the ranker (a user whose positive isn't in their
    candidate union cannot be reranked into Recall@K)."""
    cand_per_user: Dict[str, Set[str]] = {}
    for u, g in df.groupby("user_id"):
        cand_per_user[u] = set(g["parent_asin"].astype(str))
    rates = []
    for u, gt in groundtruth.items():
        if not gt:
            continue
        cand = cand_per_user.get(u, set())
        rates.append(len(gt & cand) / len(gt))
    return float(np.mean(rates)) if rates else 0.0


# ---------------------------------------------------------------------------
# Train loop + rerank eval
# ---------------------------------------------------------------------------

def train_pointwise_bce(
    model: nn.Module,
    train_inputs: Dict[str, torch.Tensor],
    val_inputs: Dict[str, torch.Tensor],
    val_user_ids: np.ndarray,
    val_pa: np.ndarray,
    val_groundtruth: Dict[str, Set[str]],
    candidate_pool: Set[str],
    *,
    epochs: int = 20,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 5.0,
    early_stopping_patience: int = 3,
    device: str = "cpu",
    log_every_n_batches: int = 100,
    seed: int = 42,
    pos_weight: Optional[float] = None,
) -> Dict:
    """Pointwise BCE-with-logits training loop shared by both rankers.

    `val_inputs` should be the held-out ranker_val tensors (NOT the same as
    `train_inputs`); the caller is responsible for splitting val users into
    ranker_train (used for `train_inputs`) and ranker_val (used here).

    `pos_weight` (optional): scalar applied to the positive class in BCE to
    counter the heavy negative imbalance at candidate scale. If None, use 1.0.

    Returns history + best metrics. Model is mutated in place; the best
    state by ranker_val Recall@100 is reloaded at the end.
    """
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    device_train = {k: v.to(device) for k, v in train_inputs.items()}
    device_val = {k: v.to(device) for k, v in val_inputs.items()}
    label_train = device_train["label"]
    label_val = device_val["label"]
    n_train = label_train.numel()
    n_val = label_val.numel()

    pw = torch.tensor(float(pos_weight), device=device) if pos_weight else None
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)
    bce_eval = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _model_inputs(inputs: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v.index_select(0, idx) for k, v in inputs.items() if k != "label"}

    history = []
    best_recall = -1.0
    best_state = None
    patience_left = early_stopping_patience
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=rng).to(device)
        sum_loss = 0.0
        seen = 0
        for bi, start in enumerate(range(0, n_train, batch_size)):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            inputs = _model_inputs(device_train, idx)
            labels = label_train.index_select(0, idx)
            logits = model(**inputs)
            loss = bce(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = labels.numel()
            seen += bs
            sum_loss += float(loss.detach()) * bs
            if (bi + 1) % log_every_n_batches == 0 or bi == 0 or bi + 1 == n_batches:
                print(f"    batch {bi+1}/{n_batches}  loss={float(loss.detach()):.4f}  "
                      f"running_mean={sum_loss/max(1,seen):.4f}", flush=True)

        # Eval pass on ranker_val.
        model.eval()
        with torch.no_grad():
            val_logits_chunks = []
            val_no_label = {k: v for k, v in device_val.items() if k != "label"}
            chunk = max(batch_size * 4, 16384)
            for s in range(0, n_val, chunk):
                e = min(s + chunk, n_val)
                idx = torch.arange(s, e, device=device)
                val_logits_chunks.append(model(**_model_inputs(val_no_label, idx)).cpu())
            val_logits = torch.cat(val_logits_chunks).numpy()
            val_loss = float(bce_eval(
                torch.from_numpy(val_logits).to(device), label_val,
            ).detach())

        topk = _scores_to_topk(val_logits, val_user_ids, val_pa, k=100)
        rep = build_split_report(
            "ranker_val", topk, val_groundtruth, candidate_pool,
            "candidate_union_top100", ks=(10, 50, 100),
        )
        recall_100 = rep.metrics.get("Recall@100", -1.0)

        history.append({
            "epoch": epoch,
            "train_loss": sum_loss / max(1, seen),
            "ranker_val_loss": val_loss,
            "ranker_val_metrics": rep.metrics,
        })
        print(f"  ep{epoch:02d}  train_loss={sum_loss/max(1,seen):.4f}  "
              f"ranker_val_loss={val_loss:.4f}  R@10={rep.metrics['Recall@10']:.4f}  "
              f"R@50={rep.metrics['Recall@50']:.4f}  R@100={recall_100:.4f}",
              flush=True)

        if recall_100 > best_recall:
            best_recall = recall_100
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at ep{epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_ranker_val_recall@100": best_recall}


def _scores_to_topk(
    logits: np.ndarray,
    user_ids: np.ndarray,
    parent_asins: np.ndarray,
    k: int,
) -> Dict[str, List[str]]:
    """Group logits by user_id, return ranked top-K parent_asin per user."""
    out: Dict[str, List[Tuple[str, float]]] = {}
    for u, pa, sc in zip(user_ids, parent_asins, logits):
        out.setdefault(u, []).append((pa, float(sc)))
    topk: Dict[str, List[str]] = {}
    for u, pairs in out.items():
        pairs.sort(key=lambda t: -t[1])
        topk[u] = [pa for pa, _ in pairs[:k]]
    return topk


def evaluate_split(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    user_ids: np.ndarray,
    parent_asins: np.ndarray,
    groundtruth: Dict[str, Set[str]],
    candidate_pool: Set[str],
    split_name: str,
    candidate_pool_type: str = "candidate_union_top100",
    device: str = "cpu",
    batch_size: int = 65536,
):
    """Run the trained model over a split, build per-user top-K, return SplitReport."""
    model.eval()
    with torch.no_grad():
        device_in = {k: v.to(device) for k, v in inputs.items() if k != "label"}
        # `.shape[0]` not `.numel()`: when the first iter value is `dense`
        # (2-D, [N, n_dense]), .numel() returns N*n_dense and the loop walks
        # past N -> embedding OOB on GPU. shape[0] is row count for any rank.
        sizes = {k: v.shape[0] for k, v in device_in.items()}
        if sizes:
            sz = next(iter(sizes.values()))
            mismatched = {k: s for k, s in sizes.items() if s != sz}
            assert not mismatched, (
                f"evaluate_split tensor row counts disagree: "
                f"{ {k: sizes[k] for k in sizes} }"
            )
            n = sz
        else:
            n = 0
        logits_chunks = []
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            idx = torch.arange(s, e, device=device)
            sub = {k: v.index_select(0, idx) for k, v in device_in.items()}
            logits_chunks.append(model(**sub).cpu())
        logits = torch.cat(logits_chunks).numpy() if logits_chunks else np.empty(0)
    topk = _scores_to_topk(logits, user_ids, parent_asins, k=100)
    return build_split_report(
        split_name, topk, groundtruth, candidate_pool, candidate_pool_type,
        ks=(10, 50, 100),
    ), topk
