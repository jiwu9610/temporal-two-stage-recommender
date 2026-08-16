"""Clean temporal ranker: time-separated training, model selection, refit, test.

CLI:
    python -m scripts.ranker.train_temporal_ranker --category All_Beauty [--smoke]

Protocol (walk-forward, no within-snapshot user split):

    TRAIN       candidates_ranker_train.parquet    labels from [T0, T1)
    SELECTION   candidates_model_selection.parquet labels from [T1, T2)
                -- early stopping, architecture comparison (MLP vs Deep&Cross),
                   learning-rate selection, epoch selection happen HERE ONLY.
    REFIT       chosen config retrained on ranker_train + model_selection rows
                for the exact epoch count chosen during selection.
                NO early stopping, NO metric peeking at test.
    FINAL TEST  the fixed refit model applied once to candidates_test.parquet,
                scored against groundtruth_test ([T2, T3) labels).

Test labels influence nothing upstream: not fitting, not early stopping, not
architecture/lr/epoch choice, not normalization or vocabularies (refit stats
come from train+selection rows only).

Reported on final test: overall Recall/Precision@10/50/100 over ALL ground
truth users (missed users stay in the denominator), conditional Recall@K
given the target was retrieved, the candidate ceiling (retrieved coverage),
0/1/2/3+ history cohorts, and warm/cold target-item cohorts.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.retrieval.evaluator import build_split_report, recall_precision_at_k
from scripts.ranker.mlp_ranker import MLPRanker, MLPRankerConfig
from scripts.ranker.complex_ranker import DeepCrossConfig, DeepCrossRanker
from scripts.ranker.ranker_features import (
    RankerFeatureSpec,
    _build_cat_vocab,
    _scores_to_topk,
    train_pointwise_bce,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "phase2_temporal"
KS = (10, 50, 100)

# Dense features available in candidates_{snapshot}.parquet. Snapshot-scoped
# `_hist` aggregates + the three temporal features; NO price / average_rating /
# rating_number (dump-time whole-timeline statistics).
TEMPORAL_DENSE_FEATURES = (
    "popularity_score", "rule_score", "two_tower_score",
    "popularity_rank", "rule_rank", "two_tower_rank",
    "best_rank", "num_sources",
    "source_popularity", "source_rule", "source_two_tower",
    "user_n_reviews_hist", "user_avg_rating_hist", "user_std_rating_hist",
    "user_n_unique_items_hist", "user_active_days_hist",
    "user_verified_rate_hist",
    "days_since_last_interaction", "interactions_last_30d",
    "positives_last_30d", "n_history_events", "user_has_history",
    "item_n_reviews_hist", "item_n_positives_hist", "item_avg_rating_hist",
    "item_n_unique_reviewers_hist",
    "n_features", "n_description", "n_categories",
    "user_store_affinity", "user_category_affinity",
    "same_top_store", "same_top_category",
)
TEMPORAL_LOG1P = {
    "popularity_score", "rule_score",
    "popularity_rank", "rule_rank", "two_tower_rank", "best_rank",
    "user_n_reviews_hist", "user_n_unique_items_hist", "user_active_days_hist",
    "days_since_last_interaction", "interactions_last_30d", "positives_last_30d",
    "n_history_events",
    "item_n_reviews_hist", "item_n_positives_hist",
    "item_n_unique_reviewers_hist",
    "n_features", "n_description", "n_categories",
}
TEMPORAL_CATEGORICALS = ("main_category", "store")

# ---------------------------------------------------------------------------
# Wave 3 sequence arm (DIN at the ranking position). Default OFF: with
# `seq_arm=False` every code path below is byte-identical to the frozen
# protocol (no seq tensors built, grid unchanged, arch "din" unreachable).
# ---------------------------------------------------------------------------
SEQ_HIST_LEN = 20


class _SeqContext:
    """Wave-3 sequence feature (advisor spec, advanced_seq.pdf): per user the
    last-L POSITIVE engagements before the snapshot cutoff, each carrying
      item index      frozen vocab from the ranker_train HISTORY only
                      (later-snapshot items unseen there -> 0 = pad)
      cat / store idx sparse side ids (vocab from the ranker_train catalog)
      abs bucket      absolute timestamp, monthly bucket since a fixed
                      epoch (0 = pad)                      -> sin/cos or emb
      delta bucket    log-spaced gap to the NEXT engagement (the last one
                      measures to the cutoff), 0 = pad     -> emb
    All from history_{snap}.parquet, strictly ts < cutoff by construction --
    the advisor's "aggregate up to ds-1" leakage rule holds per snapshot."""

    ABS_EPOCH_MS = 946684800000          # 2000-01-01
    ABS_MONTH_MS = 30 * 86400 * 1000
    N_ABS = 400                          # ~33 years of months
    DELTA_EDGES_DAYS = (0, 1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1e9)

    def __init__(self, tdir: Path, snapshots, hist_len: int = SEQ_HIST_LEN,
                 positive_min_rating: float = 4.0):
        self.hist_len = hist_len
        h0 = pd.read_parquet(tdir / "history_ranker_train.parquet",
                             columns=["parent_asin"])
        items = sorted(h0["parent_asin"].astype(str).unique())
        self.item_to_idx = {a: i + 1 for i, a in enumerate(items)}   # 0 = pad
        self.n_items = len(items) + 1
        # sparse side ids from the ranker_train catalog (history-only rows)
        itf = pd.read_parquet(tdir / "item_features_ranker_train.parquet",
                              columns=["parent_asin", "main_category", "store"])
        itf["parent_asin"] = itf["parent_asin"].astype(str)
        self.side_vocab = {}
        self.item_side = {}
        for col in ("main_category", "store"):
            vals = sorted(v for v in itf[col].fillna("").astype(str).unique()
                          if v not in ("", "nan", "None"))
            self.side_vocab[col] = {v: i + 1 for i, v in enumerate(vals)}
            m = dict(zip(itf["parent_asin"],
                         itf[col].fillna("").astype(str).map(self.side_vocab[col]).fillna(0).astype(int)))
            self.item_side[col] = m
        self.n_side = {c: len(v) + 1 for c, v in self.side_vocab.items()}
        self.n_delta = len(self.DELTA_EDGES_DAYS)   # bucket ids 1..n-1, 0 pad
        manifest = json.loads((tdir / "snapshot_manifest.json").read_text())
        self.user_hist: Dict[str, Dict[str, np.ndarray]] = {}
        for snap in snapshots:
            cutoff = int(manifest["snapshots"][snap]["history_end_ms"])
            h = pd.read_parquet(tdir / f"history_{snap}.parquet",
                                columns=["user_id", "parent_asin", "timestamp",
                                         "rating"])
            h = h[h["rating"] >= positive_min_rating]
            h["user_id"] = h["user_id"].astype(str)
            h["parent_asin"] = h["parent_asin"].astype(str)
            assert int(h["timestamp"].max()) < cutoff, (snap, "history not < cutoff")
            h = h.sort_values(["user_id", "timestamp"], kind="mergesort")
            h = h.groupby("user_id", sort=False).tail(hist_len)
            h["idx"] = h["parent_asin"].map(self.item_to_idx).fillna(0).astype(np.int64)
            for col in ("main_category", "store"):
                h[col] = h["parent_asin"].map(self.item_side[col]).fillna(0).astype(np.int64)
            ts = h["timestamp"].to_numpy(np.int64)
            h["abs_b"] = np.clip((ts - self.ABS_EPOCH_MS) // self.ABS_MONTH_MS + 1,
                                 1, self.N_ABS - 1)
            per_user: Dict[str, np.ndarray] = {}
            edges_ms = np.asarray(self.DELTA_EDGES_DAYS[1:]) * 86400 * 1000
            for uid, grp in h.groupby("user_id", sort=False):
                t = grp["timestamp"].to_numpy(np.int64)
                nxt = np.append(t[1:], cutoff)                 # gap to next / cutoff
                gap = np.maximum(nxt - t, 0)
                delta_b = np.searchsorted(edges_ms, gap, side="right") + 1
                L = len(t); row = np.zeros((hist_len, 5), dtype=np.int64)
                row[hist_len - L:, 0] = grp["idx"].to_numpy()
                row[hist_len - L:, 1] = grp["main_category"].to_numpy()
                row[hist_len - L:, 2] = grp["store"].to_numpy()
                row[hist_len - L:, 3] = grp["abs_b"].to_numpy()
                row[hist_len - L:, 4] = delta_b
                per_user[uid] = row                             # right-aligned, oldest first
            self.user_hist[snap] = per_user
            print(f"[seq] {snap}: users_with_history={len(per_user):,} "
                  f"vocab={self.n_items:,} cutoff={cutoff}", flush=True)

    def tensors(self, df: pd.DataFrame, snap: str) -> Dict[str, torch.Tensor]:
        per_user = self.user_hist[snap]
        zero = np.zeros((self.hist_len, 5), dtype=np.int64)
        uids = df["user_id"].to_numpy()
        uniq, inv = np.unique(uids, return_inverse=True)
        mat = (np.stack([per_user.get(u, zero) for u in uniq]) if len(uniq)
               else np.zeros((0, self.hist_len, 5), dtype=np.int64))
        hist = mat[inv]                                          # [N, L, 5]
        pa = df["parent_asin"]
        cand = np.stack([
            pa.map(self.item_to_idx).fillna(0).to_numpy(np.int64),
            pa.map(self.item_side["main_category"]).fillna(0).to_numpy(np.int64),
            pa.map(self.item_side["store"]).fillna(0).to_numpy(np.int64),
        ], axis=1)                                               # [N, 3]
        return {"seq__hist": torch.from_numpy(hist),
                "seq__cand": torch.from_numpy(cand)}


_SEQ: Optional[_SeqContext] = None


def resolve_feature_list(df: pd.DataFrame):
    r"""Dense feature list for a candidates table: the frozen base tuple plus
    any Phase 3A extra-source columns (source_*/\*_rank/\*_score triplets),
    sorted for a stable order. Extra ranks/scores get log1p (non-negative,
    long-tailed)."""
    base = [f for f in TEMPORAL_DENSE_FEATURES if f in df.columns]
    known = set(TEMPORAL_DENSE_FEATURES) | {
        "snapshot", "user_id", "parent_asin", "label",
        *TEMPORAL_CATEGORICALS,
    }
    extras = sorted(
        c for c in df.columns
        if c not in known and (
            c.startswith("source_") or c.endswith("_rank")
            or c.endswith("_score"))
    )
    features = tuple(base + extras)
    log1p = set(TEMPORAL_LOG1P) | {
        c for c in extras if c.endswith("_rank") or c.endswith("_score")
    }
    return features, log1p


def build_temporal_feature_spec(train_df: pd.DataFrame,
                                features=None, log1p=None) -> RankerFeatureSpec:
    """Norm stats + categorical vocabs from the given TRAINING rows only."""
    if features is None:
        features, log1p = TEMPORAL_DENSE_FEATURES, TEMPORAL_LOG1P
    dense = np.zeros((len(train_df), len(features)), dtype=np.float64)
    for j, col in enumerate(features):
        x = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        if col in log1p:
            x = np.log1p(np.maximum(x, 0.0))
        dense[:, j] = x
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    cat_vocabs = {c: _build_cat_vocab(train_df[c]) for c in TEMPORAL_CATEGORICALS}
    return RankerFeatureSpec(
        n_dense=len(features),
        dense_mean=mean, dense_std=std, cat_vocabs=cat_vocabs,
    )


def build_temporal_tensors(df: pd.DataFrame, spec: RankerFeatureSpec,
                           features=None, log1p=None,
                           snapshot: Optional[str] = None) -> Dict[str, torch.Tensor]:
    if features is None:
        features, log1p = TEMPORAL_DENSE_FEATURES, TEMPORAL_LOG1P
    dense = np.zeros((len(df), len(features)), dtype=np.float32)
    for j, col in enumerate(features):
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        if col in log1p:
            x = np.log1p(np.maximum(x, 0.0))
        dense[:, j] = ((x - spec.dense_mean[j]) / spec.dense_std[j]).astype(np.float32)
    out = {
        "dense": torch.from_numpy(dense),
        "label": torch.from_numpy(df["label"].to_numpy(np.float32)),
    }
    for c, vocab in spec.cat_vocabs.items():
        out[f"cat__{c}"] = torch.from_numpy(
            df[c].astype(str).map(vocab).fillna(0).to_numpy(np.int64)
        )
    if _SEQ is not None:
        if snapshot is None:
            # combined refit frame: rows are train-then-selection by concat;
            # the caller passes the per-row snapshot column instead
            snaps = df["snapshot"].to_numpy()
            parts = []
            for sn in ("ranker_train", "model_selection", "test"):
                m = snaps == sn
                if m.any():
                    parts.append((np.flatnonzero(m), _SEQ.tensors(df[m], sn)))
            n = len(df)
            hist = torch.zeros((n, _SEQ.hist_len, 5), dtype=torch.int64)
            cand = torch.zeros((n, 3), dtype=torch.int64)
            for idx, t in parts:
                hist[idx] = t["seq__hist"]; cand[idx] = t["seq__cand"]
            out["seq__hist"], out["seq__cand"] = hist, cand
        else:
            out.update(_SEQ.tensors(df, snapshot))
    return out


def _make_model(arch: str, spec: RankerFeatureSpec) -> nn.Module:
    if arch == "mlp":
        return MLPRanker(spec, MLPRankerConfig())
    if arch == "deep_cross":
        return DeepCrossRanker(spec, DeepCrossConfig())
    if arch.startswith("seq:"):
        # arch = "seq:<variant>[:pos=<abs|delta|both>][:d=<emb>][:L=<layers>][:H=<heads>]"
        from scripts.ranker.seq_ranker import SeqRanker, SeqConfig
        assert _SEQ is not None, "seq arch needs the sequence context (seq_arm)"
        cfg = SeqConfig.parse(arch)
        return SeqRanker(spec, _SEQ.n_items, _SEQ.n_side, _SEQ.N_ABS,
                         _SEQ.n_delta, cfg)
    raise ValueError(f"unknown arch {arch!r}")


def _score_df(model: nn.Module, inputs: Dict[str, torch.Tensor],
              device: str, batch_size: int = 65536) -> np.ndarray:
    model.eval()
    feats = {k: v for k, v in inputs.items() if k != "label"}
    n = next(iter(feats.values())).shape[0]
    chunks = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            sub = {k: v[s:e].to(device) for k, v in feats.items()}
            chunks.append(model(**sub).cpu())
    return torch.cat(chunks).numpy() if chunks else np.empty(0)


def train_fixed_epochs(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    n_epochs: int,
    *,
    batch_size: int, lr: float, weight_decay: float,
    grad_clip: float = 5.0, device: str = "cpu", seed: int = 42,
    pos_weight: Optional[float] = None,
) -> List[Dict]:
    """Deterministic-schedule training: exactly n_epochs, NO evaluation, NO
    early stopping, NO best-state selection. Used for the final refit so test
    labels cannot influence when training stops."""
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    dev = {k: v.to(device) for k, v in inputs.items()}
    labels = dev["label"]
    n = labels.numel()
    pw = torch.tensor(float(pos_weight), device=device) if pos_weight else None
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    log = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=rng).to(device)
        sum_loss, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:min(start + batch_size, n)]
            feats = {k: v.index_select(0, idx) for k, v in dev.items() if k != "label"}
            y = labels.index_select(0, idx)
            logits = model(**feats)
            loss = bce(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = y.numel()
            seen += bs
            sum_loss += float(loss.detach()) * bs
        log.append({"epoch": epoch, "train_loss": sum_loss / max(1, seen)})
        print(f"  refit ep{epoch:02d}/{n_epochs} loss={sum_loss/max(1,seen):.4f}",
              flush=True)
    return log


def _cohort_of(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"


def _gt_sets(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """gt[u] = the SET of that user's target items.

    Replaces `{u: {pa} for u, pa in zip(...)}`, which reads like it builds sets
    but is a per-ROW dict comprehension: with several groundtruth rows per user
    later keys overwrite earlier ones and each user silently keeps exactly one
    item. Under all_positive_labels' (user, ts, item) ordering the survivor is
    deterministically the user's LATEST positive, so the metric would answer a
    third question that is neither the next-item nor the coverage one.
    """
    s = pd.DataFrame({"u": gt_df["user_id"].astype(str),
                      "i": gt_df["parent_asin"].astype(str)})
    out = {u: set(g) for u, g in s.groupby("u")["i"].agg(set).items()}
    assert sum(len(v) for v in out.values()) == len(s), "groundtruth rows dropped"
    return out


# Columns carried into prediction dumps so the calibration analysis never has
# to re-open the multi-GB candidate tables: ids + label + the two bucketing
# keys from the advisor spec (user history cohort, item review-count bucket).
def _dump_prediction_frame(path: Path, df: pd.DataFrame, logits: np.ndarray) -> None:
    if len(df) != len(logits):
        raise ValueError(f"logits/rows mismatch: {len(logits)} vs {len(df)}")
    out = pd.DataFrame({
        "user_id": df["user_id"].astype(str).to_numpy(),
        "parent_asin": df["parent_asin"].astype(str).to_numpy(),
        "label": pd.to_numeric(df["label"], errors="coerce").fillna(0).to_numpy(np.int8),
        "logit": logits.astype(np.float32),
        "n_history_events": pd.to_numeric(
            df["n_history_events"], errors="coerce").fillna(0).to_numpy(np.int32),
        "item_n_reviews_hist": pd.to_numeric(
            df["item_n_reviews_hist"], errors="coerce").fillna(0).to_numpy(np.float32),
    })
    tmp = path.with_name(path.name + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def _eval_test(
    model: nn.Module,
    test_df: pd.DataFrame,
    test_inputs: Dict[str, torch.Tensor],
    gt_df: pd.DataFrame,
    device: str,
    logits: Optional[np.ndarray] = None,
) -> Dict:
    """Final test evaluation with all spec-required breakdowns."""
    if logits is None:
        logits = _score_df(model, test_inputs, device)
    topk = _scores_to_topk(
        logits, test_df["user_id"].to_numpy(), test_df["parent_asin"].to_numpy(), k=100,
    )
    gt = _gt_sets(gt_df)
    pool = set(test_df["parent_asin"].astype(str))
    overall = build_split_report("test", topk, gt, pool,
                                 "candidate_union_top100", KS)

    cand_per_user: Dict[str, Set[str]] = {
        u: set(g) for u, g in
        test_df.groupby("user_id")["parent_asin"].agg(set).items()
    }
    # A user counts as "retrieved" if ANY of their positives made the candidate
    # union. next(iter(g)) used to stand in for this -- correct only while g was
    # a singleton, and nondeterministic once it is not (set iteration over str
    # is hash-randomised per process).
    retrieved_users = {u for u, g in gt.items()
                       if g & cand_per_user.get(u, set())}
    ceiling = len(retrieved_users) / len(gt) if gt else 0.0
    # Item-level ceiling: the fraction of PURCHASES the union covers. Reported
    # separately because the user-level figure above is the frozen key that
    # every cross-phase comparison keys off -- its meaning must not drift.
    n_pos_total = sum(len(g) for g in gt.values())
    ceiling_positives = (
        sum(len(g & cand_per_user.get(u, set())) for u, g in gt.items())
        / n_pos_total) if n_pos_total else 0.0
    gt_retrieved = {u: g for u, g in gt.items() if u in retrieved_users}
    conditional = {}
    for k in KS:
        r, p, n = recall_precision_at_k(topk, gt_retrieved, k)
        conditional[f"Recall@{k}"] = r
        conditional[f"Precision@{k}"] = p

    n_hist = dict(zip(gt_df["user_id"].astype(str),
                      gt_df["n_history_events"].astype(int)))
    cohorts = {}
    for name in ("0", "1", "2", "3+"):
        users = {u for u in gt if _cohort_of(n_hist.get(u, 0)) == name}
        sub_gt = {u: g for u, g in gt.items() if u in users}
        rep = {}
        for k in KS:
            r, p, n = recall_precision_at_k(topk, sub_gt, k)
            rep[f"Recall@{k}"] = r
            rep[f"Precision@{k}"] = p
        cohorts[name] = {"n_users": len(users), **rep}
    # warm/cold is a property of the TARGET ITEM, not of the user, so the split
    # has to be per (user, item). Keying it off a per-user dict assigned every
    # user with mixed warm/cold targets wholly to whichever row came last.
    item_cohorts = {}
    for name, flag in (("warm_target_item", 1), ("cold_target_item", 0)):
        sub = gt_df[gt_df["item_in_history"].astype(int) == flag]
        sub_gt = _gt_sets(sub)
        rep = {}
        for k in KS:
            r, p, n = recall_precision_at_k(topk, sub_gt, k)
            rep[f"Recall@{k}"] = r
        item_cohorts[name] = {
            "n_users": len(sub_gt),
            "n_targets": sum(len(g) for g in sub_gt.values()),
            **rep,
        }

    return {
        "n_groundtruth_users": len(gt),
        "n_groundtruth_positives": n_pos_total,
        "overall": overall.metrics,
        "candidate_ceiling_retrieved_coverage": ceiling,
        "candidate_ceiling_positive_coverage": ceiling_positives,
        "conditional_given_retrieved": {
            "n_users_retrieved": len(retrieved_users), **conditional,
        },
        "history_cohorts": cohorts,
        "target_item_cohorts": item_cohorts,
    }


def run(
    category: str,
    *,
    max_epochs: int = 20,
    batch_size: int = 8192,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 3,
    seed: int = 42,
    device: Optional[str] = None,
    smoke: bool = False,
    max_users_per_snapshot: int = 0,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    variant: Optional[str] = None,
    stage_b_only: bool = False,
    dump_predictions: bool = False,
    label_mode: str = "first_positive",
    seq_arm: bool = False,
    seq_grid: Sequence[str] = ("seq:vanilla:pos=delta", "seq:mh_pool:pos=delta",
                               "seq:causal:pos=delta", "seq:hstu:pos=delta"),
) -> Dict:
    """Full run: selection -> refit -> single locked test eval.

    Phase 3A additions: `variant` reads candidates from
    temporal_ranker/variants/{variant}/ (ground truth still comes from the
    base dir -- it is split-defined, not config-defined). `stage_b_only`
    stops after model selection and reports selection-snapshot metrics ONLY;
    the test snapshot is neither read nor required to exist.

    `label_mode` selects the ground-truth frame: "first_positive" reads the
    frozen groundtruth_{snapshot}.parquet (each user's FIRST positive in the
    window -- next-item framing), "all_positive" reads
    groundtruth_all_{snapshot}.parquet (every distinct positive -- the
    coverage framing "did we recommend what the user went on to buy"). The
    candidate rows are identical either way; only the label column and the
    evaluation denominators differ. Both stay runnable so the frozen numbers
    remain reproducible.

    Evaluation & Calibration additions: `dump_predictions` writes per-row
    logits to {candidate_dir}/predictions/ — the SELECTION-phase best model
    scored on model_selection (out-of-sample; the only legal fitting data for
    calibration under test-lock) plus the refit model scored on all three
    snapshots (train/selection are in-sample for it). Dump runs should pass a
    non-default `results_dir` so the locked one-shot test report is never
    overwritten. Ignored in stage_b_only mode.
    """
    t0 = time.time()
    # Value-level guard (the CLI has its own): a dump run re-executes the
    # locked test eval, so its report must never land where the frozen
    # one-shot reports live — whatever path the caller passed.
    if (dump_predictions and not stage_b_only
            and Path(results_dir).resolve() == RESULTS_DIR.resolve()):
        raise ValueError(
            "dump_predictions reruns the locked test eval; pass a results_dir "
            f"outside {RESULTS_DIR} so the frozen report is never overwritten")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tdir = Path(processed_dir) / category / "temporal_ranker"
    cdir = (tdir / "variants" / variant) if variant else tdir

    # Memory-bounded runs (login-node smoke): deterministic per-snapshot user
    # subsample, applied AT READ TIME via a parquet filter -- loading the full
    # candidate table first and subsetting after peaks at several GB and gets
    # OOM-killed inside the 6GB login-node cgroup. Ground truth is subset to
    # the SAME users so denominators stay honest (scale cap, not a filter).
    cands = {}
    kept_users: Dict[str, set] = {}
    rng = np.random.default_rng(seed)
    snaps_needed = (("ranker_train", "model_selection") if stage_b_only
                    else ("ranker_train", "model_selection", "test"))
    for snap in snaps_needed:
        path = cdir / f"candidates_{snap}.parquet"
        if max_users_per_snapshot:
            users = np.array(sorted(pd.read_parquet(
                path, columns=["user_id"])["user_id"].astype(str).unique()))
            if len(users) > max_users_per_snapshot:
                keep = sorted(rng.choice(users, size=max_users_per_snapshot,
                                         replace=False))
                kept_users[snap] = set(keep)
                df = pd.read_parquet(path, filters=[("user_id", "in", keep)])
            else:
                df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)
        df["user_id"] = df["user_id"].astype(str)
        df["parent_asin"] = df["parent_asin"].astype(str)
        cands[snap] = df.reset_index(drop=True)
    if label_mode not in ("first_positive", "all_positive"):
        raise ValueError(f"unknown label_mode {label_mode!r}")
    gt_prefix = "groundtruth_" if label_mode == "first_positive" else "groundtruth_all_"
    gt_sel = pd.read_parquet(tdir / f"{gt_prefix}model_selection.parquet")
    gt_test = (None if stage_b_only
               else pd.read_parquet(tdir / f"{gt_prefix}test.parquet"))
    if "model_selection" in kept_users:
        gt_sel = gt_sel[gt_sel["user_id"].astype(str).isin(
            kept_users["model_selection"])]
    if gt_test is not None and "test" in kept_users:
        gt_test = gt_test[gt_test["user_id"].astype(str).isin(kept_users["test"])]
    if max_users_per_snapshot:
        print(f"[ranker] capped users/snapshot to {max_users_per_snapshot:,} "
              f"(smoke scale cap, filtered at read)", flush=True)

    # Dense feature list: frozen base + any Phase 3A extra-source columns.
    features, log1p = resolve_feature_list(cands["ranker_train"])

    # Wave 3 sequence arm: build the history context ONCE (ranker_train-frozen
    # item vocab; per-snapshot last-L positives, all ts < cutoff).
    global _SEQ
    _SEQ = _SeqContext(tdir, snaps_needed) if seq_arm else None

    # Selection grid. Architecture + learning rate; epoch count comes from
    # early stopping against model_selection. Smoke restricts to one config.
    grid: List[Tuple[str, float]] = (
        [("seq:causal:pos=delta", 1e-3)] if (smoke and seq_arm) else
        [("mlp", 1e-3)] if smoke
        else [("mlp", 1e-3), ("mlp", 3e-4), ("deep_cross", 1e-3), ("deep_cross", 3e-4)]
    )
    if seq_arm and not smoke:
        grid = grid + [(a, lr) for a in seq_grid for lr in (1e-3, 3e-4)]
    if smoke:
        max_epochs = min(max_epochs, 3)

    # Feature spec for SELECTION runs: ranker_train rows only.
    spec_sel = build_temporal_feature_spec(cands["ranker_train"], features, log1p)
    train_inputs = build_temporal_tensors(cands["ranker_train"], spec_sel,
                                          features, log1p, snapshot="ranker_train")
    sel_inputs = build_temporal_tensors(cands["model_selection"], spec_sel,
                                        features, log1p, snapshot="model_selection")
    sel_gt = _gt_sets(gt_sel)
    sel_pool = set(cands["model_selection"]["parent_asin"])
    n_pos = int(cands["ranker_train"]["label"].sum())
    n_neg = int((cands["ranker_train"]["label"] == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    print(f"[ranker] train rows={len(cands['ranker_train']):,} "
          f"(pos={n_pos:,}) selection rows={len(cands['model_selection']):,} "
          f"pos_weight={pos_weight:.1f} device={device}", flush=True)

    selection_runs = []
    best = None  # (recall, -idx) maximize; tie -> earlier grid entry
    best_model_state = None
    for gi, (arch, lr) in enumerate(grid):
        print(f"[ranker] selection run {gi+1}/{len(grid)}: arch={arch} lr={lr}",
              flush=True)
        # Seed BEFORE construction: model init draws from the global RNG, and
        # unseeded init makes selection irreproducible across processes.
        torch.manual_seed(seed + 1000 * (gi + 1))
        model = _make_model(arch, spec_sel).to(device)
        summary = train_pointwise_bce(
            model,
            train_inputs=train_inputs,
            val_inputs=sel_inputs,
            val_user_ids=cands["model_selection"]["user_id"].to_numpy(),
            val_pa=cands["model_selection"]["parent_asin"].to_numpy(),
            val_groundtruth=sel_gt,
            candidate_pool=sel_pool,
            epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            pos_weight=pos_weight,
        )
        hist = summary["history"]
        recalls = [h["ranker_val_metrics"]["Recall@100"] for h in hist]
        best_epoch = int(np.argmax(recalls)) + 1
        run_rec = {
            "arch": arch, "lr": lr,
            "best_selection_recall@100": summary["best_ranker_val_recall@100"],
            "best_epoch": best_epoch,
            "n_epochs_run": len(hist),
        }
        selection_runs.append(run_rec)
        key = (summary["best_ranker_val_recall@100"], -gi)
        if best is None or key > best[0]:
            best = (key, run_rec)
            if stage_b_only or dump_predictions:
                # train_pointwise_bce reloaded the run's best epoch already.
                best_model_state = {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
        del model

    chosen = best[1]
    print(f"[ranker] SELECTED arch={chosen['arch']} lr={chosen['lr']} "
          f"epochs={chosen['best_epoch']} "
          f"(selection R@100={chosen['best_selection_recall@100']:.4f})", flush=True)

    if stage_b_only:
        # ---- Stage B report: selection-snapshot metrics ONLY ------------------
        model = _make_model(chosen["arch"], spec_sel).to(device)
        model.load_state_dict(best_model_state)
        sel_df = cands["model_selection"]
        sel_eval = _eval_test(model, sel_df,
                              build_temporal_tensors(sel_df, spec_sel,
                                                     features, log1p,
                                                     snapshot="model_selection"),
                              gt_sel, device)
        report = {
            "category": category,
            "variant": variant,
        "label_mode": label_mode,
            "stage": "B (model-selection confirmation; test snapshot untouched)",
            "started_utc": datetime.now(tz=timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - t0, 2),
            "features": list(features),
            "selection": {"grid": selection_runs, "chosen": chosen,
                          "metric": "model_selection Recall@100"},
            "model_selection_eval": sel_eval,
        }
        # Default location keeps the Phase 3A / P4 precedent; a non-default
        # results_dir (wave-3 seq sweeps run many specs on the SAME variant)
        # redirects the report so parallel runs never clobber each other or
        # the frozen Stage B verdicts.
        if Path(results_dir).resolve() == RESULTS_DIR.resolve():
            stage_b_dir = REPO_ROOT / "results" / "phase3a"
        else:
            stage_b_dir = Path(results_dir)
        stage_b_dir.mkdir(parents=True, exist_ok=True)
        seq_tag = ""
        if seq_arm:
            seq_tag = "_" + "+".join(a.replace(":", "_").replace("=", "_")
                                     for a in seq_grid)
        out = stage_b_dir / f"{category}_stageB_{variant or 'base'}{seq_tag}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[ranker] Stage B wrote {out} | selection R@100="
              f"{sel_eval['overall']['Recall@100']:.4f} "
              f"ceiling={sel_eval['candidate_ceiling_retrieved_coverage']:.4f}",
              flush=True)
        return report

    # Out-of-sample selection-model scores on model_selection: fitted before
    # the selection tensors are freed. This is the calibration fitting set.
    selmodel_sel_logits: Optional[np.ndarray] = None
    if dump_predictions:
        sel_model = _make_model(chosen["arch"], spec_sel).to(device)
        sel_model.load_state_dict(best_model_state)
        selmodel_sel_logits = _score_df(sel_model, sel_inputs, device)
        del sel_model
        best_model_state = None

    # Free the selection-phase tensors before building the refit set -- the
    # login-node cgroup cannot hold both generations at once.
    import gc
    del train_inputs, sel_inputs
    gc.collect()

    # ---- refit on train + selection rows, fixed epochs, no early stopping ----
    combined = pd.concat([cands["ranker_train"], cands["model_selection"]],
                         ignore_index=True)
    spec_final = build_temporal_feature_spec(combined, features, log1p)
    combined_inputs = build_temporal_tensors(combined, spec_final,
                                             features, log1p)
    n_pos_c = int(combined["label"].sum())
    n_neg_c = int((combined["label"] == 0).sum())
    pos_weight_c = n_neg_c / max(1, n_pos_c)
    torch.manual_seed(seed + 999)
    final_model = _make_model(chosen["arch"], spec_final).to(device)
    print(f"[ranker] refit on {len(combined):,} rows for exactly "
          f"{chosen['best_epoch']} epoch(s)...", flush=True)
    refit_log = train_fixed_epochs(
        final_model, combined_inputs, chosen["best_epoch"],
        batch_size=batch_size, lr=chosen["lr"], weight_decay=weight_decay,
        device=device, seed=seed, pos_weight=pos_weight_c,
    )

    # Refit-model scores on the refit rows (combined = train then selection,
    # order preserved by concat) captured before the tensors are freed.
    refit_combined_logits: Optional[np.ndarray] = None
    if dump_predictions:
        refit_combined_logits = _score_df(final_model, combined_inputs, device)

    # ---- final test -----------------------------------------------------------
    del combined_inputs
    gc.collect()
    test_inputs = build_temporal_tensors(cands["test"], spec_final,
                                         features, log1p, snapshot="test")
    test_logits = _score_df(final_model, test_inputs, device)
    test_eval = _eval_test(final_model, cands["test"], test_inputs, gt_test,
                           device, logits=test_logits)

    if dump_predictions:
        # Capped/smoke dumps are NOT the canonical scored model — quarantine
        # them so a full-scale analysis can never silently read them.
        capped = bool(max_users_per_snapshot) or smoke
        pred_dir = cdir / ("predictions_smoke" if capped else "predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        n_train = len(cands["ranker_train"])
        _dump_prediction_frame(
            pred_dir / "selection_model_model_selection.parquet",
            cands["model_selection"], selmodel_sel_logits)
        _dump_prediction_frame(
            pred_dir / "refit_model_ranker_train.parquet",
            cands["ranker_train"], refit_combined_logits[:n_train])
        _dump_prediction_frame(
            pred_dir / "refit_model_model_selection.parquet",
            cands["model_selection"], refit_combined_logits[n_train:])
        _dump_prediction_frame(
            pred_dir / "refit_model_test.parquet", cands["test"], test_logits)
        meta = {
            "category": category,
            "variant": variant,
        "label_mode": label_mode,
            "seed": seed,
            "device": device,
            "max_users_per_snapshot": max_users_per_snapshot,
            "smoke": smoke,
            "created_utc": datetime.now(tz=timezone.utc).isoformat(),
            "selection_model": {
                "arch": chosen["arch"], "lr": chosen["lr"],
                "best_epoch": chosen["best_epoch"],
                "pos_weight": pos_weight,
                "trained_on": "candidates_ranker_train (labels T0->T1)",
                "out_of_sample_on": ["model_selection"],
            },
            "refit_model": {
                "arch": chosen["arch"], "lr": chosen["lr"],
                "fixed_epochs": chosen["best_epoch"],
                "pos_weight": pos_weight_c,
                "trained_on": "ranker_train + model_selection",
                "out_of_sample_on": ["test"],
            },
            "n_rows": {
                "ranker_train": n_train,
                "model_selection": int(len(cands["model_selection"])),
                "test": int(len(cands["test"])),
            },
            "note": ("calibration must be FITTED on "
                     "selection_model_model_selection.parquet only "
                     "(out-of-sample, pre-test); refit train/selection dumps "
                     "are in-sample and for distribution analysis only"),
        }
        with open(pred_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ranker] dumped predictions to {pred_dir}", flush=True)

    # Retrieval-only context on the same test candidates.
    retrieval_only = {}
    gt_all = _gt_sets(gt_test)
    test_pool = set(cands["test"]["parent_asin"])
    for col, name in (("two_tower_score", "two_tower"),
                      ("popularity_score", "popularity"),
                      ("rule_score", "rule_based")):
        topk = _scores_to_topk(
            cands["test"][col].to_numpy(np.float32),
            cands["test"]["user_id"].to_numpy(),
            cands["test"]["parent_asin"].to_numpy(), k=100,
        )
        rep = build_split_report(f"test_{name}", topk, gt_all, test_pool,
                                 "candidate_union_top100", KS)
        retrieval_only[name] = rep.metrics

    report = {
        "category": category,
        "variant": variant,
        "label_mode": label_mode,
        "features": list(features),
        "protocol": "three_snapshot_walk_forward",
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "device": device,
        "provenance": {
            "training_data": "candidates_ranker_train.parquet (labels T0->T1)",
            "selection_data": "candidates_model_selection.parquet (labels T1->T2)",
            "refit_data": "ranker_train + model_selection candidates",
            "refit_fixed_epochs": chosen["best_epoch"],
            "no_early_stopping_on_refit": True,
            "test_labels_used_for_fitting_or_tuning": False,
        },
        "selection": {
            "grid": selection_runs,
            "chosen": chosen,
            "metric": "model_selection Recall@100",
        },
        "refit": {
            "n_rows": int(len(combined)),
            "pos_weight": pos_weight_c,
            "log": refit_log,
        },
        "final_test": test_eval,
        "retrieval_only_on_test_candidates": retrieval_only,
        "config": {
            "max_epochs": max_epochs, "batch_size": batch_size,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed, "smoke": smoke,
            "dense_features": list(TEMPORAL_DENSE_FEATURES),
            "categorical_features": list(TEMPORAL_CATEGORICALS),
        },
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    out = results_dir / f"{category}_ranker{suffix}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[ranker] wrote {out} | final test R@10="
          f"{test_eval['overall']['Recall@10']:.4f} "
          f"R@100={test_eval['overall']['Recall@100']:.4f} "
          f"ceiling={test_eval['candidate_ceiling_retrieved_coverage']:.4f}",
          flush=True)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", required=True)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-users", type=int, default=0,
                   help="Per-snapshot deterministic user cap (0 = all users). "
                        "For memory-bounded smoke runs only.")
    p.add_argument("--variant", default=None,
                   help="Read candidates from temporal_ranker/variants/{name}/")
    p.add_argument("--label-mode", default="first_positive",
                   choices=["first_positive", "all_positive"],
                   help="Ground-truth frame: each user's first window positive "
                        "(frozen default) or every distinct window positive")
    p.add_argument("--stage-b", action="store_true",
                   help="Stop after model selection; never touch the test "
                        "snapshot (Phase 3A Stage B confirmation).")
    p.add_argument("--dump-predictions", action="store_true",
                   help="Write per-row logits to the candidate dir's "
                        "predictions/ subdir (calibration analysis input).")
    p.add_argument("--seq-arm", action="store_true",
                   help="Wave 3: add the DIN sequence ranker (arch=din, two "
                        "lrs) to the selection grid. Default off = frozen "
                        "protocol byte-for-byte.")
    p.add_argument("--seq-grid", default=None,
                   help="Comma-separated seq arch specs to add to the grid "
                        "(each at lr 1e-3 and 3e-4), e.g. "
                        "'seq:causal:pos=delta:d=64:L=2:H=2,seq:hstu:pos=both'. "
                        "Default: the four advisor variants at pos=delta.")
    p.add_argument("--results-dir", default=None,
                   help="Override the report output dir. REQUIRED with "
                        "--dump-predictions so the locked one-shot test "
                        "report is never overwritten by a rerun.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.dump_predictions and not args.stage_b and args.results_dir is None:
        raise SystemExit("--dump-predictions reruns the locked test eval; "
                         "pass --results-dir to keep the frozen report intact")
    kwargs = {}
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    run(args.category, max_epochs=args.max_epochs, batch_size=args.batch_size,
        seed=args.seed, smoke=args.smoke, max_users_per_snapshot=args.max_users,
        variant=args.variant, stage_b_only=args.stage_b,
        label_mode=args.label_mode,
        seq_arm=args.seq_arm,
        **({"seq_grid": tuple(args.seq_grid.split(","))} if args.seq_grid else {}),
        dump_predictions=args.dump_predictions, **kwargs)


if __name__ == "__main__":
    main()
