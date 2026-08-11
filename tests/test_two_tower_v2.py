"""Track A two-tower v2 unit gates (spec P1.6 / P3.1, worktree branch only).

Covers the spec-locked gates:
  * defaults preserve v1 (use_hist_pool=False, loss_mode='bce',
    optimizer='adam', FeatureSpec tail fields default None);
  * hard-negative pool decoupled from the old use_hard_negatives flag;
  * logQ math on a hand-computable 3-row batch, all three masks exercised;
  * hist-pool masking: zero-history user gets the EXACT zero vector, target
    masking removes the scored item, masking-to-empty also yields zeros;
  * AdamW no-decay group split (all embedding tables + logit_scale);
  * frozen config missing loss_mode/optimizer -> refuse-to-load;
  * recency weight on positive pairs (hand-computed exp decay);
  * temporal hist-array builder (positive-only, in-pool-only, recent-L,
    deterministic tie-break) + a deterministic softmax-epoch integration run.

The cross-version "loss_mode='bce' with new flags off reproduces the pre-diff
loss trajectory bit-for-bit" gate runs OUTSIDE pytest against a base-commit
checkout (it needs two code versions at once); see the P3.1 gate report.

Run from repo root (conda env tae):

    pytest tests/test_two_tower_v2.py
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.retrieval.two_tower import (  # noqa: E402
    ItemTower,
    TwoTower,
    TwoTowerConfig,
    UserTower,
    build_optimizer,
    split_decay_param_groups,
)
from scripts.retrieval.two_tower_dataset import (  # noqa: E402
    MS_PER_DAY,
    FeatureSpec,
    PairSamplingConfig,
    TwoTowerPairDataset,
)
from scripts.retrieval.train_two_tower import (  # noqa: E402
    TrainConfig,
    _move_spec_to_tensors,
    inbatch_softmax_ce,
    train_one_epoch_softmax,
)
from scripts.retrieval.temporal_two_tower import (  # noqa: E402
    build_temporal_feature_spec,
    validate_frozen_config,
)


# ---- fixtures ----------------------------------------------------------------

def _tiny_spec() -> FeatureSpec:
    """PAD + 3 users (U1..U3), PAD + 3 items (I1..I3), all-zero dense tables."""
    return FeatureSpec(
        n_users=4, n_items=4, n_stores=2, n_main_cats=2, n_deeper_cats=1,
        has_deeper_cat=False,
        user_id_to_idx={"<PAD>": 0, "U1": 1, "U2": 2, "U3": 3},
        item_id_to_idx={"<PAD>": 0, "I1": 1, "I2": 2, "I3": 3},
        store_to_idx={"<PAD>": 0, "S": 1},
        main_cat_to_idx={"<PAD>": 0, "C": 1},
        deeper_cat_to_idx={"<PAD>": 0},
        user_dense=np.zeros((4, 2), dtype=np.float32),
        item_dense=np.zeros((4, 3), dtype=np.float32),
        item_store_idx=np.array([0, 1, 1, 1], dtype=np.int64),
        item_main_cat_idx=np.array([0, 1, 1, 1], dtype=np.int64),
        item_deeper_cat_idx=np.zeros(4, dtype=np.int64),
        candidate_item_idx=np.array([1, 2, 3], dtype=np.int64),
        user_seen_per_user_idx={},
    )


def _pairs_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "label", "timestamp"])


def _hist_cfg(**kw) -> TwoTowerConfig:
    base = dict(embedding_dim=8, hidden_dim=8, id_emb_dim=4, cat_emb_dim=2,
                dropout=0.0, use_user_id_emb=False, use_item_id_emb=True,
                use_deeper_cat_emb=False, use_hist_pool=True)
    base.update(kw)
    return TwoTowerConfig(**base)


def _hist_towers(cfg: TwoTowerConfig, n_users=4, n_items=4, n_user_dense=2,
                 n_item_dense=3):
    item_tower = ItemTower(n_items=n_items, n_stores=2, n_main_cats=2,
                           n_deeper_cats=1, n_dense=n_item_dense, cfg=cfg,
                           has_deeper_cat=False)
    user_tower = UserTower(
        n_users, n_user_dense, cfg,
        shared_item_emb=item_tower.item_id_emb if cfg.use_hist_pool else None,
    )
    return user_tower, item_tower


# ---- 1. defaults preserve v1 -------------------------------------------------

def test_v2_defaults_preserve_v1():
    cfg = TwoTowerConfig()
    assert cfg.use_hist_pool is False
    assert cfg.hist_pool_len == 20
    assert cfg.hist_pool_recency_days is None

    pc = PairSamplingConfig()
    assert pc.loss_mode == "bce"
    assert pc.hard_negative_lambda == 0.0
    assert pc.pair_recency_decay_days is None

    tc = TrainConfig()
    assert tc.optimizer == "adam"

    # FeatureSpec: the new fields sit at the TAIL with None defaults so the
    # existing positional field order is untouched (Track A correction 4).
    fs_fields = dataclasses.fields(FeatureSpec)
    assert [f.name for f in fs_fields[-2:]] == ["user_hist_idx", "user_hist_w"]
    assert fs_fields[-2].default is None and fs_fields[-1].default is None


# ---- 2. hard-negative pool decoupled from use_hard_negatives ----------------

def test_hard_pool_decoupled_from_flag():
    df = _pairs_df([
        ("U1", "I1", 1, 10), ("U2", "I2", 1, 20),
        ("U1", "I3", 0, 30), ("U3", "I1", 0, 40),
    ])
    spec = _tiny_spec()

    pc_off = PairSamplingConfig(n_soft_neg=1, use_hard_negatives=False, seed=0)
    ds = TwoTowerPairDataset(df, spec, pc_off)
    # Pool built unconditionally (Track A correction 2)...
    assert ds.n_hard_pool == 2
    # ...but the BCE flat tensors still respect the flag (v1 semantics).
    assert ds.composition() == {"n_positive": 2, "n_hard_neg": 0, "n_soft_neg": 2}
    assert len(ds) == 4
    assert float(ds._label_t.sum()) == 2.0

    pc_on = dataclasses.replace(pc_off, use_hard_negatives=True)
    ds_on = TwoTowerPairDataset(df, spec, pc_on)
    assert ds_on.n_hard_pool == 2
    assert ds_on.composition()["n_hard_neg"] == 2
    assert len(ds_on) == 6


# ---- 3. logQ math on a hand-computable 3-row batch --------------------------

def test_inbatch_softmax_ce_hand_computed():
    """B=3 rows: u=[1,1,2], i=[1,2,1]. Exercises all three masks:
      mask 2 (same user):      (0,1) and (1,0);
      mask 1 (duplicate item): (0,2) and (2,0);
      mask 3 (tail collision): tail item2 vs row 1's positive.
    Unigram counts: item1 x2, item2 x1 over n_pos=3."""
    u_vec = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    i_vec = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    u_idx = torch.tensor([1, 1, 2])
    i_idx = torch.tensor([1, 2, 1])
    lq1, lq2 = math.log(2 / 3), math.log(1 / 3)
    log_q_in = torch.tensor([lq1, lq2, lq1])
    scale = torch.tensor(2.0)
    tail_idx = torch.tensor([2, 3])
    tail_vec = torch.tensor([[0.0, 1.0], [0.6, 0.8]])
    log_q_tail = -math.log(3)   # uniform over the 3-item pool

    ce, n_valid_neg = inbatch_softmax_ce(
        u_vec, i_vec, u_idx, i_idx, log_q_in, scale,
        tail_vec, tail_idx, log_q_tail,
    )

    def lse(xs):
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    log3 = math.log(3)
    # row 0: survivors = diag + both tail items.
    d0 = 2.0 - lq1
    exp0 = -(d0 - lse([d0, 0.0 + log3, 1.2 + log3]))
    # row 1: survivors = diag, in-batch col 2, tail item 3 (item2 collides).
    d1 = 2.0 - lq2
    exp1 = -(d1 - lse([d1, 0.0 - lq1, 1.6 + log3]))
    # row 2: survivors = diag, in-batch col 1, both tail items.
    d2 = 2.0 - lq1
    exp2 = -(d2 - lse([d2, 0.0 - lq2, 0.0 + log3, 1.2 + log3]))

    torch.testing.assert_close(
        ce, torch.tensor([exp0, exp1, exp2]), rtol=1e-5, atol=1e-6,
    )
    assert n_valid_neg.tolist() == [2, 2, 3]

    # Without the tail, the masks alone leave row 0 with only its diagonal.
    ce_nt, nv_nt = inbatch_softmax_ce(u_vec, i_vec, u_idx, i_idx, log_q_in, scale)
    assert nv_nt.tolist() == [0, 1, 1]
    torch.testing.assert_close(ce_nt[0], torch.tensor(0.0), rtol=0, atol=1e-6)


# ---- 4. hist-pool masking ----------------------------------------------------

def test_hist_pool_zero_history_and_target_masking():
    cfg = _hist_cfg()
    user_tower, _ = _hist_towers(cfg)
    with torch.no_grad():
        user_tower.shared_item_emb.weight.copy_(torch.tensor(
            [[0.0] * 4, [1.0] * 4, [2.0] * 4, [4.0] * 4]
        ))

    # Zero-history user: all-PAD idxs, all-zero weights -> EXACT zero vector.
    pooled = user_tower.pool_history(
        torch.tensor([[0, 0, 0]]), torch.zeros(1, 3))
    assert torch.equal(pooled, torch.zeros(1, 4))

    # Mean pool over two items.
    pooled = user_tower.pool_history(
        torch.tensor([[1, 2, 0]]), torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.equal(pooled, torch.full((1, 4), 1.5))

    # Target masking removes the scored item from its own history.
    pooled = user_tower.pool_history(
        torch.tensor([[1, 2, 0]]), torch.tensor([[1.0, 1.0, 0.0]]),
        exclude_item_idx=torch.tensor([2]))
    assert torch.equal(pooled, torch.ones(1, 4))

    # Masking down to an empty history also yields the exact zero vector.
    pooled = user_tower.pool_history(
        torch.tensor([[2, 0, 0]]), torch.tensor([[1.0, 0.0, 0.0]]),
        exclude_item_idx=torch.tensor([2]))
    assert torch.equal(pooled, torch.zeros(1, 4))

    # Loud failures: no shared table / no hist tensors at forward time.
    with pytest.raises(AssertionError):
        UserTower(4, 2, cfg, shared_item_emb=None)
    with pytest.raises(AssertionError):
        user_tower(torch.tensor([1]), torch.zeros(1, 2))


# ---- 5. AdamW no-decay group split ------------------------------------------

def test_adamw_no_decay_group_split():
    cfg = TwoTowerConfig(embedding_dim=8, hidden_dim=8, id_emb_dim=4,
                         cat_emb_dim=2, dropout=0.0, use_user_id_emb=True,
                         use_item_id_emb=True, use_deeper_cat_emb=True,
                         use_hist_pool=True)
    item_tower = ItemTower(n_items=4, n_stores=2, n_main_cats=2,
                           n_deeper_cats=3, n_dense=3, cfg=cfg,
                           has_deeper_cat=True)
    user_tower = UserTower(4, 2, cfg, shared_item_emb=item_tower.item_id_emb)
    model = TwoTower(user_tower, item_tower)

    expected_no_decay = {
        id(user_tower.user_id_emb.weight),
        id(item_tower.item_id_emb.weight),   # shared with the taste channel
        id(item_tower.store_emb.weight),
        id(item_tower.main_cat_emb.weight),
        id(item_tower.deeper_cat_emb.weight),
        id(model.logit_scale),
    }

    decay, no_decay, no_decay_names = split_decay_param_groups(model)
    assert {id(p) for p in no_decay} == expected_no_decay
    assert not expected_no_decay & {id(p) for p in decay}
    # Decay group = the 2-layer MLP of each tower: 2 towers x 2 linears x (W,b).
    assert len(decay) == 8
    # Full coverage, shared param counted once.
    unique_model_params = {id(p) for p in model.parameters()}
    assert {id(p) for p in decay} | {id(p) for p in no_decay} == unique_model_params
    assert "logit_scale" in no_decay_names

    opt = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, torch.optim.AdamW)
    wd_by_group = {g["weight_decay"]: {id(p) for p in g["params"]}
                   for g in opt.param_groups}
    assert set(wd_by_group) == {0.01, 0.0}
    assert wd_by_group[0.0] == expected_no_decay

    # 'adam' reproduces the v1 construction: one group, decay applied to all.
    opt_v1 = build_optimizer(model, "adam", lr=1e-3, weight_decay=1e-5)
    assert isinstance(opt_v1, torch.optim.Adam)
    assert len(opt_v1.param_groups) == 1
    assert opt_v1.param_groups[0]["weight_decay"] == 1e-5

    with pytest.raises(ValueError):
        build_optimizer(model, "sgd", lr=1e-3, weight_decay=0.0)


# ---- 6. frozen config refuse-to-load ----------------------------------------

def test_frozen_config_refuse_to_load_pre_v2():
    v1_era = {
        "frozen_epochs": 3,
        "config": {
            "two_tower": {"embedding_dim": 64},
            "pair_sampling": {"n_soft_neg": 4, "seed": 42},
            "train": {"batch_size": 4096, "lr": 1e-3,
                      "weight_decay": 1e-5, "grad_clip": 5.0},
        },
        "seed": 42,
    }
    with pytest.raises(ValueError, match="refusing to load"):
        validate_frozen_config(v1_era)

    v2 = {
        "frozen_epochs": 3,
        "config": {
            "two_tower": {"embedding_dim": 64},
            "pair_sampling": {"n_soft_neg": 4, "seed": 42, "loss_mode": "softmax"},
            "train": {"batch_size": 4096, "lr": 1e-3, "weight_decay": 1e-5,
                      "grad_clip": 5.0, "optimizer": "adamw"},
        },
        "seed": 42,
    }
    validate_frozen_config(v2)   # must not raise


# ---- 7. recency weight on positive pairs ------------------------------------

def test_pair_recency_decay_weights():
    df = _pairs_df([
        ("U1", "I1", 1, 1 * MS_PER_DAY),                 # age 1.0 d
        ("U2", "I2", 1, MS_PER_DAY + MS_PER_DAY // 2),   # age 0.5 d
        ("U1", "I3", 0, 5),
    ])
    spec = _tiny_spec()
    pc = PairSamplingConfig(n_soft_neg=0, pair_recency_decay_days=1.0, seed=0)
    ds = TwoTowerPairDataset(df, spec, pc, as_of_ms=2 * MS_PER_DAY)
    np.testing.assert_allclose(
        ds._pos_recency_w, [math.exp(-1.0), math.exp(-0.5)], rtol=1e-6,
    )
    # Flat layout is [pos..., hard..., soft...]; positives carry the decay,
    # the hard negative keeps its plain weight.
    np.testing.assert_allclose(
        ds._weight_t.numpy(),
        [math.exp(-1.0), math.exp(-0.5), 1.0], rtol=1e-6,
    )
    # Decay without a boundary is a loud failure, not a silent no-op.
    with pytest.raises(AssertionError):
        TwoTowerPairDataset(df, spec, pc)

    # Default config: all-ones recency -> v1 flat weights bit-for-bit.
    ds_v1 = TwoTowerPairDataset(df, spec, PairSamplingConfig(n_soft_neg=0, seed=0))
    assert np.array_equal(ds_v1._weight_t.numpy(), np.ones(3, dtype=np.float32))


# ---- 8. temporal hist-array builder + deterministic softmax epoch ------------

def _toy_temporal():
    d = MS_PER_DAY
    hist = pd.DataFrame(
        [
            ("UA", "I3", 1, 0 * d),   # oldest positive -> truncated at L=2
            ("UA", "I1", 1, 1 * d),
            ("UA", "I2", 1, 2 * d),
            ("UA", "IX", 1, 4 * d),   # positive but NOT in the eligible pool
            ("UA", "I3", 0, 5 * d),   # hard negative
            ("UB", "I2", 0, 1 * d),   # hard-negative-only user -> zero taste row
            ("UC", "I1", 1, 5 * d),
        ],
        columns=["user_id", "parent_asin", "label", "timestamp"],
    )
    uf = pd.DataFrame({"user_id": ["UA", "UB", "UC"]})
    itf = pd.DataFrame({
        "parent_asin": ["I1", "I2", "I3"],
        "in_eligible_pool": [1, 1, 1],
        "store": ["S", "S", "S"],
        "main_category": ["C", "C", "C"],
    })
    return hist, uf, itf


def test_temporal_hist_arrays_positive_inpool_recent():
    hist, uf, itf = _toy_temporal()
    spec = build_temporal_feature_spec(
        hist, uf, itf, None, use_deeper_cat=False,
        build_hist_pool=True, hist_pool_len=2,
    )
    # Vocab: UA->1, UB->2, UC->3; I1->1, I2->2, I3->3.
    assert spec.user_hist_idx.shape == (4, 2)
    # UA: most recent in-pool positives are I2 (d2) then I1 (d1); I3 (d0)
    # truncated; IX (no embedding slot) and the label==0 rows excluded.
    assert spec.user_hist_idx[1].tolist() == [2, 1]
    assert spec.user_hist_w[1].tolist() == [1.0, 1.0]
    # UB: no positive history -> all-PAD, all-zero weights.
    assert spec.user_hist_idx[2].tolist() == [0, 0]
    assert spec.user_hist_w[2].tolist() == [0.0, 0.0]
    # UC: single positive.
    assert spec.user_hist_idx[3].tolist() == [1, 0]
    assert spec.user_hist_w[3].tolist() == [1.0, 0.0]

    # Recency-weighted pooling against a boundary.
    spec_r = build_temporal_feature_spec(
        hist, uf, itf, None, use_deeper_cat=False,
        build_hist_pool=True, hist_pool_len=2,
        hist_pool_recency_days=1.0, boundary_ms=6 * MS_PER_DAY,
    )
    np.testing.assert_allclose(
        spec_r.user_hist_w[1], [math.exp(-4.0), math.exp(-5.0)], rtol=1e-6,
    )
    # v1 spec builds carry no arrays at all.
    spec_off = build_temporal_feature_spec(hist, uf, itf, None, use_deeper_cat=False)
    assert spec_off.user_hist_idx is None and spec_off.user_hist_w is None


def _run_softmax_epoch(seed=0):
    hist, uf, itf = _toy_temporal()
    spec = build_temporal_feature_spec(
        hist, uf, itf, None, use_deeper_cat=False,
        build_hist_pool=True, hist_pool_len=2,
    )
    pair_cfg = PairSamplingConfig(
        n_soft_neg=0, seed=seed, loss_mode="softmax",
        hard_negative_lambda=0.5, softmax_tail_size=4,
    )
    dataset = TwoTowerPairDataset(hist, spec, pair_cfg)
    torch.manual_seed(seed)
    cfg = _hist_cfg()
    user_tower, item_tower = _hist_towers(
        cfg, n_users=spec.n_users, n_items=spec.n_items,
        n_user_dense=spec.n_user_dense, n_item_dense=spec.n_item_dense,
    )
    model = TwoTower(user_tower, item_tower)
    opt = build_optimizer(model, "adamw", lr=1e-2, weight_decay=0.01)
    spec_t = _move_spec_to_tensors(spec, "cpu")
    assert "user_hist_idx" in spec_t and "user_hist_w" in spec_t
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    metrics = train_one_epoch_softmax(
        model, dataset, spec_t, opt, "cpu", 5.0, 2, rng,
        tail_size=4, hard_negative_lambda=0.5,
    )
    return metrics, model


def test_softmax_epoch_runs_and_is_deterministic():
    m1, model1 = _run_softmax_epoch(seed=0)
    # UA(3 in-pool positives) + UC(1); the soft-neg sampler was skipped.
    assert m1["n_examples"] == 4
    for key in ("loss", "softmax_loss", "hard_neg_loss",
                "mean_logit_pos", "mean_logit_neg"):
        assert math.isfinite(m1[key]), key
    assert m1["hard_neg_loss"] > 0.0        # lambda=0.5 exercised the aux term

    m2, model2 = _run_softmax_epoch(seed=0)
    assert m1 == m2
    s1, s2 = model1.state_dict(), model2.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k

    # The shared taste table IS the item tower's table (one storage).
    assert model1.user_tower.shared_item_emb.weight.data_ptr() == \
        model1.item_tower.item_id_emb.weight.data_ptr()


def test_softmax_lambda_on_empty_hard_pool_is_loud():
    hist, uf, itf = _toy_temporal()
    hist = hist[hist["label"] == 1].reset_index(drop=True)   # no label==0 rows
    spec = build_temporal_feature_spec(
        hist, uf, itf, None, use_deeper_cat=False,
        build_hist_pool=True, hist_pool_len=2,
    )
    pair_cfg = PairSamplingConfig(n_soft_neg=0, seed=0, loss_mode="softmax",
                                  hard_negative_lambda=0.5, softmax_tail_size=2)
    dataset = TwoTowerPairDataset(hist, spec, pair_cfg)
    torch.manual_seed(0)
    cfg = _hist_cfg()
    user_tower, item_tower = _hist_towers(
        cfg, n_users=spec.n_users, n_items=spec.n_items,
        n_user_dense=spec.n_user_dense, n_item_dense=spec.n_item_dense,
    )
    model = TwoTower(user_tower, item_tower)
    opt = build_optimizer(model, "adamw", lr=1e-2, weight_decay=0.01)
    spec_t = _move_spec_to_tensors(spec, "cpu")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    with pytest.raises(AssertionError, match="hard_negative_lambda"):
        train_one_epoch_softmax(model, dataset, spec_t, opt, "cpu", 5.0, 2, rng,
                                tail_size=2, hard_negative_lambda=0.5)
