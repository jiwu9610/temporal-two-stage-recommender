"""Three-snapshot temporal ranker invariants (T0/T1/T2/T3 walk-forward).

Covers the 15 spec-required checks plus a synthetic end-to-end run
(data layer -> three two-tower checkpoints -> three candidate files ->
selection -> refit -> final test) on a hand-verifiable toy universe.

The full chain runs ONCE in a module-scoped fixture; tests assert on its
artifacts. Run from repo root (conda env tae):

    pytest tests/test_temporal_ranker.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.temporal_split import MS_PER_DAY  # noqa: E402
from scripts.data.temporal_ranker_pipeline import (  # noqa: E402
    SNAPSHOT_NAMES,
    run_category as build_data_layer,
)
from scripts.retrieval.temporal_two_tower import (  # noqa: E402
    build_temporal_feature_spec,
    run_all_snapshots,
)
from scripts.retrieval.two_tower import TwoTowerConfig  # noqa: E402
from scripts.retrieval.two_tower_dataset import PairSamplingConfig  # noqa: E402
from scripts.retrieval.train_two_tower import TrainConfig  # noqa: E402
from scripts.ranker.temporal_candidate_builder import (  # noqa: E402
    build_all as build_candidates,
)
from scripts.ranker.train_temporal_ranker import run as run_ranker  # noqa: E402


def day(n: float) -> int:
    return int(n * MS_PER_DAY)


# Global cutoffs for the toy universe: T0=40, T1=60, T2=80, T3=max+1.
T0, T1, T2 = day(40), day(60), day(80)
CAT = "SyntheticCat"


def _mk(rows):
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    df["label"] = (df["rating"] >= 3).astype("int8")
    df["label_type"] = np.where(df["label"] == 1, "positive", "hard_negative")
    return df[["user_id", "parent_asin", "rating", "label", "label_type", "timestamp"]]


def _toy_clean() -> pd.DataFrame:
    """Toy universe. Eligibility threshold is 2 history events.

    Items: IP, IP2, IP3 eligible from T0 (>=2 pre-T0 positives each).
           IQ appears only inside [T0, T1) -> eligible from T1, NOT at T0.
           IRARE has exactly 1 pre-T0 event -> never eligible.
    Users: UA warm (3 events); labels IP@45 (train), IQ@65 (selection),
           IP3@85 (test; IP3 unseen by UA).
           UB warm; label IP@50 only.
           UC zero-history at T0, first event IQ@52 -> cold train-label user.
           UD hard-negative-only in [T0,T1) -> no train label.
           UE single positive IP@90 -> cold test-label user.
           Q1..Q3 zero-history users whose IQ positives create IQ + labels.
    Soft-neg sampler safety: every user contributing a positive pair has at
    least one UNSEEN eligible item (n_soft_neg=1 in the e2e config).
    """
    rows = []
    for i, u in enumerate(["P1", "P2", "P3", "P4"]):
        rows.append((u, "IP", 5, day(1 + i)))
        rows.append((u, "IP2", 4, day(21 + i)))
    rows += [("P5", "IP3", 5, day(25)), ("P6", "IP3", 5, day(26))]
    rows += [("P1", "IRARE", 4, day(30))]

    rows += [
        ("UA", "IA1", 5, day(5)),
        ("UA", "IA2", 4, day(10)),
        ("UA", "IB1", 5, day(20)),
        ("UA", "IP", 5, day(45)),      # ranker_train label
        ("UA", "IQ", 4, day(65)),      # model_selection label
        ("UA", "IP3", 5, day(85)),     # test label (warm item, in pool)
    ]
    rows += [
        ("UB", "IB1", 5, day(8)),
        ("UB", "IB2", 3, day(18)),
        ("UB", "IB3", 4, day(28)),
        ("UB", "IP", 5, day(50)),      # ranker_train label
    ]
    rows += [
        ("UC", "IQ", 5, day(52)),      # cold ranker_train label user
        ("UC", "IP2", 4, day(66)),     # model_selection label (2 hist events @T2)
    ]
    rows += [
        ("UD", "ID1", 5, day(9)),
        ("UD", "ID2", 4, day(19)),
        ("UD", "ID3", 5, day(29)),
        ("UD", "IP", 1, day(42)),      # hard negative -> no label
    ]
    rows += [("UE", "IP", 5, day(90))]  # cold test label user
    for i, u in enumerate(["Q1", "Q2", "Q3"]):
        rows.append((u, "IQ", 5, day(45 + i)))
    return _mk(rows)


def _write_config(path: Path) -> None:
    path.write_text(
        "temporal_snapshots:\n"
        "  eligibility:\n"
        "    min_item_history_interactions: 2\n"
        "    warm_user_min_history: 3\n"
        "  features:\n"
        "    recent_days: 30\n"
        "  three_snapshot:\n"
        "    cutoffs:\n"
        f"      {CAT}: {{t0: '1970-02-10', t1: '1970-03-02', t2: '1970-03-22'}}\n"
    )


def _run_chain(root: Path, clean: pd.DataFrame, catalog: pd.DataFrame,
               reuse_tt_from: Path | None = None) -> dict:
    """data layer -> two-tower (or reuse predictions) -> candidates."""
    cat_dir = root / CAT
    cat_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(cat_dir / "interactions_clean.parquet", index=False)
    catalog.to_parquet(cat_dir / "item_features.parquet", index=False)
    cfg_path = root / "cfg.yaml"
    _write_config(cfg_path)
    manifest = build_data_layer(CAT, config_path=cfg_path, processed_dir=root)

    if reuse_tt_from is not None:
        for snap in SNAPSHOT_NAMES:
            shutil.copy(
                reuse_tt_from / CAT / "temporal_ranker"
                / f"two_tower_predictions_{snap}.parquet",
                cat_dir / "temporal_ranker" / f"two_tower_predictions_{snap}.parquet",
            )
        tt_metas = None
    else:
        tt_metas = run_all_snapshots(
            CAT,
            TwoTowerConfig(embedding_dim=8, hidden_dim=16, id_emb_dim=8,
                           cat_emb_dim=4, dropout=0.0),
            TrainConfig(epochs=2, batch_size=64, device="cpu",
                        early_stopping_patience=2),
            PairSamplingConfig(n_soft_neg=1, seed=7),
            seed=7, processed_dir=root, raw_dir=root / "no_raw",
        )
    cand_report = build_candidates(CAT, top_k=10, seed=7, processed_dir=root,
                                   results_dir=root / "results")
    return {"manifest": manifest, "tt_metas": tt_metas, "cand": cand_report,
            "dir": cat_dir / "temporal_ranker"}


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root = tmp_path_factory.mktemp("temporal_ranker_e2e")
    catalog = pd.DataFrame({
        "parent_asin": sorted(_toy_clean()["parent_asin"].unique()),
        "main_category": "C",
        "price": 1.0,
    })
    catalog["store"] = ["S" + str(i % 2) for i in range(len(catalog))]
    # Missing-categorical witness: IP3's store is NaN.
    catalog.loc[catalog["parent_asin"] == "IP3", "store"] = np.nan
    out = _run_chain(root, _toy_clean(), catalog)
    out["root"] = root
    out["ranker_report"] = run_ranker(
        CAT, smoke=True, seed=7, processed_dir=root, results_dir=root / "results",
    )
    return out


def _load(chain, name):
    return pd.read_parquet(chain["dir"] / name)


# ---- 1. all users share the same global cutoffs -----------------------------

def test_1_shared_global_cutoffs(chain):
    c = chain["manifest"]["cutoffs"]
    assert c["t0_ms"] == T0 and c["t1_ms"] == T1 and c["t2_ms"] == T2
    assert c["t0_ms"] < c["t1_ms"] < c["t2_ms"] < c["t3_ms"]
    bounds = {"ranker_train": (T0, T1), "model_selection": (T1, T2),
              "test": (T2, c["t3_ms"])}
    for snap, (hist_end, label_end) in bounds.items():
        hist = _load(chain, f"history_{snap}.parquet")
        gt = _load(chain, f"groundtruth_{snap}.parquet")
        assert (hist["timestamp"] < hist_end).all()
        assert (gt["timestamp"] >= hist_end).all()
        assert (gt["timestamp"] < label_end).all()


# ---- 2-4. checkpoint training boundaries ------------------------------------

def test_2_checkpoints_use_only_history_before_snapshot(chain):
    for snap in SNAPSHOT_NAMES:
        meta = json.loads((chain["dir"] / f"two_tower_{snap}_meta.json").read_text())
        hist = _load(chain, f"history_{snap}.parquet")
        assert int(hist["timestamp"].max()) < meta["training_boundary_ms"]
        assert meta["n_training_events"] == len(hist)


def test_3_checkpoint_b_includes_t0_t1_history(chain):
    hist_b = _load(chain, "history_model_selection.parquet")
    hist_a = _load(chain, "history_ranker_train.parquet")
    witness = hist_b.query("user_id == 'Q1' and parent_asin == 'IQ'")
    assert len(witness) == 1 and witness["timestamp"].iloc[0] == day(45)
    assert hist_a.query("user_id == 'Q1'").empty
    window = hist_b[(hist_b["timestamp"] >= T0) & (hist_b["timestamp"] < T1)]
    assert len(window) > 0 and len(hist_b) == len(hist_a) + len(window)


def test_4_checkpoint_c_includes_all_history_before_t2(chain):
    hist_c = _load(chain, "history_test.parquet")
    clean = _toy_clean()
    assert len(hist_c) == int((clean["timestamp"] < T2).sum())
    assert len(hist_c.query("user_id == 'UA' and parent_asin == 'IQ'")) == 1


# ---- 5. no future events in pairs / features / vocabularies / negatives -----

def test_5_no_future_ids_in_vocab_or_sampling(chain):
    hist = _load(chain, "history_ranker_train.parquet")
    uf = _load(chain, "user_features_ranker_train.parquet")
    itf = _load(chain, "item_features_ranker_train.parquet")
    spec = build_temporal_feature_spec(hist, uf, itf)
    # IQ exists only after T0; UC/UE/Q* have no pre-T0 events.
    assert "IQ" not in spec.item_id_to_idx
    for u in ("UC", "UE", "Q1", "Q2", "Q3"):
        assert u not in spec.user_id_to_idx
    # Soft negatives can only come from the eligible (pre-T0) pool.
    # IB1 has two pre-T0 events (UA@20, UB@8) so it clears the threshold too.
    pool = {pa for pa, i in spec.item_id_to_idx.items() if i != 0}
    assert pool == {"IP", "IP2", "IP3", "IB1"}
    # Eligible pool == candidate_item_idx slots.
    assert len(spec.candidate_item_idx) == 4


# ---- 6. three checkpoints trained separately --------------------------------

def test_6_three_separate_checkpoints(chain):
    boundaries = set()
    states = []
    for snap in SNAPSHOT_NAMES:
        meta = json.loads((chain["dir"] / f"two_tower_{snap}_meta.json").read_text())
        boundaries.add(meta["training_boundary_ms"])
        states.append(torch.load(chain["dir"] / f"two_tower_{snap}.pt",
                                 weights_only=True))
    assert boundaries == {T0, T1, T2}
    # A and C trained on different histories -> weights must differ somewhere.
    a, c = states[0], states[2]
    assert any(not torch.equal(a[k], c[k]) for k in a if k in c)


# ---- 7. hyperparameters frozen across checkpoints ----------------------------

def test_7_config_hash_frozen(chain):
    hashes, epochs = set(), {}
    for snap in SNAPSHOT_NAMES:
        meta = json.loads((chain["dir"] / f"two_tower_{snap}_meta.json").read_text())
        hashes.add(meta["config_hash"])
        epochs[snap] = meta["epochs_trained"]
    assert len(hashes) == 1
    frozen = json.loads((chain["dir"] / "two_tower_frozen_config.json").read_text())
    assert epochs["model_selection"] == frozen["frozen_epochs"]
    assert epochs["test"] == frozen["frozen_epochs"]
    # Rule weights were tuned once, on T0, and frozen.
    rw = json.loads((chain["dir"] / "rule_weights.json").read_text())
    assert "ranker_train" in rw["tuned_on"]


# ---- 8. candidate sets independently generated -------------------------------

def test_8_candidates_independent_per_snapshot(chain):
    c_train = _load(chain, "candidates_ranker_train.parquet")
    c_sel = _load(chain, "candidates_model_selection.parquet")
    # IQ only becomes eligible at T1: absent from T0 candidates, present at T1.
    assert "IQ" not in set(c_train["parent_asin"])
    assert "IQ" in set(c_sel["parent_asin"])
    # Different user sets too (labels come from different windows).
    assert set(c_train["user_id"]) != set(c_sel["user_id"])
    # Labels were NOT copied: same (user,item) key can differ across snapshots.
    assert (c_train["snapshot"] == "ranker_train").all()
    assert (c_sel["snapshot"] == "model_selection").all()


# ---- 9/10/11. ranker data provenance + test labels never used ----------------

def test_9_10_provenance_fields(chain):
    prov = chain["ranker_report"]["provenance"]
    assert "ranker_train" in prov["training_data"]
    assert "model_selection" in prov["selection_data"]
    assert prov["no_early_stopping_on_refit"] is True
    assert prov["test_labels_used_for_fitting_or_tuning"] is False
    sel = chain["ranker_report"]["selection"]
    assert sel["metric"] == "model_selection Recall@100"


def test_11_test_labels_never_affect_fit_or_selection(chain):
    """Functional proof: permute groundtruth_test targets and rerun the whole
    ranker stage. Selection outcome and refit trajectory must be IDENTICAL;
    only final-test metrics may change."""
    tdir = chain["dir"]
    gt_path = tdir / "groundtruth_test.parquet"
    original = pd.read_parquet(gt_path)
    corrupted = original.copy()
    corrupted["parent_asin"] = list(reversed(corrupted["parent_asin"].tolist()))
    try:
        corrupted.to_parquet(gt_path, index=False)
        report2 = run_ranker(CAT, smoke=True, seed=7,
                             processed_dir=chain["root"],
                             results_dir=chain["root"] / "results_corrupt")
    finally:
        original.to_parquet(gt_path, index=False)
    r1, r2 = chain["ranker_report"], report2
    assert r1["selection"] == r2["selection"]
    assert r1["refit"]["log"] == r2["refit"]["log"]
    assert r1["refit"]["pos_weight"] == r2["refit"]["pos_weight"]


# ---- 12. missing categorical values handled ----------------------------------

def test_12_missing_categoricals_not_nan_string(chain):
    for snap in SNAPSHOT_NAMES:
        c = _load(chain, f"candidates_{snap}.parquet")
        assert "nan" not in set(c["store"]), snap
        assert "nan" not in set(c["main_category"]), snap
    # IP3's store was NaN in the catalog -> must surface as 'Unknown'.
    c_test = _load(chain, "candidates_test.parquet")
    ip3 = c_test[c_test["parent_asin"] == "IP3"]
    assert len(ip3) > 0 and (ip3["store"] == "Unknown").all()


# ---- 13. candidates invariant to catalog input row order ---------------------

def test_13_candidates_invariant_to_catalog_row_order(chain, tmp_path):
    catalog_rev = pd.read_parquet(
        chain["root"] / CAT / "item_features.parquet"
    ).iloc[::-1].reset_index(drop=True)
    out2 = _run_chain(tmp_path, _toy_clean(), catalog_rev,
                      reuse_tt_from=chain["root"])
    for snap in SNAPSHOT_NAMES:
        a = _load(chain, f"candidates_{snap}.parquet")
        b = pd.read_parquet(out2["dir"] / f"candidates_{snap}.parquet")
        pd.testing.assert_frame_equal(a, b)


# ---- 14. zero-history users never crash and stay in the denominator ----------

def test_14_zero_history_users_survive(chain):
    # UC had zero history at T0 yet has a ranker_train label + candidates.
    c_train = _load(chain, "candidates_ranker_train.parquet")
    uc = c_train[c_train["user_id"] == "UC"]
    assert len(uc) > 0
    assert (uc["user_has_history"] == 0).all()
    assert (uc["source_two_tower"] == 0).all()   # fallback = pop/rule only
    # Two-tower predictions exclude zero-history users, but metadata counts them.
    meta = json.loads((chain["dir"] / "two_tower_ranker_train_meta.json").read_text())
    assert meta["n_zero_history_label_users"] >= 4        # UC, Q1, Q2, Q3
    preds = _load(chain, "two_tower_predictions_ranker_train.parquet")
    assert "UC" not in set(preds["user_id"])
    # Final test denominator includes cold users (UE).
    ft = chain["ranker_report"]["final_test"]
    assert ft["history_cohorts"]["0"]["n_users"] == 1     # UE (zero history at T2)
    assert ft["n_groundtruth_users"] == 2                 # UA, UE


# ---- 15. manual targets + report sanity (popularity path, coverage) ----------

def test_15_manual_popularity_targets(chain):
    rep = chain["cand"]["snapshots"]["ranker_train"]
    # gt users: UA, UB (IP), UC, Q1..Q3 (IQ) = 6. IP in pool, IQ not.
    assert rep["n_positive_eval_users"] == 6
    assert rep["candidate_catalog_size"] == 4             # IP, IP2, IP3, IB1
    assert rep["coverage"]["eligible_pool"] == pytest.approx(2 / 6)
    assert rep["coverage"]["historical_catalog"] == pytest.approx(2 / 6)
    # Popularity ranks IP first (4 positives) -> UA, UB hit at rank 1.
    assert rep["sources"]["popularity"]["Recall@10"] == pytest.approx(2 / 6)
    assert rep["cold_users"] == 4                          # UC, Q1, Q2, Q3
    # Retrieved union covers exactly the two in-pool targets.
    assert rep["coverage"]["retrieved_union"] == pytest.approx(2 / 6)
    # Final ranker report: ceiling on test = fraction of test targets retrieved.
    ft = chain["ranker_report"]["final_test"]
    assert 0.0 <= ft["candidate_ceiling_retrieved_coverage"] <= 1.0
    cond = ft["conditional_given_retrieved"]
    if cond["n_users_retrieved"]:
        assert cond["Recall@100"] >= ft["overall"]["Recall@100"]


# ---- 15-17. wave 3 sequence arm (DIN at the ranking position) ---------------

def test_15_seq_arm_off_is_frozen_protocol(chain):
    """Default run never builds seq tensors and never reaches arch=din: the
    fixture report (seq_arm=False) selects from the frozen grid only."""
    import scripts.ranker.train_temporal_ranker as trt
    rep = chain["ranker_report"]
    archs = {r["arch"] for r in rep["selection"]["grid"]}
    assert not any(a.startswith("seq:") for a in archs)
    assert trt._SEQ is None


def test_16_seq_arm_din_runs_and_history_is_pre_cutoff(chain):
    import scripts.ranker.train_temporal_ranker as trt
    root = chain["root"]
    rep = run_ranker(CAT, smoke=True, seed=7, processed_dir=root,
                     results_dir=root / "results_seq", seq_arm=True)
    assert rep["selection"]["chosen"]["arch"].startswith("seq:")
    assert rep["final_test"]["overall"]["Recall@100"] >= 0.0
    ctx = trt._SEQ
    assert ctx is not None
    cut = chain["manifest"]["cutoffs"]
    hist_a = _load(chain, "history_ranker_train.parquet")
    hist_c = _load(chain, "history_test.parquet")
    # every item index in the ranker_train seq rows resolves to an item whose
    # history event is < T0; test rows < T2 (history parquet is the source)
    inv = {i: a for a, i in ctx.item_to_idx.items()}
    tab = ctx.table.numpy()
    for snap, hist, t in (("ranker_train", hist_a, cut["t0_ms"]),
                          ("test", hist_c, cut["t2_ms"])):
        assert int(hist["timestamp"].max()) < t
        for uid, ri in ctx.user_row[snap].items():
            row = tab[ri]
            items = {inv[int(i)] for i in row[:, 0] if int(i) > 1}   # 1 = <unk>
            seen = set(hist.loc[hist["user_id"].astype(str) == uid,
                                "parent_asin"].astype(str))
            assert items <= seen, (snap, uid, items - seen)
    trt._SEQ = None   # do not leak into later tests


def test_17_seq_vocab_frozen_on_ranker_train_history(chain):
    import scripts.ranker.train_temporal_ranker as trt
    ctx = trt._SeqContext(chain["dir"], ("ranker_train", "test"))
    hist_a = _load(chain, "history_ranker_train.parquet")
    hist_c = _load(chain, "history_test.parquet")
    hist_b = _load(chain, "history_model_selection.parquet")
    ctx2 = trt._SeqContext(chain["dir"], ("ranker_train", "test"),
                           vocab_snapshots=("ranker_train", "model_selection"))
    assert set(ctx2.item_to_idx) == (set(hist_a["parent_asin"].astype(str))
                                     | set(hist_b["parent_asin"].astype(str)))
    only_later = (set(hist_c["parent_asin"].astype(str))
                  - set(hist_a["parent_asin"].astype(str)))
    assert set(ctx.item_to_idx) == set(hist_a["parent_asin"].astype(str))
    # items first seen after T0 map to <unk>=1 (never pad 0, never a fresh index)
    if only_later:
        pa = next(iter(only_later))
        df = pd.DataFrame({"user_id": ["u"], "parent_asin": [pa]})
        assert int(ctx.tensors(df, "test")["seq__cand"][0, 0]) == 1
    assert min(ctx.item_to_idx.values()) == 2


def test_18_seq_variants_forward_shapes_and_empty_history_is_zero():
    """All 4 variants x 3 position encodings run; a user with no history
    yields a zero sequence vector so the head sees only DCN-side inputs."""
    import torch
    from scripts.ranker.ranker_features import RankerFeatureSpec
    from scripts.ranker.seq_ranker import SeqRanker, SeqConfig
    spec = RankerFeatureSpec(n_dense=4, dense_mean=np.zeros(4), dense_std=np.ones(4),
                             cat_vocabs={"main_category": {"<pad>": 0, "C": 1},
                                         "store": {"<pad>": 0, "S": 1}})
    B, L = 6, 20
    # history table: row 0 = no history, rows 1-2 = two users with history
    table = torch.zeros((3, L, 6), dtype=torch.int64)
    table[1, -3:] = torch.tensor([[3, 1, 1, 100, 2, 5], [5, 1, 1, 101, 4, 4], [2, 1, 1, 105, 7, 2]])
    table[2, -1] = torch.tensor([4, 1, 1, 120, 3, 5])
    uix = torch.tensor([1, 2, 0, 0, 0, 0], dtype=torch.int32)
    hist = table[uix.long()]
    cand = torch.tensor([[2, 1, 1]] * B, dtype=torch.int32)
    dense = torch.randn(B, 4)
    cats = {"cat__main_category": torch.ones(B, dtype=torch.int64),
            "cat__store": torch.ones(B, dtype=torch.int64)}
    for v in ("vanilla", "mh_pool", "causal", "hstu"):
        for pos in ("abs", "delta", "both"):
            cfg = SeqConfig.parse(f"seq:{v}:pos={pos}:d=16:L=2:H=2")
            m = SeqRanker(spec, n_items=10, n_side={"main_category": 3, "store": 3},
                          n_abs=400, n_delta=11, cfg=cfg, hist_table=table,
                          max_len=L).eval()
            with torch.no_grad():
                out = m(dense, seq__uix=uix, seq__cand=cand, **cats)
                x = m.encoder(hist); valid = hist[..., 0] > 0
                s_vec = m.seq(x, valid, m.encoder.item_vec(cand.long()),
                              abs_month=hist[..., 3])
            assert out.shape == (B,) and torch.isfinite(out).all(), (v, pos)
            assert torch.count_nonzero(s_vec[2:]) == 0, (v, pos, "empty history must be zero")
            assert torch.count_nonzero(s_vec[0]) > 0, (v, pos)


def test_19_seq_ranker_is_exact_dcn_at_alpha_zero():
    """Residual design: logit = DCN(x0) + alpha * g(cand, seq). With the DCN
    weights copied from a DeepCrossRanker and alpha=0 (its init), the seq
    ranker must reproduce the DCN logits exactly -> deep_cross is the exact
    ablation. Also: the rt-only vs rt+ms vocab contexts differ in size."""
    import torch
    from scripts.ranker.ranker_features import RankerFeatureSpec
    from scripts.ranker.seq_ranker import SeqRanker, SeqConfig
    from scripts.ranker.complex_ranker import DeepCrossRanker, DeepCrossConfig
    spec = RankerFeatureSpec(n_dense=4, dense_mean=np.zeros(4), dense_std=np.ones(4),
                             cat_vocabs={"main_category": {"<pad>": 0, "C": 1},
                                         "store": {"<pad>": 0, "S": 1}})
    B = 6
    table = torch.zeros((3, 20, 6), dtype=torch.int64)
    table[1, -2:] = torch.tensor([[3, 1, 1, 100, 2, 5], [2, 1, 1, 105, 7, 4]])
    torch.manual_seed(1)
    dcn = DeepCrossRanker(spec, DeepCrossConfig()).eval()
    m = SeqRanker(spec, 10, {"main_category": 3, "store": 3}, 400, 11,
                  SeqConfig.parse("seq:hstu:pos=both:ffn=4"), hist_table=table).eval()
    m.dcn.load_state_dict(dcn.state_dict())
    assert float(m.alpha) == 0.0
    dense = torch.randn(B, 4)
    cats = {"cat__main_category": torch.ones(B, dtype=torch.int64),
            "cat__store": torch.ones(B, dtype=torch.int64)}
    uix = torch.tensor([1, 1, 0, 2, 0, 1], dtype=torch.int32)
    cand = torch.tensor([[2, 1, 1]] * B, dtype=torch.int32)
    with torch.no_grad():
        a = dcn(dense, **cats)
        b = m(dense, seq__uix=uix, seq__cand=cand, **cats)
        assert torch.equal(a, b), "alpha=0 must reproduce the DCN exactly"
        m.alpha.fill_(1.0)
        c = m(dense, seq__uix=uix, seq__cand=cand, **cats)
        assert not torch.equal(a, c)              # residual path is live
    assert m.cfg.ffn_mult == 4


def test_20_all_positive_label_guard_refuses_first_positive_tables():
    """Regression for the rebuild-wave bug: a table whose label column came
    from the first-positive frame must be refused under all_positive."""
    from scripts.ranker.train_temporal_ranker import assert_all_positive_labels
    gt_all = pd.DataFrame({"user_id": ["u1", "u1", "u2"],
                           "parent_asin": ["A", "B", "C"]})
    cands_first = pd.DataFrame({"user_id": ["u1", "u1", "u1", "u2", "u2"],
                                "parent_asin": ["A", "B", "X", "C", "Y"],
                                "label": [1, 0, 0, 1, 0]})       # B labelled 0
    with pytest.raises(RuntimeError, match="first-positive frame"):
        assert_all_positive_labels(cands_first, gt_all)
    cands_all = cands_first.assign(label=[1, 1, 0, 1, 0])
    out = assert_all_positive_labels(cands_all, gt_all)
    assert out["future_positives_in_table"] == 3 and out["labelled"] == 3
    assert out["positives_per_user"] == pytest.approx(1.5)


def test_21_freeze_rule_guardrail_and_tiebreak():
    from scripts.analysis.p4_freeze_ap import decide
    a0 = {"R@10": 0.0438, "R@100": 0.1035}
    # Electronics-like: A5 beats A0 on R@100 but regresses R@10 -> A0 wins
    d = decide({"A0": a0, "A5": {"R@10": 0.0421, "R@100": 0.1038}}, 0.0005)
    assert d["winner"] == "A0" and d["table"][0]["guardrail_pass"] is False
    # VG-like: tiny R@100 gain, big R@10 gain -> A5 wins
    d = decide({"A0": a0, "A5": {"R@10": 0.0514, "R@100": 0.1036}}, 0.0005)
    assert d["winner"] == "A5"
    # tie on R@100 within tol -> broken by R@10
    d = decide({"A0": a0, "A2": {"R@10": 0.0450, "R@100": 0.1100},
                "A3": {"R@10": 0.0440, "R@100": 0.1102}}, 0.0005)
    assert d["winner"] == "A2" and "tie" in d["tie_note"]
    # tie on R@100 AND R@10 -> fewer candidate rows wins
    d = decide({"A0": a0, "A2": {"R@10": 0.0450, "R@100": 0.1100, "n_rows": 100},
                "A3": {"R@10": 0.0450, "R@100": 0.1101, "n_rows": 90}}, 0.0005)
    assert d["winner"] == "A3"
    # no config beats A0 -> A0
    d = decide({"A0": a0, "A1": {"R@10": 0.0500, "R@100": 0.1000}}, 0.0005)
    assert d["winner"] == "A0"


def test_22_frozen_ranker_trains_exact_epochs_no_reselection(chain):
    """--frozen-ranker must train EXACTLY the frozen epoch count with no
    best-epoch argmax/reload (review 2026-08-26): grid has one entry, its
    best_epoch equals the frozen epoch, and the run is marked frozen."""
    root = chain["root"]
    rep = run_ranker(CAT, smoke=True, seed=7, processed_dir=root,
                     results_dir=root / "results_frozen", stage_b_only=True,
                     frozen_ranker={"arch": "mlp", "lr": 1e-3, "epoch": 2})
    grid = rep["selection"]["grid"]
    assert len(grid) == 1
    assert grid[0]["frozen"] is True
    assert grid[0]["best_epoch"] == 2 and grid[0]["n_epochs_run"] == 2
    assert rep["selection"]["chosen"]["arch"] == "mlp"
