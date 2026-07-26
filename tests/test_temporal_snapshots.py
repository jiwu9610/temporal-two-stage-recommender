"""Global temporal snapshot invariants (mode: global_temporal_snapshots).

All synthetic, no real data/processed dependency. Verifies the spec contract:

  1. validation history contains only past events (ts < train_end)
  2. test history contains validation-window events (superset of val history)
  3. future events never affect validation features
  4. validation/test candidate pools are generated independently
  5. seen-item masks change between snapshots
  6. no temporal leakage: labels in-window, first-positive, popularity blind
     to the future
  7. old pipeline (leave_last_two_split) still works, and the temporal
     orchestrator never touches Phase 0 artifacts
  8. synthetic end-to-end: pipeline -> feature stores -> retrieval -> metrics

Run from repo root (conda env tae):
    pytest tests/test_temporal_snapshots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.splitting import leave_last_two_split  # noqa: E402
from scripts.data.temporal_feature_store import (  # noqa: E402
    ITEM_SNAPSHOT_FEATURE_COLUMNS,
    USER_SNAPSHOT_FEATURE_COLUMNS,
    build_snapshot_item_features,
    build_snapshot_user_features,
)
from scripts.data.temporal_split import (  # noqa: E402
    MS_PER_DAY,
    CutoffSpec,
    build_snapshot,
    build_temporal_snapshots,
    first_positive_labels,
    resolve_cutoffs,
)
from scripts.retrieval.evaluator import build_groundtruth  # noqa: E402
from scripts.retrieval.popularity import (  # noqa: E402
    compute_popularity,
    recommend_popularity,
    user_seen_from_train,
)


def day(n: float) -> int:
    """Synthetic epoch-ms timestamp: day n of the toy timeline."""
    return int(n * MS_PER_DAY)


# Cutoffs shared by every fixture: train_end=day 50, val_end=day 70, test_end=day 100.
CUTOFFS = CutoffSpec(
    train_end_ms=day(50), val_end_ms=day(70), test_end_ms=day(100),
    source="explicit",
)


def _mk(rows):
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    df["label"] = (df["rating"] >= 3).astype("int8")
    df["label_type"] = np.where(df["label"] == 1, "positive", "hard_negative")
    return df[["user_id", "parent_asin", "rating", "label", "label_type", "timestamp"]]


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Toy universe crafted so every invariant has a witness.

    Items: IPOP gets 5 pre-day-50 positives (eligible in BOTH snapshots).
           INEW gets 5 positives inside [50, 70) (eligible ONLY in test snapshot).
           IRARE has 1 history event (never eligible).
           IFUT only exists after day 50 (invisible to val snapshot entirely).
    Users: UA warm, positive val label on IPOP + test label on INEW.
           UB warm, val-window event only (becomes test-history, not val-history).
           UC cold at val time (zero history; still cold at test time: 2 < 3).
           UD hard-negative-only in val window -> no val label.
           UE cold forever: single positive in [70,100) -> cold test user.
           N1-N3 / F1 / F2: zero-history users whose in-window positives make
           them cold label users too (nothing is silently dropped).
    """
    rows = []
    # 5 distinct users give IPOP 5 positive history events before day 50.
    for i, u in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        rows.append((u, "IPOP", 5, day(10 + i)))
    # A second pre-cutoff event per P-user; P1's lands on IRARE so IRARE has
    # exactly one history event (below the pool threshold).
    for i, u in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        rows.append((u, "IRARE" if i == 0 else f"IB{i}", 4, day(20 + i)))

    # UA: 3 history events (warm), then val label IPOP at day 55, test label INEW at day 75.
    rows += [
        ("UA", "IA1", 5, day(5)),
        ("UA", "IA2", 4, day(15)),
        ("UA", "IA3", 5, day(30)),
        ("UA", "IPOP", 5, day(55)),
        ("UA", "INEW", 4, day(75)),
    ]
    # UB: 3 history events + a val-window positive (INEW) at day 60; no test-window event.
    rows += [
        ("UB", "IB1", 5, day(8)),
        ("UB", "IB2", 3, day(18)),
        ("UB", "IB3", 4, day(28)),
        ("UB", "INEW", 5, day(60)),
    ]
    # INEW eligibility in test snapshot: 5 positives inside [50, 70).
    for i, u in enumerate(["N1", "N2", "N3"]):
        rows.append((u, "INEW", 5, day(61 + i)))
    # (UA day-75 event is OUTSIDE test history; UB day-60 + N1..N3 + UC give INEW 5 by day 70)

    # UC: cold at val time -- first ever event day 52 (positive, INEW), another day 65.
    rows += [
        ("UC", "INEW", 5, day(52)),
        ("UC", "IPOP", 4, day(65)),
        ("UC", "IC1", 5, day(80)),   # test-window label for UC
    ]
    # UD: warm history, but ONLY a hard negative in the val window.
    rows += [
        ("UD", "ID1", 5, day(9)),
        ("UD", "ID2", 4, day(19)),
        ("UD", "ID3", 5, day(29)),
        ("UD", "IPOP", 1, day(56)),   # hard negative -> no val label
    ]
    # UE: single positive in the test window -> cold test user.
    rows += [("UE", "IPOP", 5, day(85))]
    # IFUT: exists only after train_end.
    rows += [("F1", "IFUT", 5, day(58)), ("F2", "IFUT", 4, day(72))]
    return _mk(rows)


@pytest.fixture
def snapshots(clean_df):
    return build_temporal_snapshots(
        clean_df, CUTOFFS,
        min_item_history_interactions=5, warm_user_min_history=3,
    )


# ---------------------------------------------------------------------------
# 1-2. history boundaries
# ---------------------------------------------------------------------------

def test_val_history_only_past_events(snapshots):
    val, test, _ = snapshots
    assert (val.history["timestamp"] < CUTOFFS.train_end_ms).all()
    assert (val.labels["timestamp"] >= CUTOFFS.train_end_ms).all()
    assert (val.labels["timestamp"] < CUTOFFS.val_end_ms).all()


def test_test_history_contains_val_window_events(snapshots, clean_df):
    val, test, _ = snapshots
    assert (test.history["timestamp"] < CUTOFFS.val_end_ms).all()
    # test history = val history UNION val-window events, as row sets.
    key = ["user_id", "parent_asin", "timestamp"]
    val_window = clean_df[
        (clean_df["timestamp"] >= CUTOFFS.train_end_ms)
        & (clean_df["timestamp"] < CUTOFFS.val_end_ms)
    ]
    expected = set(map(tuple, pd.concat([val.history, val_window])[key].to_numpy()))
    actual = set(map(tuple, test.history[key].to_numpy()))
    assert actual == expected
    # UB's val-window INEW row is a concrete witness.
    ub = test.history.query("user_id == 'UB' and parent_asin == 'INEW'")
    assert len(ub) == 1
    assert val.history.query("user_id == 'UB' and parent_asin == 'INEW'").empty


# ---------------------------------------------------------------------------
# 3. future events never affect validation features
# ---------------------------------------------------------------------------

def test_future_events_never_affect_val_features(clean_df):
    val, _, _ = build_temporal_snapshots(clean_df, CUTOFFS, 5, 3)
    feats_before = build_snapshot_user_features(val.history, CUTOFFS.train_end_ms)

    # Inject a flood of future events (both windows) and rebuild everything.
    extra = _mk([
        ("UA", f"IX{i}", 5, day(51 + (i % 40))) for i in range(200)
    ])
    noisy = pd.concat([clean_df, extra], ignore_index=True)
    val2, _, _ = build_temporal_snapshots(noisy, CUTOFFS, 5, 3)
    feats_after = build_snapshot_user_features(val2.history, CUTOFFS.train_end_ms)

    pd.testing.assert_frame_equal(
        feats_before.sort_values("user_id").reset_index(drop=True),
        feats_after.sort_values("user_id").reset_index(drop=True),
    )
    # Same for the eligible pool and popularity.
    assert set(val.eligible_items) == set(val2.eligible_items)
    pd.testing.assert_series_equal(
        compute_popularity(val.history), compute_popularity(val2.history)
    )


def test_feature_store_rejects_future_rows(clean_df):
    _, test, _ = build_temporal_snapshots(clean_df, CUTOFFS, 5, 3)
    with pytest.raises(ValueError, match="leakage"):
        build_snapshot_user_features(test.history, CUTOFFS.train_end_ms)
    with pytest.raises(ValueError, match="leakage"):
        build_snapshot_item_features(
            pd.DataFrame({"parent_asin": ["IPOP"]}),
            test.history, pd.Index([]), CUTOFFS.train_end_ms,
        )


# ---------------------------------------------------------------------------
# 4. candidate pools generated independently
# ---------------------------------------------------------------------------

def test_candidate_pools_independent(snapshots):
    val, test, _ = snapshots
    # INEW reaches 5 interactions only via val-window events: test-only eligible.
    assert "INEW" not in set(val.eligible_items)
    assert "INEW" in set(test.eligible_items)
    assert "IPOP" in set(val.eligible_items)
    assert "IPOP" in set(test.eligible_items)
    assert "IRARE" not in set(val.eligible_items)
    assert "IFUT" not in set(val.eligible_items)


# ---------------------------------------------------------------------------
# 5. seen-item masks change between snapshots
# ---------------------------------------------------------------------------

def test_seen_masks_change_between_snapshots(snapshots):
    val, test, _ = snapshots
    seen_val = user_seen_from_train(val.history)
    seen_test = user_seen_from_train(test.history)
    # UA's val-window IPOP purchase is unseen at val time, seen at test time.
    assert "IPOP" not in seen_val.get("UA", set())
    assert "IPOP" in seen_test["UA"]
    # Masks only grow: every val-seen item stays test-seen.
    for u, items in seen_val.items():
        assert items <= seen_test.get(u, set())


# ---------------------------------------------------------------------------
# 6. no temporal leakage in labels / popularity
# ---------------------------------------------------------------------------

def test_labels_are_first_positive_in_window(snapshots):
    val, test, _ = snapshots
    # One row per user, label==1, inside the window.
    for snap, lo, hi in [(val, CUTOFFS.train_end_ms, CUTOFFS.val_end_ms),
                         (test, CUTOFFS.val_end_ms, CUTOFFS.test_end_ms)]:
        assert snap.labels["user_id"].is_unique
        assert (snap.labels["label"] == 1).all()
        assert (snap.labels["timestamp"] >= lo).all()
        assert (snap.labels["timestamp"] < hi).all()
    # UC has two val-window positives (day 52 INEW, day 65 IPOP): first wins.
    uc = val.labels.query("user_id == 'UC'").iloc[0]
    assert uc["parent_asin"] == "INEW" and uc["timestamp"] == day(52)
    # UD's val-window event is a hard negative -> no val label row.
    assert "UD" not in set(val.labels["user_id"])


def test_warm_cold_flags_and_no_silent_drops(snapshots):
    val, test, _ = snapshots
    # UC is cold at val time (zero history) but still present with a label.
    uc = val.labels.query("user_id == 'UC'").iloc[0]
    assert uc["n_history_events"] == 0 and uc["is_warm_user"] == 0
    # UA is warm at val time.
    ua = val.labels.query("user_id == 'UA'").iloc[0]
    assert ua["n_history_events"] == 3 and ua["is_warm_user"] == 1
    # UE: single test-window positive, zero test history -> cold, present.
    ue = test.labels.query("user_id == 'UE'").iloc[0]
    assert ue["is_warm_user"] == 0
    # Manifest counts match the flags.
    assert val.stats["labels"]["n_cold_users"] == int(
        (val.labels["is_warm_user"] == 0).sum()
    )


def test_popularity_is_blind_to_future(snapshots):
    val, test, _ = snapshots
    pop_val = compute_popularity(val.history)
    # INEW/IFUT have zero pre-train_end events: absent from val popularity.
    assert "INEW" not in pop_val.index
    assert "IFUT" not in pop_val.index
    # In the test snapshot, INEW's val-window positives are (correctly) counted.
    pop_test = compute_popularity(test.history)
    assert pop_test["INEW"] == 5


def test_first_positive_tie_broken_deterministically():
    window = _mk([
        ("U", "IB", 5, day(55)),
        ("U", "IA", 5, day(55)),   # same ms -> alphabetical tiebreak
    ])
    lab = first_positive_labels(window)
    assert lab.iloc[0]["parent_asin"] == "IA"


# ---------------------------------------------------------------------------
# temporal feature values
# ---------------------------------------------------------------------------

def test_basic_temporal_feature_values():
    hist = _mk([
        ("U1", "I1", 5, day(10)),
        ("U1", "I2", 1, day(44)),   # hard negative inside last-30d window
        ("U1", "I3", 4, day(48)),
        ("U2", "I1", 5, day(2)),
    ])
    feats = build_snapshot_user_features(hist, as_of_ms=day(50), recent_days=30)
    u1 = feats.set_index("user_id").loc["U1"]
    assert u1["days_since_last_interaction"] == pytest.approx(2.0)
    assert u1["interactions_last_30d"] == 2          # days 44, 48 (>= day 20)
    assert u1["positives_last_30d"] == 1             # day 48 only
    assert u1["n_reviews_hist"] == 3
    assert u1["active_days_hist"] == pytest.approx(38.0)
    u2 = feats.set_index("user_id").loc["U2"]
    assert u2["days_since_last_interaction"] == pytest.approx(48.0)
    assert u2["interactions_last_30d"] == 0
    assert list(feats.columns) == USER_SNAPSHOT_FEATURE_COLUMNS


def test_item_snapshot_features_exclude_dump_aggregates(snapshots):
    val, _, _ = snapshots
    catalog = pd.DataFrame({
        "parent_asin": ["IPOP", "INEW", "IRARE", "IFUT"],
        "main_category": ["C", "C", "C", "C"],
        "store": ["S", "S", "S", "S"],
        "price": [1.0, 2.0, 3.0, 4.0],
        # Leaky dump columns that must NOT survive into snapshot features:
        "average_rating": [4.9, 4.8, 4.7, 4.6],
        "rating_number": [1000, 500, 10, 5],
    })
    items = build_snapshot_item_features(
        catalog, val.history, val.eligible_items, CUTOFFS.train_end_ms
    )
    assert "average_rating" not in items.columns
    assert "rating_number" not in items.columns
    assert list(items.columns) == ITEM_SNAPSHOT_FEATURE_COLUMNS
    row = items.set_index("parent_asin")
    assert row.loc["IPOP", "n_positives_hist"] == 5
    assert row.loc["IPOP", "in_eligible_pool"] == 1
    assert row.loc["INEW", "n_reviews_hist"] == 0
    assert row.loc["INEW", "in_history_catalog"] == 0
    assert row.loc["IFUT", "in_eligible_pool"] == 0


# ---------------------------------------------------------------------------
# cutoff resolution
# ---------------------------------------------------------------------------

def test_resolve_cutoffs_explicit_and_quantile(clean_df):
    spec = resolve_cutoffs(clean_df, explicit={
        "train_end": "1970-02-20", "val_end": "1970-03-12",
    })
    assert spec.source == "explicit"
    assert spec.train_end_ms == day(50)
    assert spec.val_end_ms == day(70)
    assert spec.test_end_ms == int(clean_df["timestamp"].max()) + 1

    spec_q = resolve_cutoffs(clean_df)
    assert spec_q.source == "quantile"
    assert spec_q.train_end_ms < spec_q.val_end_ms <= spec_q.test_end_ms

    with pytest.raises(ValueError, match="cutoffs"):
        CutoffSpec(day(70), day(50), day(100), source="explicit")


# ---------------------------------------------------------------------------
# 7. old pipeline still works + no Phase 0 artifacts touched
# ---------------------------------------------------------------------------

def test_leave_last_two_still_works():
    rows = []
    for ts, it in enumerate(["I1", "I2", "I3", "I4", "I5"], start=1):
        rows.append(("UA", it, 5, ts))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    df["label"] = 1
    train, val, test, manifest = leave_last_two_split(df)
    assert manifest.strategy == "per_user_leave_last_two"
    assert sorted(train["parent_asin"]) == ["I1", "I2", "I3"]
    assert val["parent_asin"].iloc[0] == "I4"
    assert test["parent_asin"].iloc[0] == "I5"


def test_temporal_pipeline_writes_only_under_temporal_dir(clean_df, tmp_path):
    from scripts.data.temporal_pipeline import (
        TEMPORAL_OUTPUT_FILES,
        run_category,
    )

    cat_dir = tmp_path / "SyntheticCat"
    cat_dir.mkdir(parents=True)
    clean_df.to_parquet(cat_dir / "interactions_clean.parquet", index=False)
    sentinel = cat_dir / "train.parquet"          # fake Phase 0 artifact
    sentinel.write_bytes(b"phase0-sentinel")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "temporal_snapshots:\n"
        "  cutoffs:\n"
        "    SyntheticCat: {train_end: '1970-02-20', val_end: '1970-03-12'}\n"
        "  eligibility:\n"
        "    min_item_history_interactions: 5\n"
        "    warm_user_min_history: 3\n"
    )
    manifest = run_category("SyntheticCat", config_path=cfg, processed_dir=tmp_path)

    assert sentinel.read_bytes() == b"phase0-sentinel"
    top_level = {p.name for p in cat_dir.iterdir()}
    assert top_level == {"interactions_clean.parquet", "train.parquet", "temporal"}
    written = {p.name for p in (cat_dir / "temporal").iterdir()}
    assert written == TEMPORAL_OUTPUT_FILES
    assert manifest["strategy"] == "global_temporal_snapshots"
    assert manifest["cutoffs"]["source"] == "explicit"


# ---------------------------------------------------------------------------
# 8. synthetic end-to-end: snapshots -> features -> retrieval -> metrics
# ---------------------------------------------------------------------------

def test_synthetic_end_to_end(clean_df, tmp_path):
    from scripts.data.temporal_pipeline import run_category as build_snapshots
    from scripts.retrieval.run_temporal_retrieval import (
        run_category as run_retrieval,
    )

    cat_dir = tmp_path / "SyntheticCat"
    cat_dir.mkdir(parents=True)
    clean_df.to_parquet(cat_dir / "interactions_clean.parquet", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "temporal_snapshots:\n"
        "  cutoffs:\n"
        "    SyntheticCat: {train_end: '1970-02-20', val_end: '1970-03-12'}\n"
    )
    build_snapshots("SyntheticCat", config_path=cfg, processed_dir=tmp_path)

    report = run_retrieval(
        "SyntheticCat",
        processed_dir=tmp_path,
        results_dir=tmp_path / "results",
        n_random_seeds=2,
    )

    val_payload = report["snapshots"]["val"]
    test_payload = report["snapshots"]["test"]

    # All label users evaluated, cold included, none silently dropped.
    # val labels: UA, UB (warm) + UC, N1, N2, N3, F1 (zero-history cold).
    assert val_payload["n_eval_users"] == 7
    assert val_payload["n_warm_users"] == 2
    assert val_payload["n_cold_users"] == 5
    # test labels: UA (warm) + UC (2 history events < 3), UE, F2 (cold).
    assert test_payload["n_eval_users"] == 4
    assert test_payload["n_warm_users"] == 1
    assert test_payload["n_cold_users"] == 3

    # Val pool is {IPOP} only; label items are IPOP x1 (UA) and INEW/IFUT x6.
    assert val_payload["candidate_pool"]["size"] == 1
    cov = val_payload["heldout_positive_coverage_by_candidate_pool"]
    assert cov == pytest.approx(1 / 7)

    # Popularity@K: UA's label IPOP is the only in-pool label and UA hasn't
    # seen IPOP at val time -> exactly 1 of 7 users hit -> recall 1/7.
    pop_val = val_payload["models"]["popularity"]["overall"]
    assert pop_val["Recall@10"] == pytest.approx(1 / 7)

    # Cold-item stats surface the out-of-pool labels (INEW x5, IFUT x1).
    assert val_payload["cold_item_stats"]["n_label_items_not_in_eligible_pool"] == 6
    assert val_payload["cold_item_stats"]["n_label_items_not_in_history"] == 6

    # Report is valid JSON on disk.
    on_disk = json.loads((tmp_path / "results" / "SyntheticCat.json").read_text())
    assert on_disk["protocol"] == "global_temporal_snapshots"

    # Test snapshot: pool = {IPOP, INEW}. UA's label INEW is unseen and in
    # pool (hit); UA's now-seen IPOP is masked. UE's label IPOP is in pool
    # (cold hit). UC/F2 labels are out of pool. -> overall 2/4, warm 1/1,
    # cold 1/3.
    assert test_payload["candidate_pool"]["size"] == 2
    pop_test = test_payload["models"]["popularity"]
    assert pop_test["overall"]["Recall@10"] == pytest.approx(2 / 4)
    assert pop_test["warm"]["Recall@10"] == pytest.approx(1.0)
    assert pop_test["cold"]["Recall@10"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# tie-break leakage regression: metrics must be invariant to catalog row order
# ---------------------------------------------------------------------------

def test_report_invariant_to_catalog_row_order(clean_df, tmp_path):
    """The Phase 0 catalog is sorted by descending 2023 rating_number (a
    whole-timeline statistic). If any scorer inherited that row order for
    positional tie-breaking, metrics would depend on future information.
    Running the full pipeline with two adversarial catalog orderings must
    produce identical reports."""
    from scripts.data.temporal_pipeline import run_category as build_snapshots
    from scripts.retrieval.run_temporal_retrieval import (
        run_category as run_retrieval,
    )

    all_items = sorted(clean_df["parent_asin"].unique())

    def _run(tag, item_order):
        cat_dir = tmp_path / tag / "SyntheticCat"
        cat_dir.mkdir(parents=True)
        clean_df.to_parquet(cat_dir / "interactions_clean.parquet", index=False)
        catalog = pd.DataFrame({
            "parent_asin": item_order,
            "main_category": "C",
            "store": [f"S{i % 3}" for i in range(len(item_order))],
            "price": 1.0,
        })
        # Store facets must describe the ITEM, not its position, or the two
        # runs legitimately differ; map store by item id.
        catalog["store"] = ["S" + str(hash(a) % 3) for a in catalog["parent_asin"]]
        catalog.to_parquet(cat_dir / "item_features.parquet", index=False)
        cfg = tmp_path / tag / "cfg.yaml"
        cfg.write_text(
            "temporal_snapshots:\n"
            "  cutoffs:\n"
            "    SyntheticCat: {train_end: '1970-02-20', val_end: '1970-03-12'}\n"
        )
        build_snapshots("SyntheticCat", config_path=cfg,
                        processed_dir=tmp_path / tag)
        report = run_retrieval(
            "SyntheticCat",
            processed_dir=tmp_path / tag,
            results_dir=tmp_path / tag / "results",
            n_random_seeds=1,
        )
        report.pop("started_utc")
        report.pop("elapsed_seconds")
        return json.dumps(report, sort_keys=True)

    # Ordering A: "future popularity" descending analog; B: reversed.
    rep_a = _run("orderA", all_items)
    rep_b = _run("orderB", list(reversed(all_items)))
    assert rep_a == rep_b
