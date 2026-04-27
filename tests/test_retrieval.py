"""Phase 1 retrieval module unit tests (synthetic, no disk I/O)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.retrieval.evaluator import (
    build_groundtruth,
    build_split_report,
    coverage_of_pool,
    recall_precision_at_k,
)
from scripts.retrieval.popularity import (
    compute_popularity,
    recommend_popularity,
    user_seen_from_train,
)
from scripts.retrieval.random_baseline import recommend_random
from scripts.retrieval.rule_based import (
    build_user_facet_affinity,
    normalized_popularity_prior,
    recommend_rule_based,
    recommend_weighted_facets,
    tune_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_df() -> pd.DataFrame:
    # 4 users, 5 items. I1 is super popular, I5 is rare. U1 hates I5.
    rows = [
        ("U1", "I1", 5, 1, "positive"),
        ("U1", "I2", 4, 1, "positive"),
        ("U1", "I5", 1, 0, "hard_negative"),     # train-seen, hard-neg
        ("U2", "I1", 5, 1, "positive"),
        ("U2", "I3", 4, 1, "positive"),
        ("U3", "I1", 5, 1, "positive"),
        ("U3", "I2", 5, 1, "positive"),
        ("U3", "I4", 5, 1, "positive"),
        ("U4", "I1", 4, 1, "positive"),
    ]
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "label", "label_type"])


@pytest.fixture
def item_features() -> pd.DataFrame:
    # 5 items, 2 stores, 2 categories. I5 is in pool but not positive-popular.
    return pd.DataFrame({
        "parent_asin": ["I1", "I2", "I3", "I4", "I5"],
        "store": ["S1", "S1", "S2", "S2", "S3"],
        "main_category": ["A", "A", "B", "B", "C"],
        "in_train_catalog": [1, 1, 1, 1, 1],
        "in_train_positive_catalog": [1, 1, 1, 1, 0],
        "in_filtered_universe": [1, 1, 1, 1, 1],
    })


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------

def test_build_groundtruth_only_positives():
    val = pd.DataFrame({
        "user_id": ["A", "A", "B", "C"],
        "parent_asin": ["x", "y", "z", "w"],
        "label": [1, 0, 1, 0],
    })
    gt = build_groundtruth(val)
    assert gt == {"A": {"x"}, "B": {"z"}}              # C dropped: only neg


def test_recall_precision_at_k_basic():
    topk = {"A": ["x", "y", "z"], "B": ["a", "b", "c"]}
    gt = {"A": {"x", "y"}, "B": {"d"}}
    r, p, n = recall_precision_at_k(topk, gt, k=2)
    # A: top-2 = [x,y], hits=2, recall=2/2=1, prec=2/2=1
    # B: top-2 = [a,b], hits=0, recall=0, prec=0
    # mean recall = 0.5, mean prec = 0.5, n=2
    assert r == pytest.approx(0.5)
    assert p == pytest.approx(0.5)
    assert n == 2


def test_recall_skips_users_with_empty_gt():
    topk = {"A": ["x"], "B": ["y"]}
    gt = {"A": {"x"}}                                  # B has no positives -> skipped
    r, _, n = recall_precision_at_k(topk, gt, k=1)
    assert r == 1.0 and n == 1


def test_coverage_of_pool_row_level():
    gt = {"A": {"x", "y"}, "B": {"z"}}
    pool = {"x", "z", "irrelevant"}
    # 3 total positive (user, item) rows; 2 in pool -> coverage 2/3
    assert coverage_of_pool(gt, pool) == pytest.approx(2 / 3)


def test_build_split_report_packs_all_required_fields():
    topk = {"A": ["x", "y"], "B": ["z"], "C": ["q"]}
    gt = {"A": {"x"}, "B": {"q"}}                      # C empty
    pool = {"x", "y", "z", "q"}
    rep = build_split_report("val", topk, gt, pool, "in_train_catalog", ks=(1, 2))
    d = rep.to_dict()
    assert d["split"] == "val"
    assert d["candidate_pool_type"] == "in_train_catalog"
    assert d["candidate_pool_size"] == 4
    assert d["n_total_eval_users"] == 3
    assert d["n_positive_eval_users"] == 2
    assert d["positive_eval_user_rate"] == pytest.approx(2 / 3)
    assert "Recall@1" in d["metrics"] and "Precision@2" in d["metrics"]
    assert d["heldout_positive_coverage_by_candidate_pool"] == 1.0


# ---------------------------------------------------------------------------
# popularity
# ---------------------------------------------------------------------------

def test_compute_popularity_train_positives_only(train_df):
    pop = compute_popularity(train_df)
    # U1's I5 is hard_negative (label=0), so I5 is NOT counted -> absent here.
    assert "I5" not in pop.index
    # I1 is in 4 rows all positive -> highest count.
    assert pop.iloc[0] == 4 and pop.index[0] == "I1"


def test_recommend_popularity_excludes_train_seen(train_df, item_features):
    pop = compute_popularity(train_df)
    pool = set(item_features["parent_asin"])
    seen = user_seen_from_train(train_df)
    topk = recommend_popularity(pop, pool, ["U1", "U2"], seen, k=10)
    # U1 saw I1, I2, I5 -> should NOT appear in U1's recs.
    assert "I1" not in topk["U1"]
    assert "I2" not in topk["U1"]
    assert "I5" not in topk["U1"]
    # U2 saw I1, I3 -> should NOT appear.
    assert "I1" not in topk["U2"]
    assert "I3" not in topk["U2"]


def test_recommend_popularity_includes_zero_count_pool_items(train_df, item_features):
    """Fix B: items in candidate_pool with zero positive train count
    must still appear in the ranking (at the end), not be silently dropped."""
    pop = compute_popularity(train_df)
    pool = set(item_features["parent_asin"])                 # includes I5 (zero pos)
    seen = user_seen_from_train(train_df)
    # U4 has only seen I1; remaining pool of unseen items = {I2, I3, I4, I5}.
    # Old behavior dropped I5; new behavior keeps it (with score 0) at the end.
    topk = recommend_popularity(pop, pool, ["U4"], seen, k=10)
    assert "I5" in topk["U4"]                                # would be missing pre-fix
    assert topk["U4"][-1] == "I5"                            # zero-count tail


def test_recommend_popularity_deterministic_alphabetical_tiebreak(train_df, item_features):
    """When two pool items tie on popularity count, ranking is alphabetical
    so the report is reproducible. I3 and I4 both have 1 positive train row;
    U4 hasn't seen either, so they appear in alphabetical order."""
    pop = compute_popularity(train_df)
    pool = set(item_features["parent_asin"])
    seen = user_seen_from_train(train_df)
    topk = recommend_popularity(pop, pool, ["U4"], seen, k=10)
    # I3 should come before I4 by alphabetical order within their tied count.
    assert topk["U4"].index("I3") < topk["U4"].index("I4")


# ---------------------------------------------------------------------------
# random_baseline
# ---------------------------------------------------------------------------

def test_random_baseline_excludes_seen_and_is_deterministic(train_df, item_features):
    pool = set(item_features["parent_asin"])
    seen = user_seen_from_train(train_df)
    a = recommend_random(pool, ["U1", "U2", "U3"], seen, k=10, seed=123)
    b = recommend_random(pool, ["U1", "U2", "U3"], seen, k=10, seed=123)
    assert a == b                                            # deterministic for same seed
    # No seen items leak in.
    for u, recs in a.items():
        assert seen[u].isdisjoint(set(recs)), f"{u} got seen items: {set(recs) & seen[u]}"
    # Different seeds produce different orderings (overwhelmingly likely).
    # Use U4 -- only 1 train-seen item -> 4 unseen -> 24 permutations.
    a4 = recommend_random(pool, ["U4"], seen, k=10, seed=123)
    c4 = recommend_random(pool, ["U4"], seen, k=10, seed=999)
    assert a4["U4"] != c4["U4"]


# ---------------------------------------------------------------------------
# rule_based
# ---------------------------------------------------------------------------

def test_user_facet_affinity_train_positive_only(train_df, item_features):
    aff = build_user_facet_affinity(train_df, item_features, "store")
    # U1: train-positive on I1, I2 (both store S1). Hard-neg I5 (store S3) excluded.
    assert "S1" in aff["U1"] and aff["U1"]["S1"] == 1.0
    assert "S3" not in aff["U1"]
    # U3: I1, I2 in S1; I4 in S2. S1 count=2, S2 count=1 -> normalized 1.0, 0.5.
    assert aff["U3"]["S1"] == 1.0 and aff["U3"]["S2"] == pytest.approx(0.5)


def test_normalized_popularity_prior_aligns_to_index(train_df, item_features):
    items = item_features["parent_asin"].to_numpy()
    prior = normalized_popularity_prior(train_df, items)
    assert len(prior) == len(items)
    # I1 has count 4 (max) -> prior 1.0. I5 has 0 -> prior 0.
    assert prior[0] == pytest.approx(1.0)              # I1
    assert prior[4] == 0.0                              # I5


def test_recommend_rule_based_seen_masking_and_scoring(train_df, item_features):
    items = item_features["parent_asin"].to_numpy()
    aff_store = build_user_facet_affinity(train_df, item_features, "store")
    aff_cat = build_user_facet_affinity(train_df, item_features, "main_category")
    prior = normalized_popularity_prior(train_df, items)
    seen = user_seen_from_train(train_df)
    topk = recommend_rule_based(
        user_ids=["U1"],
        candidate_items=items,
        item_store_arr=item_features["store"].to_numpy(),
        item_cat_arr=item_features["main_category"].to_numpy(),
        item_pop_prior=prior,
        user_store_aff=aff_store,
        user_cat_aff=aff_cat,
        user_seen=seen,
        weights=(1.0, 1.0, 0.5),
        k=2,
    )
    # U1 has seen I1, I2, I5; possible candidates are I3, I4. Both share store S2;
    # U1 has no S2 affinity, so both rely on category + popularity prior.
    # I3 and I4 should be the only items returned.
    assert set(topk["U1"]) <= {"I3", "I4"}
    assert "I1" not in topk["U1"] and "I2" not in topk["U1"]


def test_recommend_weighted_facets_matches_rule_based_on_2_facets(train_df, item_features):
    """The generic scorer must agree with the legacy 2-facet `recommend_rule_based`
    when given the same store + main_category + pop weights."""
    items = item_features["parent_asin"].to_numpy()
    aff_store = build_user_facet_affinity(train_df, item_features, "store")
    aff_cat = build_user_facet_affinity(train_df, item_features, "main_category")
    prior = normalized_popularity_prior(train_df, items)
    seen = user_seen_from_train(train_df)

    legacy = recommend_rule_based(
        user_ids=["U1", "U2", "U3", "U4"],
        candidate_items=items,
        item_store_arr=item_features["store"].to_numpy(),
        item_cat_arr=item_features["main_category"].to_numpy(),
        item_pop_prior=prior,
        user_store_aff=aff_store,
        user_cat_aff=aff_cat,
        user_seen=seen,
        weights=(1.0, 0.5, 0.5),
        k=5,
    )
    generic = recommend_weighted_facets(
        user_ids=["U1", "U2", "U3", "U4"],
        candidate_items=items,
        facet_arrays={
            "store": item_features["store"].to_numpy(),
            "main_category": item_features["main_category"].to_numpy(),
        },
        user_affinities={"store": aff_store, "main_category": aff_cat},
        facet_weights={"store": 1.0, "main_category": 0.5},
        item_pop_prior=prior,
        pop_weight=0.5,
        user_seen=seen,
        k=5,
    )
    assert legacy == generic


def test_recommend_weighted_facets_drops_zero_weight_facets(train_df, item_features):
    """Facets with weight 0 should not influence the score (and the impl skips
    them in the inner loop). Result equals popularity-only when both facet
    weights are zero."""
    items = item_features["parent_asin"].to_numpy()
    aff_store = build_user_facet_affinity(train_df, item_features, "store")
    prior = normalized_popularity_prior(train_df, items)
    seen = user_seen_from_train(train_df)
    only_pop = recommend_weighted_facets(
        user_ids=["U4"],
        candidate_items=items,
        facet_arrays={"store": item_features["store"].to_numpy()},
        user_affinities={"store": aff_store},
        facet_weights={"store": 0.0},
        item_pop_prior=prior,
        pop_weight=1.0,
        user_seen=seen,
        k=5,
    )
    # U4 has only seen I1; remaining items ranked purely by popularity.
    # Popularity counts: I2=2, I3=1, I4=1, I5=0; alphabetical tie break inside
    # the generic scorer is by argpartition, not deterministic, so we only
    # assert the most popular unseen item leads.
    assert only_pop["U4"][0] == "I2"


def test_tune_weights_picks_a_winner(train_df, item_features):
    val = pd.DataFrame({
        "user_id": ["U1", "U1", "U2"],
        "parent_asin": ["I3", "I4", "I4"],
        "label": [1, 1, 1],
        "rating": [5, 5, 5],
        "timestamp": [10, 11, 12],
    })
    items = item_features["parent_asin"].to_numpy()
    aff_store = build_user_facet_affinity(train_df, item_features, "store")
    aff_cat = build_user_facet_affinity(train_df, item_features, "main_category")
    prior = normalized_popularity_prior(train_df, items)
    seen = user_seen_from_train(train_df)
    res = tune_weights(
        val_df=val,
        user_ids_for_tuning=["U1", "U2"],
        candidate_items=items,
        item_store_arr=item_features["store"].to_numpy(),
        item_cat_arr=item_features["main_category"].to_numpy(),
        item_pop_prior=prior,
        user_store_aff=aff_store,
        user_cat_aff=aff_cat,
        user_seen=seen,
        grid=[(1.0, 1.0, 0.5), (0.0, 0.0, 1.0)],
        k_for_tuning=5,
    )
    assert res.best_weights in {(1.0, 1.0, 0.5), (0.0, 0.0, 1.0)}
    assert 0.0 <= res.best_recall <= 1.0
    assert len(res.grid) == 2
