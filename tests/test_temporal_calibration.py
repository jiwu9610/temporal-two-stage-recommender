"""Tests for the Evaluation & Calibration layer.

Covers:
  - grouped_eval.evaluate_flat against a naive pure-Python reference on
    random tie-free jagged data (AUC/gAUC/MRR/nDCG/P/R), batch invariance,
    monotone-transform invariance, denominator semantics, tie policy
  - cross-implementation check: recall@k_overall vs the legacy
    _scores_to_topk + recall_precision_at_k path
  - ECE tables (equal-width + quantile), sigmoid stability, Platt recovery
    of a known logistic transform, threshold layer bucket routing
  - test-lock: fit never opens the test dump; shuffled test labels leave the
    frozen config byte-identical; test stage refuses to run without frozen
  - prediction dump frame schema + length guard, CLI results-dir guard
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.evaluation.grouped_eval import evaluate_flat
from scripts.evaluation import temporal_calibration as tc
from scripts.ranker.train_temporal_ranker import _dump_prediction_frame


# ---------------------------------------------------------------------------
# Naive reference implementations (pure Python, no torch)
# ---------------------------------------------------------------------------

def _naive_user_metrics(rows, ks):
    """rows: list of (score, label) for ONE user, tie-free scores."""
    ranked = sorted(rows, key=lambda t: -t[0])
    labels_ranked = [l for _, l in ranked]
    n_pos = sum(labels_ranked)
    n_neg = len(labels_ranked) - n_pos
    out = {}
    if n_pos and n_neg:
        conc = 0
        for sp, lp in rows:
            if lp == 1:
                conc += sum(1 for sn, ln in rows if ln == 0 and sn < sp)
        out["auc"] = conc / (n_pos * n_neg)
    out["mrr"] = 0.0
    for i, l in enumerate(labels_ranked):
        if l == 1:
            out["mrr"] = 1.0 / (i + 1)
            break
    for k in ks:
        eff_k = min(k, len(ranked))
        top = labels_ranked[:eff_k]
        dcg = sum(l / math.log2(i + 2) for i, l in enumerate(top))
        ideal = sorted(labels_ranked, reverse=True)[:eff_k]
        idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        out[f"prec@{k}"] = sum(top) / k
        out[f"rec@{k}"] = sum(top) / n_pos if n_pos else 0.0
        out[f"hits@{k}"] = sum(top)
    out["n_pos"] = n_pos
    out["n"] = len(rows)
    return out


def _make_jagged(seed=0, n_users=40, max_len=25):
    rng = np.random.default_rng(seed)
    users, scores, labels = [], [], []
    for u in range(n_users):
        L = int(rng.integers(1, max_len))
        users += [f"u{u:03d}"] * L
        scores += list(rng.normal(size=L))
        labels += list((rng.random(L) < 0.15).astype(float))
    return (np.array(users), np.array(scores, dtype=np.float64),
            np.array(labels, dtype=np.float64))


class TestEvaluateFlat:
    KS = (3, 10)

    def _naive_aggregate(self, users, scores, labels):
        per_user = {}
        for u, s, l in zip(users, scores, labels):
            per_user.setdefault(u, []).append((s, l))
        stats = [_naive_user_metrics(rows, self.KS)
                 for rows in per_user.values()]
        auc_users = [s for s in stats if "auc" in s]
        pos_users = [s for s in stats if s["n_pos"] > 0]
        agg = {
            "auc": np.mean([s["auc"] for s in auc_users]),
            "gauc": (sum(s["auc"] * s["n"] for s in auc_users)
                     / sum(s["n"] for s in auc_users)),
            "mrr": np.mean([s["mrr"] for s in pos_users]),
        }
        for k in self.KS:
            agg[f"ndcg@{k}"] = np.mean([s[f"ndcg@{k}"] for s in pos_users])
            agg[f"precision@{k}"] = np.mean([s[f"prec@{k}"] for s in stats])
            agg[f"recall@{k}_conditional"] = np.mean(
                [s[f"rec@{k}"] for s in pos_users])
            agg[f"recall@{k}_listed"] = np.mean([s[f"rec@{k}"] for s in stats])
        return agg

    def test_matches_naive_reference(self):
        users, scores, labels = _make_jagged(seed=1)
        got = evaluate_flat(users, scores, labels, ks=self.KS)
        want = self._naive_aggregate(users, scores, labels)
        for key, val in want.items():
            assert got[key] == pytest.approx(val, abs=1e-6), key

    def test_batch_size_invariance(self):
        users, scores, labels = _make_jagged(seed=2)
        a = evaluate_flat(users, scores, labels, ks=self.KS, batch_users=3)
        b = evaluate_flat(users, scores, labels, ks=self.KS, batch_users=4096)
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-6), key

    def test_row_order_invariance(self):
        users, scores, labels = _make_jagged(seed=3)
        perm = np.random.default_rng(0).permutation(len(users))
        a = evaluate_flat(users, scores, labels, ks=self.KS)
        b = evaluate_flat(users[perm], scores[perm], labels[perm], ks=self.KS)
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-6), key

    def test_monotone_transform_invariance(self):
        users, scores, labels = _make_jagged(seed=4)
        a = evaluate_flat(users, scores, labels, ks=self.KS)
        b = evaluate_flat(users, 0.37 * scores - 5.0, labels, ks=self.KS)
        for key in ("auc", "gauc", "global_auc", "mrr", "ndcg@3",
                    "recall@10_listed", "precision@3"):
            assert a[key] == pytest.approx(b[key], abs=1e-6), key

    def test_overall_recall_denominator(self):
        # 2 listed users (one with the target, one without) + 2 gt users who
        # never got candidate rows -> recall@k_overall divides by 4.
        users = np.array(["a", "a", "b", "b"])
        scores = np.array([2.0, 1.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 0.0, 0.0])
        got = evaluate_flat(users, scores, labels, ks=(1,), n_gt_users=4)
        assert got["recall@1_overall"] == pytest.approx(0.25)
        assert got["recall@1_conditional"] == pytest.approx(1.0)
        assert got["recall@1_listed"] == pytest.approx(0.5)

    def test_n_gt_users_smaller_than_listed_raises(self):
        users = np.array(["a", "b"])
        ones = np.ones(2)
        with pytest.raises(ValueError):
            evaluate_flat(users, ones, ones, n_gt_users=1)

    def test_tie_policy_no_auc_credit_any_row_order(self):
        # tied pos/neg pair -> zero credit regardless of row order
        for labels in ([1.0, 0.0], [0.0, 1.0]):
            users = np.array(["a", "a"])
            scores = np.array([1.0, 1.0])
            got = evaluate_flat(users, scores, np.array(labels), ks=(1,))
            assert got["auc"] == 0.0, labels
            assert got["global_auc"] == 0.0, labels

    def test_zeroed_target_is_deterministically_conservative(self):
        # rule-layer shape: 10 surviving negatives (scores 1..10) + 20 rows
        # zeroed to 0.0 including the single positive. The zeroed positive
        # must rank below every co-zeroed negative under ANY row permutation:
        # no recall@10 credit, mrr = 1/30, auc = 0.
        rng = np.random.default_rng(3)
        scores = np.array([float(i + 1) for i in range(10)] + [0.0] * 20)
        labels = np.array([0.0] * 10 + [1.0] + [0.0] * 19)
        users = np.array(["u"] * 30)
        for _ in range(5):
            perm = rng.permutation(30)
            got = evaluate_flat(users[perm], scores[perm], labels[perm],
                                ks=(10, 25))
            assert got["recall@10_listed"] == 0.0
            assert got["recall@25_listed"] == 0.0
            assert got["mrr"] == pytest.approx(1 / 30)
            assert got["auc"] == 0.0

    def test_tied_data_permutation_invariance(self):
        # heavy ties across many users: results identical under permutation
        rng = np.random.default_rng(9)
        users = np.repeat([f"u{i}" for i in range(12)], 20)
        scores = rng.choice([0.0, 0.1, 0.5], size=240)
        labels = (rng.random(240) < 0.2).astype(float)
        a = evaluate_flat(users, scores, labels, ks=(5,))
        perm = rng.permutation(240)
        b = evaluate_flat(users[perm], scores[perm], labels[perm], ks=(5,))
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-12), key

    def test_matches_legacy_topk_recall(self):
        # recall@k_overall == legacy mean recall only under the snapshot
        # protocol's one-positive-per-user ground truth, so build such data.
        from scripts.ranker.ranker_features import _scores_to_topk
        from scripts.retrieval.evaluator import recall_precision_at_k
        rng = np.random.default_rng(5)
        users_l, scores_l, labels_l = [], [], []
        for u in range(40):
            L = int(rng.integers(2, 25))
            pos = int(rng.integers(0, L)) if rng.random() < 0.7 else -1
            users_l += [f"u{u:03d}"] * L
            scores_l += list(rng.normal(size=L))
            labels_l += [1.0 if j == pos else 0.0 for j in range(L)]
        users, scores, labels = (np.array(users_l), np.array(scores_l),
                                 np.array(labels_l))
        gt = {}
        for u, l, pa in zip(users, labels, range(len(users))):
            if l == 1:
                gt.setdefault(u, set()).add(f"i{pa}")
        asins = np.array([f"i{j}" for j in range(len(users))])
        topk = _scores_to_topk(scores, users, asins, k=10)
        for k in (3, 10):
            legacy_r, _, _ = recall_precision_at_k(topk, gt, k)
            got = evaluate_flat(users[np.isin(users, list(gt))],
                                scores[np.isin(users, list(gt))],
                                labels[np.isin(users, list(gt))],
                                ks=(k,), n_gt_users=len(gt))
            assert got[f"recall@{k}_overall"] == pytest.approx(legacy_r, abs=1e-9)


class TestMultiPositiveSemantics:
    """recall@k_overall must stay a RECALL when a user has several positives.

    The snapshot protocol used to emit exactly one groundtruth row per user
    (first positive in the label window), under which mean-per-user-recall and
    mean-hit-count are the same number. Moving to all-positives-in-the-window
    separates them, and only the first is a recall.
    """

    def test_backward_compatible_under_one_positive_per_user(self):
        # a: positive at rank 1. b: positive at rank 2. Plus 2 gt users with
        # no candidate rows at all. All three overall flavours must coincide.
        users = np.array(["a", "a", "b", "b"])
        scores = np.array([2.0, 1.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 0.0, 1.0])
        got = evaluate_flat(users, scores, labels, ks=(1,), n_gt_users=4)
        assert got["recall@1_overall"] == pytest.approx(0.25)
        assert got["hit_rate@1_overall"] == pytest.approx(0.25)
        assert got["hits@1_per_gt_user"] == pytest.approx(0.25)

    def test_recall_is_macro_average_not_hit_count(self):
        # a: 4 rows, positives at ranks 1 and 3 -> recall@2 = 1/2, hits@2 = 1
        # b: 4 rows, single positive at rank 4  -> recall@2 = 0,   hits@2 = 0
        # 2 further gt users contribute no rows. n_gt_users = 4.
        users = np.array(["a"] * 4 + ["b"] * 4)
        scores = np.array([4.0, 3.0, 2.0, 1.0] * 2)
        labels = np.array([1.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 1.0])
        got = evaluate_flat(users, scores, labels, ks=(2,), n_gt_users=4)
        assert got["recall@2_overall"] == pytest.approx(0.5 / 4)
        assert got["hits@2_per_gt_user"] == pytest.approx(1 / 4)
        assert got["hit_rate@2_overall"] == pytest.approx(1 / 4)
        # The three genuinely disagree -> the distinction is load bearing.
        assert got["recall@2_overall"] != pytest.approx(got["hits@2_per_gt_user"])
        assert got["recall@2_conditional"] == pytest.approx(0.25)

    def test_recall_overall_never_exceeds_one(self):
        # One gt user holding three positives, all inside top-3. The pre-fix
        # numerator (sum of hit COUNTS) would report 3.0 here.
        users = np.array(["a"] * 3)
        scores = np.array([3.0, 2.0, 1.0])
        labels = np.array([1.0, 1.0, 1.0])
        got = evaluate_flat(users, scores, labels, ks=(3,), n_gt_users=1)
        assert got["recall@3_overall"] == pytest.approx(1.0)
        assert got["hits@3_per_gt_user"] == pytest.approx(3.0)

    def test_matches_legacy_evaluator_under_multi_positive(self):
        # scripts/retrieval/evaluator.py has always divided by |gt[u]|, so it
        # was already multi-positive correct. The two paths must now agree on
        # data where several positives per user are the norm, not just on the
        # one-positive data the older test had to construct.
        from scripts.ranker.ranker_features import _scores_to_topk
        from scripts.retrieval.evaluator import recall_precision_at_k
        rng = np.random.default_rng(11)
        users_l, scores_l, labels_l = [], [], []
        for u in range(40):
            L = int(rng.integers(4, 25))
            n_pos = int(rng.integers(0, 5))
            pos_at = set(rng.choice(L, size=min(n_pos, L), replace=False).tolist())
            users_l += [f"u{u:03d}"] * L
            scores_l += list(rng.normal(size=L))
            labels_l += [1.0 if j in pos_at else 0.0 for j in range(L)]
        users = np.array(users_l)
        scores = np.array(scores_l)
        labels = np.array(labels_l)
        asins = np.array([f"i{j}" for j in range(len(users))])
        gt = {}
        for u, l, a in zip(users, labels, asins):
            if l == 1:
                gt.setdefault(u, set()).add(a)
        assert max(len(v) for v in gt.values()) > 1, "fixture must be multi-positive"

        topk = _scores_to_topk(scores, users, asins, k=10)
        keep = np.isin(users, list(gt))
        for k in (3, 10):
            legacy_r, legacy_p, _ = recall_precision_at_k(topk, gt, k)
            got = evaluate_flat(users[keep], scores[keep], labels[keep],
                                ks=(k,), n_gt_users=len(gt))
            # 1e-6 is this module's house tolerance against the pure-Python
            # references: evaluate_flat accumulates per-user values in float32,
            # which drifts ~1e-8 over these fixtures. It still pins semantics —
            # the pre-fix numerator was off by whole multiples here, not 1e-8.
            assert got[f"recall@{k}_overall"] == pytest.approx(legacy_r, abs=1e-6)
            # Precision stops being recall/k once |gt[u]| varies across users.
            assert got[f"precision@{k}"] == pytest.approx(legacy_p, abs=1e-6)

    def test_precision_is_independent_information_under_multi_positive(self):
        users = np.array(["a"] * 4 + ["b"] * 4)
        scores = np.array([4.0, 3.0, 2.0, 1.0] * 2)
        labels = np.array([1.0, 1.0, 0.0, 0.0,      # a: 2 positives, both top-2
                           1.0, 0.0, 0.0, 0.0])     # b: 1 positive at rank 1
        got = evaluate_flat(users, scores, labels, ks=(2,), n_gt_users=2)
        # Both users have recall@2 == 1.0, but precision differs (2/2 vs 1/2).
        assert got["recall@2_overall"] == pytest.approx(1.0)
        assert got["precision@2"] == pytest.approx((1.0 + 0.5) / 2)
        assert got["precision@2"] != pytest.approx(got["recall@2_overall"] / 2)

    def test_missed_positives_must_cost_recall(self):
        # User bought 4 things; retrieval surfaced exactly one of them and the
        # ranker put it first. The rows alone cannot show the other three, so
        # without true_positives_per_user this scores a perfect 1.0.
        users = np.array(["a"] * 3)
        scores = np.array([3.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 0.0])

        blind = evaluate_flat(users, scores, labels, ks=(3,), n_gt_users=1)
        assert blind["recall@3_overall"] == pytest.approx(1.0)

        honest = evaluate_flat(users, scores, labels, ks=(3,), n_gt_users=1,
                               true_positives_per_user={"a": 4})
        assert honest["recall@3_overall"] == pytest.approx(0.25)
        # hit_rate ignores how many were missed -- that is its job, and it is
        # why it must not stand in for recall.
        assert honest["hit_rate@3_overall"] == pytest.approx(1.0)

    def test_ndcg_ideal_uses_full_ground_truth(self):
        # Retrieval surfaced 1 of the user's 4 purchases; the ranker put it
        # first. Judged against only what was retrieved that is a perfect
        # ranking (nDCG 1.0) -- the users retrieval served worst score best.
        users = np.array(["a"] * 10)
        scores = np.arange(10, 0, -1).astype(float)
        labels = np.zeros(10)
        labels[0] = 1.0

        blind = evaluate_flat(users, scores, labels, ks=(10,), n_gt_users=1)
        assert blind["ndcg@10"] == pytest.approx(1.0)

        # IDCG over the true ground truth is the first min(|P_u|, k) discounts:
        # 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) = 2.561606…
        idcg = sum(1.0 / math.log2(i + 2) for i in range(4))
        honest = evaluate_flat(users, scores, labels, ks=(10,), n_gt_users=1,
                               true_positives_per_user={"a": 4})
        assert honest["ndcg@10"] == pytest.approx(1.0 / idcg, abs=1e-6)
        assert honest["ndcg@10"] < blind["ndcg@10"]

    def test_ndcg_unchanged_when_all_positives_were_retrieved(self):
        users = np.array(["a"] * 10)
        scores = np.arange(10, 0, -1).astype(float)
        labels = np.zeros(10)
        labels[[0, 3]] = 1.0
        a = evaluate_flat(users, scores, labels, ks=(10,), n_gt_users=1)
        b = evaluate_flat(users, scores, labels, ks=(10,), n_gt_users=1,
                          true_positives_per_user={"a": 2})
        assert a["ndcg@10"] == pytest.approx(b["ndcg@10"], abs=1e-6)

    def test_listed_recall_also_honours_true_counts(self):
        users = np.array(["a"] * 3)
        scores = np.array([3.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 0.0])
        got = evaluate_flat(users, scores, labels, ks=(3,), n_gt_users=1,
                            true_positives_per_user={"a": 4})
        assert got["recall@3_listed"] == pytest.approx(0.25)
        # _conditional keeps the retrieved-positive denominator on purpose:
        # it isolates the ranker from retrieval.
        assert got["recall@3_conditional"] == pytest.approx(1.0)

    def test_true_positive_count_below_retrieved_is_rejected(self):
        users = np.array(["a", "a"])
        scores = np.array([2.0, 1.0])
        labels = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="must be part of the user"):
            evaluate_flat(users, scores, labels, ks=(2,), n_gt_users=1,
                          true_positives_per_user={"a": 1})

    def test_true_positive_counts_are_noop_under_one_positive(self):
        users = np.array(["a", "a", "b", "b"])
        scores = np.array([2.0, 1.0, 2.0, 1.0])
        labels = np.array([1.0, 0.0, 0.0, 1.0])
        a = evaluate_flat(users, scores, labels, ks=(1,), n_gt_users=4)
        b = evaluate_flat(users, scores, labels, ks=(1,), n_gt_users=4,
                          true_positives_per_user={"a": 1, "b": 1})
        assert a["recall@1_overall"] == pytest.approx(b["recall@1_overall"])

    def test_auc_scale_context_recovers_the_true_rank(self):
        # One user: 1 positive at rank 3 among 10 negatives.
        # AUC = 8/10 = 0.8; implied mean rank = 1 + (1-0.8)*10 = 3 exactly;
        # the AUC needed to reach top-3 is 1 - 2/10 = 0.8, i.e. exactly at the
        # boundary -- which is where this positive actually sits.
        scores = np.arange(11, 0, -1).astype(float)
        labels = np.zeros(11)
        labels[2] = 1.0
        users = np.array(["a"] * 11)
        got = evaluate_flat(users, scores, labels, ks=(3,), n_gt_users=1)
        assert got["auc"] == pytest.approx(0.8)
        assert got["mean_negatives_per_auc_user"] == pytest.approx(10.0)
        assert got["auc_implied_mean_rank"] == pytest.approx(3.0)
        assert got["auc_needed_for_top3"] == pytest.approx(0.8)
        assert got["recall@3_overall"] == pytest.approx(1.0)

    def test_auc_threshold_explains_high_auc_with_low_recall(self):
        # The reported paradox, in miniature: a good percentile on a long list
        # is still a bad absolute rank. Positive at rank 30 of 201 -> AUC 0.855
        # (looks strong) yet it misses top-10 entirely.
        n = 201
        scores = np.arange(n, 0, -1).astype(float)
        labels = np.zeros(n)
        labels[29] = 1.0
        users = np.array(["a"] * n)
        got = evaluate_flat(users, scores, labels, ks=(10,), n_gt_users=1)
        assert got["auc"] == pytest.approx(171 / 200)
        assert got["recall@10_overall"] == pytest.approx(0.0)
        assert got["auc"] < got["auc_needed_for_top10"]
        assert got["auc_implied_mean_rank"] == pytest.approx(30.0)


class TestCalibrationPrimitives:
    def test_sigmoid_matches_and_is_stable(self):
        x = np.array([-800.0, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0])
        got = tc._sigmoid(x)
        assert np.all(np.isfinite(got))
        mid = 1.0 / (1.0 + np.exp(-x[1:-1]))
        assert got[1:-1] == pytest.approx(mid, abs=1e-12)
        assert got[0] == pytest.approx(0.0, abs=1e-12)
        assert got[-1] == pytest.approx(1.0, abs=1e-12)

    def test_ece_equal_width_hand_case(self):
        # bin [0.6,0.667): probs 0.62/0.64 pos_rate 0.5 -> |0.63-0.5|=0.13
        # bin [0,0.067): probs 0.02/0.04 pos_rate 0.0 -> |0.03-0|=0.03
        probs = np.array([0.62, 0.64, 0.02, 0.04])
        labels = np.array([1.0, 0.0, 0.0, 0.0])
        out = tc.ece_table(probs, labels, n_bins=15)
        assert out["ece"] == pytest.approx(0.5 * 0.13 + 0.5 * 0.03, abs=1e-9)
        assert out["n"] == 4

    def test_ece_prob_one_in_last_bin(self):
        out = tc.ece_table(np.array([1.0]), np.array([1.0]), n_bins=15)
        assert out["ece"] == pytest.approx(1.0 - 1.0, abs=1e-9)
        assert out["bins"][0]["n"] == 1

    def test_ece_quantile_reduces_to_bins(self):
        rng = np.random.default_rng(0)
        probs = rng.random(5000) * 0.01           # everything in one EW bin
        labels = (rng.random(5000) < 0.005).astype(float)
        q = tc.ece_quantile(probs, labels, n_bins=10)
        assert q["ece"] is not None and len(q["bins"]) > 3

    def test_platt_recovers_known_transform(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(0.0, 2.0, size=200_000)
        p_true = 1.0 / (1.0 + np.exp(-(1.7 * logits - 4.0)))
        labels = (rng.random(len(logits)) < p_true).astype(np.float64)
        fit = tc.fit_platt(logits, labels, init_shift=-5.0)
        assert fit["monotone"]
        assert fit["a"] == pytest.approx(1.7, abs=0.1)
        assert fit["b"] == pytest.approx(-4.0, abs=0.15)
        assert fit["final_nll"] <= fit["init_nll"] + 1e-12

    def test_cohort_and_bucket_codes(self):
        assert list(tc._cohort_codes(np.array([0, 1, 2, 3, 7]))) == [0, 1, 2, 3, 3]
        got = tc._item_bucket_codes(np.array([0.0, 9.9, 10.0, 99.0, 300.0, 1e6]))
        assert list(got) == [0, 0, 1, 2, 4, 4]

    def test_apply_thresholds_routing(self):
        probs = np.array([0.1, 0.5, 0.1, 0.5])
        cohort = np.array([0, 0, 3, 3])
        out = tc.apply_thresholds(probs, cohort, {"0": 0.2, "3+": 0.0})
        assert list(out) == [0.0, 0.5, 0.1, 0.5]
        # input untouched
        assert list(probs) == [0.1, 0.5, 0.1, 0.5]


def _write_gt(tdir: Path, snapshot: str, rows):
    """rows: (user_id, parent_asin, n_history_events) triples."""
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["user_id", "parent_asin", "n_history_events"]
                 ).to_parquet(tdir / f"groundtruth_{snapshot}.parquet",
                              index=False)


class TestGroundTruthDenominators:
    """n_gt_users must count USERS, and |P_u| must come from the label frame.

    Under all_positive_labels the groundtruth frame holds one row per
    (user, positive item). Counting rows inflates every recall@k_overall /
    hit_rate@k_overall denominator by the positives-per-user ratio, and
    reading |P_u| off the dump instead of the frame hides the positives
    retrieval never surfaced.
    """

    def test_cohort_counts_are_distinct_users_not_positives(self, tmp_path):
        _write_gt(tmp_path / "Toy" / "temporal_ranker", "test", [
            ("u1", "a", 0), ("u1", "b", 0), ("u1", "c", 0),
            ("u2", "d", 2), ("u2", "e", 2),
            ("u3", "f", 7),
        ])
        got = tc._gt_cohort_counts("Toy", "test", tmp_path)
        # 3 users over 6 rows: the row count would say total=6 and put three
        # users in cohort "0" where there is one.
        assert got == {"total": 3, "0": 1, "1": 0, "2": 1, "3+": 1}

    def test_cohort_counts_unchanged_under_one_positive(self, tmp_path):
        rows = [(f"u{i:02d}", f"i{i:02d}", i % 5) for i in range(20)]
        _write_gt(tmp_path / "Toy" / "temporal_ranker", "test", rows)
        got = tc._gt_cohort_counts("Toy", "test", tmp_path)
        assert got["total"] == 20
        assert sum(got[c] for c in tc.COHORT_NAMES) == 20

    def test_cohort_counts_reject_ambiguous_history(self, tmp_path):
        _write_gt(tmp_path / "Toy" / "temporal_ranker", "test", [
            ("u1", "a", 0), ("u1", "b", 4)])
        with pytest.raises(AssertionError, match="varies within a user"):
            tc._gt_cohort_counts("Toy", "test", tmp_path)

    def test_true_positive_counts_keyed_by_dump_code(self, tmp_path):
        _write_gt(tmp_path / "Toy" / "temporal_ranker", "test", [
            ("u1", "a", 0), ("u1", "b", 0), ("u1", "a", 0),   # dup (u1, a)
            ("u2", "d", 2),
        ])
        # user_ids are the factorize uniques, so position == user code.
        got = tc._true_positive_counts(np.array(["u2", "u1"], dtype=object),
                                       "Toy", "test", tmp_path)
        assert got == {0: 1, 1: 2}

    def test_true_positive_counts_reject_unknown_scored_user(self, tmp_path):
        _write_gt(tmp_path / "Toy" / "temporal_ranker", "test",
                  [("u1", "a", 0)])
        with pytest.raises(ValueError, match="out of sync"):
            tc._true_positive_counts(np.array(["u1", "zz"], dtype=object),
                                     "Toy", "test", tmp_path)

    def test_metrics_block_divides_by_full_ground_truth(self):
        # One user, 4 true positives, retrieval surfaced 1 and it ranks first.
        d = {"user": np.array([0, 0, 0]),
             "label": np.array([1.0, 0.0, 0.0]),
             "cohort": np.zeros(3, dtype=np.int64),
             "item_bucket": np.zeros(3, dtype=np.int64)}
        scores = np.array([3.0, 2.0, 1.0])
        counts = {"total": 1, "0": 1, "1": 0, "2": 0, "3+": 0}
        blind = tc.metrics_block(d, scores, counts, batch_users=8,
                                 device="cpu")
        honest = tc.metrics_block(d, scores, counts, batch_users=8,
                                  device="cpu", true_positives={0: 4})
        assert blind["overall"]["recall@10_overall"] == pytest.approx(1.0)
        assert honest["overall"]["recall@10_overall"] == pytest.approx(0.25)
        assert honest["cohort_0"]["recall@10_overall"] == pytest.approx(0.25)

    def test_label_mode_inferred_from_frame(self, tmp_path):
        tdir = tmp_path / "Toy" / "temporal_ranker"
        _write_gt(tdir, "test", [("u1", "a", 0), ("u2", "b", 1)])
        assert tc._label_mode_of("Toy", "test", tmp_path) == "first_positive"
        _write_gt(tdir, "test", [("u1", "a", 0), ("u1", "b", 0)])
        assert tc._label_mode_of("Toy", "test", tmp_path) == "all_positive"

    def test_label_mode_prefers_the_manifest(self, tmp_path):
        tdir = tmp_path / "Toy" / "temporal_ranker"
        # One positive per user by accident, all_positive by protocol: only
        # the manifest can tell the two apart.
        _write_gt(tdir, "test", [("u1", "a", 0), ("u2", "b", 1)])
        (tdir / "snapshot_manifest.json").write_text(json.dumps(
            {"snapshots": {"test": {"labels": {"label_mode": "all_positive"}}}}))
        assert tc._label_mode_of("Toy", "test", tmp_path) == "all_positive"

    def test_published_label_mode_inference(self):
        assert tc._published_label_mode(
            {"final_test": {"n_groundtruth_users": 10,
                            "n_groundtruth_positives": 24}})["mode"] == "all_positive"
        assert tc._published_label_mode(
            {"final_test": {"n_groundtruth_users": 10,
                            "n_groundtruth_positives": 10}})["mode"] == "first_positive"
        # Pre-switch reports record no positive count at all.
        old = tc._published_label_mode({"final_test": {"n_groundtruth_users": 10}})
        assert old["mode"] == "first_positive" and "assumed" in old["source"]


# ---------------------------------------------------------------------------
# Fixture: a miniature predictions dir + ground truth for stage tests
# ---------------------------------------------------------------------------

def _write_fixture(root: Path, seed=11, n_users=60, list_len=30):
    rng = np.random.default_rng(seed)
    cat, var = "Toy", "C2_k500_recent"
    tdir = root / cat / "temporal_ranker"
    pdir = tdir / "variants" / var / "predictions"
    pdir.mkdir(parents=True)

    def make_snapshot(fname, shift, sd):
        users, logits, labels, nhist, item_rev = [], [], [], [], []
        for u in range(n_users):
            n_h = int(rng.integers(0, 6))
            has_pos = rng.random() < 0.8
            pos_idx = int(rng.integers(0, list_len)) if has_pos else -1
            for j in range(list_len):
                users.append(f"u{u:03d}")
                lab = 1.0 if j == pos_idx else 0.0
                labels.append(lab)
                logits.append(rng.normal(shift + 2.5 * lab, sd))
                nhist.append(n_h)
                item_rev.append(float(rng.integers(0, 500)))
        df = pd.DataFrame({
            "user_id": users, "parent_asin": [f"i{j}" for j in range(len(users))],
            "label": np.array(labels, dtype=np.int8),
            "logit": np.array(logits, dtype=np.float32),
            "n_history_events": np.array(nhist, dtype=np.int32),
            "item_n_reviews_hist": np.array(item_rev, dtype=np.float32),
        })
        df.to_parquet(pdir / fname, index=False)
        return df

    make_snapshot("selection_model_model_selection.parquet", 4.0, 1.0)
    make_snapshot("refit_model_ranker_train.parquet", 5.0, 0.5)
    make_snapshot("refit_model_model_selection.parquet", 5.0, 0.5)
    test_df = make_snapshot("refit_model_test.parquet", 4.0, 1.0)

    with open(pdir / "meta.json", "w") as f:
        json.dump({
            "selection_model": {"pos_weight": float(list_len - 1)},
            "refit_model": {"pos_weight": float(list_len - 1)},
        }, f)

    for snap in ("model_selection", "test"):
        gt = pd.DataFrame({
            "user_id": [f"u{u:03d}" for u in range(n_users)],
            "n_history_events": rng.integers(0, 6, size=n_users),
        })
        gt.to_parquet(tdir / f"groundtruth_{snap}.parquet", index=False)
    return cat, var, pdir, test_df


@pytest.fixture()
def fixture_dirs(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    results = tmp_path / "results"
    cat, var, pdir, test_df = _write_fixture(processed)
    return processed, results, cat, var, pdir, test_df


class TestStages:
    def test_fit_never_needs_test_dump(self, fixture_dirs):
        processed, results, cat, var, pdir, _ = fixture_dirs
        (pdir / "refit_model_test.parquet").unlink()  # test dump absent
        report = tc.run_fit(cat, var, processed_dir=processed,
                            results_dir=results)
        assert report["frozen"]["test_dump_untouched"]
        assert (results / f"{cat}_{var}_frozen.json").exists()
        # calibration reduced ECE on the fitting set
        cal = report["calibration_selection"]
        chosen = report["frozen"]["chosen_calibrator"]
        assert cal[chosen]["ece"] <= cal["raw"]["ece"]

    def test_frozen_invariant_to_test_labels(self, fixture_dirs):
        processed, results, cat, var, pdir, test_df = fixture_dirs
        tc.run_fit(cat, var, processed_dir=processed, results_dir=results)
        frozen_a = json.loads(
            (results / f"{cat}_{var}_frozen.json").read_text())
        # scramble test labels + logits, refit: frozen must not move
        scrambled = test_df.copy()
        scrambled["label"] = scrambled["label"].sample(
            frac=1.0, random_state=0).to_numpy()
        scrambled["logit"] = 0.0
        scrambled.to_parquet(pdir / "refit_model_test.parquet", index=False)
        tc.run_fit(cat, var, processed_dir=processed, results_dir=results)
        frozen_b = json.loads(
            (results / f"{cat}_{var}_frozen.json").read_text())
        frozen_a.pop("created_utc"), frozen_b.pop("created_utc")
        assert frozen_a == frozen_b

    def test_test_requires_frozen(self, fixture_dirs):
        processed, results, cat, var, _, _ = fixture_dirs
        with pytest.raises(FileNotFoundError, match="test-lock"):
            tc.run_test(cat, var, processed_dir=processed,
                        results_dir=results)

    def test_test_stage_is_one_shot(self, fixture_dirs):
        processed, results, cat, var, _, _ = fixture_dirs
        tc.run_fit(cat, var, processed_dir=processed, results_dir=results)
        tc.run_test(cat, var, processed_dir=processed, results_dir=results)
        with pytest.raises(RuntimeError, match="one-shot"):
            tc.run_test(cat, var, processed_dir=processed,
                        results_dir=results)
        # refitting after the test was spent is also refused
        with pytest.raises(RuntimeError, match="spent"):
            tc.run_fit(cat, var, processed_dir=processed,
                       results_dir=results)
        # explicit force restarts deliberately
        tc.run_test(cat, var, processed_dir=processed, results_dir=results,
                    force=True)

    def test_frozen_bound_to_dump_generation(self, fixture_dirs):
        processed, results, cat, var, pdir, _ = fixture_dirs
        tc.run_fit(cat, var, processed_dir=processed, results_dir=results)
        meta = json.loads((pdir / "meta.json").read_text())
        meta["created_utc"] = "2099-01-01T00:00:00+00:00"
        (pdir / "meta.json").write_text(json.dumps(meta))
        with pytest.raises(RuntimeError, match="dumps changed"):
            tc.run_test(cat, var, processed_dir=processed,
                        results_dir=results)

    def test_capped_dump_uses_dump_denominators(self, fixture_dirs):
        processed, results, cat, var, pdir, _ = fixture_dirs
        meta = json.loads((pdir / "meta.json").read_text())
        meta["max_users_per_snapshot"] = 7
        (pdir / "meta.json").write_text(json.dumps(meta))
        report = tc.run_fit(cat, var, processed_dir=processed,
                            results_dir=results)
        assert report["capped_run"] is True
        # denominators derived from the dump's own 60 users, not the full
        # groundtruth file
        assert report["groundtruth_cohort_counts"]["total"] == 60

    def test_end_to_end_fit_then_test(self, fixture_dirs):
        processed, results, cat, var, _, _ = fixture_dirs
        tc.run_fit(cat, var, processed_dir=processed, results_dir=results)
        report = tc.run_test(cat, var, processed_dir=processed,
                             results_dir=results)
        assert set(report["calibration_test"]) == {
            "raw", "prior", "platt", "final_with_thresholds"}
        assert set(report["metrics_test"]) == {"raw", "final_with_thresholds"}
        s = report["success_criteria"]
        assert isinstance(s["overall_ece_down"], bool)
        # chosen chain is calibrated toward honesty: mean prob moves toward
        # the true positive rate relative to raw inflated sigmoid
        d = report["distributions"]
        pos_rate = d["refit_model@test_raw"]["overall"]["pos_rate"]
        raw_gap = abs(d["refit_model@test_raw"]["overall"]["prob"]["mean"]
                      - pos_rate)
        cal_gap = abs(
            d["refit_model@test_calibrated"]["overall"]["prob"]["mean"]
            - pos_rate)
        assert cal_gap <= raw_gap

    def test_consistency_anchor_states_both_label_modes(
            self, fixture_dirs, monkeypatch, tmp_path):
        # The anchor is the reproducibility tripwire; across a label_mode
        # change the two sides stop answering the same question, so the report
        # has to say which mode each side used instead of silently comparing.
        processed, results, cat, var, _, _ = fixture_dirs
        pub_dir = tmp_path / "repo" / "results" / "phase2_temporal"
        pub_dir.mkdir(parents=True)
        (pub_dir / f"{cat}_ranker_{var}.json").write_text(json.dumps({
            "final_test": {"overall": {"Recall@100": 0.5},
                           "n_groundtruth_users": 60,
                           "n_groundtruth_positives": 150}}))
        monkeypatch.setattr(tc, "REPO_ROOT", tmp_path / "repo")
        tc.run_fit(cat, var, processed_dir=processed, results_dir=results)
        rep = tc.run_test(cat, var, processed_dir=processed,
                          results_dir=results)
        a = rep["consistency_anchor"]
        assert a["published_recall@100"] == 0.5
        assert a["published_label_mode"] == "all_positive"
        assert a["this_rerun_label_mode"] == "first_positive"
        assert a["label_modes_match"] is False
        assert "NOT VALID HERE" in a["note"]


class TestPredictionDump:
    def test_frame_schema_and_atomicity(self, tmp_path):
        df = pd.DataFrame({
            "user_id": ["a", "b"], "parent_asin": ["x", "y"],
            "label": [1, 0], "n_history_events": [2, 3],
            "item_n_reviews_hist": [10.0, np.nan],
        })
        out = tmp_path / "p.parquet"
        _dump_prediction_frame(out, df, np.array([0.5, -0.5]))
        back = pd.read_parquet(out)
        assert list(back.columns) == ["user_id", "parent_asin", "label",
                                      "logit", "n_history_events",
                                      "item_n_reviews_hist"]
        assert back["label"].dtype == np.int8
        assert back["logit"].dtype == np.float32
        assert back["item_n_reviews_hist"].to_list() == [10.0, 0.0]
        assert not list(tmp_path.glob("*.tmp"))

    def test_length_mismatch_raises(self, tmp_path):
        df = pd.DataFrame({"user_id": ["a"], "parent_asin": ["x"],
                           "label": [1], "n_history_events": [0],
                           "item_n_reviews_hist": [1.0]})
        with pytest.raises(ValueError, match="mismatch"):
            _dump_prediction_frame(tmp_path / "p.parquet", df,
                                   np.array([0.1, 0.2]))

    def test_cli_guard_requires_results_dir(self):
        from scripts.ranker.train_temporal_ranker import main
        with pytest.raises(SystemExit, match="results-dir"):
            main(["--category", "Toy", "--dump-predictions"])

    def test_run_value_guard_protects_locked_reports(self):
        # the guard fires before any file access, whatever path form is used
        from scripts.ranker.train_temporal_ranker import RESULTS_DIR, run
        with pytest.raises(ValueError, match="frozen report"):
            run("NoSuchCategory", dump_predictions=True)
        with pytest.raises(ValueError, match="frozen report"):
            run("NoSuchCategory", dump_predictions=True,
                results_dir=Path(str(RESULTS_DIR)))
