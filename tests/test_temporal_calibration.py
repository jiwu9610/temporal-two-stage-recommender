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
