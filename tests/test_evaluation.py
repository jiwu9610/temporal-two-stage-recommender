"""Tests for evaluation metrics — verifies correctness against hand-computed values."""

import torch
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluation import Evaluator
from evaluation.metrics import (
    make_mask,
    auc_per_sample,
    global_auc,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    gauc,
)


# ---------------------------------------------------------------------------
# Fixtures: a small jagged batch
# ---------------------------------------------------------------------------
#   User 0: items=[1, 0, 0]        preds=[0.9, 0.3, 0.1]   (len=3)
#   User 1: items=[1, 1, 0, 1]     preds=[0.2, 0.8, 0.5, 0.7]  (len=4)
#   Padded to max_len=4

@pytest.fixture
def batch():
    labels = torch.tensor([
        [1, 0, 0, 0],   # user 0 (last position is padding)
        [1, 1, 0, 1],   # user 1
    ], dtype=torch.float)
    predictions = torch.tensor([
        [0.9, 0.3, 0.1, 0.0],
        [0.2, 0.8, 0.5, 0.7],
    ], dtype=torch.float)
    lengths = torch.tensor([3, 4])
    mask = make_mask(lengths, 4)
    return predictions, labels, lengths, mask


# ---------------------------------------------------------------------------
# make_mask
# ---------------------------------------------------------------------------

def test_make_mask():
    mask = make_mask(torch.tensor([3, 4]), 4)
    expected = torch.tensor([
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert mask.equal(expected)


def test_make_mask_single():
    mask = make_mask(torch.tensor([2]), 5)
    assert mask.equal(torch.tensor([[True, True, False, False, False]]))


# ---------------------------------------------------------------------------
# AUC per sample
# ---------------------------------------------------------------------------

def test_auc_per_sample_user0(batch):
    """User 0: labels=[1,0,0], preds=[0.9,0.3,0.1]
    Positive=1, Negatives=2.  Positive ranked highest -> AUC = 1.0
    """
    preds, labels, lengths, mask = batch
    auc_vals, valid = auc_per_sample(preds, labels, mask)
    assert valid[0].item()
    assert abs(auc_vals[0].item() - 1.0) < 1e-6


def test_auc_per_sample_user1(batch):
    """User 1: labels=[1,1,0,1], preds=[0.2,0.8,0.5,0.7]
    Sorted ascending by pred: (0.2,1), (0.5,0), (0.7,1), (0.8,1)
    n_pos=3, n_neg=1
    Neg at rank 2 (ascending). Positives at ranks 1,3,4.
    Concordant pairs: pos@rank3 vs neg@rank2 (1), pos@rank4 vs neg@rank2 (1)
    pos@rank1 vs neg@rank2 -> not concordant (pos ranked lower)
    concordant = 2, total pairs = 3*1 = 3, AUC = 2/3
    """
    preds, labels, lengths, mask = batch
    auc_vals, valid = auc_per_sample(preds, labels, mask)
    assert valid[1].item()
    assert abs(auc_vals[1].item() - 2.0 / 3.0) < 1e-6


def test_auc_no_negatives():
    """All positives -> AUC undefined, valid=False."""
    preds = torch.tensor([[0.5, 0.3]])
    labels = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[True, True]])
    auc_vals, valid = auc_per_sample(preds, labels, mask)
    assert not valid[0].item()


# ---------------------------------------------------------------------------
# Global AUC
# ---------------------------------------------------------------------------

def test_global_auc(batch):
    preds, labels, lengths, mask = batch
    val = global_auc(preds, labels, mask).item()
    # Valid items pooled:
    # preds: [0.9, 0.3, 0.1, 0.2, 0.8, 0.5, 0.7]
    # labels:[1,   0,   0,   1,   1,   0,   1  ]
    # Sorted ascending by pred: (0.1,0),(0.2,1),(0.3,0),(0.5,0),(0.7,1),(0.8,1),(0.9,1)
    # cum_neg at each pos:       1      1       2       3       3       3       3
    # Positives at positions 1,4,5,6 (0-indexed).  cum_neg = 1,3,3,3 -> concordant=10
    # n_pos=4, n_neg=3, total=12.  AUC = 10/12 = 5/6
    assert abs(val - 10.0 / 12.0) < 1e-6


# ---------------------------------------------------------------------------
# gAUC
# ---------------------------------------------------------------------------

def test_gauc(batch):
    preds, labels, lengths, mask = batch
    val = gauc(preds, labels, mask, lengths).item()
    # Weighted avg: (1.0*3 + 2/3*4) / (3+4) = (3 + 8/3) / 7 = 17/3 / 7 = 17/21
    expected = (1.0 * 3 + (2.0 / 3.0) * 4) / (3 + 4)
    assert abs(val - expected) < 1e-5


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

def test_mrr_user0(batch):
    """User 0 sorted desc by pred: [0.9(1), 0.3(0), 0.1(0)] -> first rel at rank 1 -> RR=1.0"""
    preds, labels, lengths, mask = batch
    mrr_vals = mrr(preds, labels, mask)
    assert abs(mrr_vals[0].item() - 1.0) < 1e-6


def test_mrr_user1(batch):
    """User 1 sorted desc by pred: [0.8(1), 0.7(1), 0.5(0), 0.2(1)] -> first rel at rank 1 -> RR=1.0"""
    preds, labels, lengths, mask = batch
    mrr_vals = mrr(preds, labels, mask)
    assert abs(mrr_vals[1].item() - 1.0) < 1e-6


def test_mrr_first_relevant_not_top():
    """First relevant item is at position 3."""
    preds = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    labels = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    mrr_vals = mrr(preds, labels, mask)
    assert abs(mrr_vals[0].item() - 1.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------

def test_ndcg_at_k_perfect_ranking():
    """Perfect ranking -> nDCG = 1.0"""
    preds = torch.tensor([[0.9, 0.1, 0.05]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    val = ndcg_at_k(preds, labels, mask, k=3)
    assert abs(val[0].item() - 1.0) < 1e-6


def test_ndcg_at_k_user1(batch):
    """User 1: preds=[0.2,0.8,0.5,0.7], labels=[1,1,0,1]
    Sorted desc: (0.8,1),(0.7,1),(0.5,0),(0.2,1)
    DCG@3 = 1/log2(2) + 1/log2(3) + 0/log2(4) = 1.0 + 0.6309 = 1.6309
    Ideal sorted labels desc: [1,1,1,0]
    IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.6309 + 0.5 = 2.1309
    nDCG@3 = 1.6309 / 2.1309
    """
    preds, labels, lengths, mask = batch
    val = ndcg_at_k(preds, labels, mask, k=3)
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 0.0
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected = dcg / idcg
    assert abs(val[1].item() - expected) < 1e-4


# ---------------------------------------------------------------------------
# Precision@k & Recall@k
# ---------------------------------------------------------------------------

def test_precision_at_k(batch):
    preds, labels, lengths, mask = batch
    # User 0 top-2 (desc): [0.9(1), 0.3(0)] -> prec@2 = 1/2
    val = precision_at_k(preds, labels, mask, k=2)
    assert abs(val[0].item() - 0.5) < 1e-6
    # User 1 top-2 (desc): [0.8(1), 0.7(1)] -> prec@2 = 1.0
    assert abs(val[1].item() - 1.0) < 1e-6


def test_recall_at_k(batch):
    preds, labels, lengths, mask = batch
    # User 0: 1 relevant total, top-2 has 1 relevant -> recall@2 = 1.0
    val = recall_at_k(preds, labels, mask, k=2)
    assert abs(val[0].item() - 1.0) < 1e-6
    # User 1: 3 relevant total, top-2 has 2 relevant -> recall@2 = 2/3
    assert abs(val[1].item() - 2.0 / 3.0) < 1e-6


def test_recall_no_relevant():
    """No relevant items -> recall = 0."""
    preds = torch.tensor([[0.5, 0.3]])
    labels = torch.tensor([[0.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    val = recall_at_k(preds, labels, mask, k=2)
    assert val[0].item() == 0.0


# ---------------------------------------------------------------------------
# Evaluator end-to-end
# ---------------------------------------------------------------------------

def test_evaluator_returns_all_keys(batch):
    preds, labels, lengths, _ = batch
    evaluator = Evaluator(ks=[5, 10])
    metrics = evaluator(preds, labels, lengths)
    expected_keys = {
        "auc", "gauc", "global_auc", "mrr",
        "ndcg@5", "ndcg@10",
        "precision@5", "precision@10",
        "recall@5", "recall@10",
    }
    assert set(metrics.keys()) == expected_keys


def test_evaluator_values_are_floats(batch):
    preds, labels, lengths, _ = batch
    evaluator = Evaluator(ks=[3])
    metrics = evaluator(preds, labels, lengths)
    for v in metrics.values():
        assert isinstance(v, float)


def test_evaluator_perfect_ranking():
    """Each user has 1 relevant item ranked first -> all metrics should be 1.0."""
    preds = torch.tensor([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
    ])
    labels = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    lengths = torch.tensor([3, 3])
    evaluator = Evaluator(ks=[1, 3])
    metrics = evaluator(preds, labels, lengths)
    assert abs(metrics["auc"] - 1.0) < 1e-6
    assert abs(metrics["mrr"] - 1.0) < 1e-6
    assert abs(metrics["ndcg@1"] - 1.0) < 1e-6
    assert abs(metrics["precision@1"] - 1.0) < 1e-6
    assert abs(metrics["recall@1"] - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
