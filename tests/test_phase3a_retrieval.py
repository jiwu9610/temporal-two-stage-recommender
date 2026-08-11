"""Phase 3A retrieval-source and Stage A sweep invariants.

All synthetic. Verifies: windowed/decayed popularity point-in-time math,
co-visitation construction + retrieval determinism, future-event guards,
cold vs low-support classification, K-sweep/union/unique-hit arithmetic,
seen-item removal, row-order invariance, zero-history handling, and that the
Stage A sweep never reads groundtruth_test (test-lock).

Run from repo root (conda env tae):
    pytest tests/test_phase3a_retrieval.py
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

from scripts.data.temporal_split import MS_PER_DAY  # noqa: E402
from scripts.retrieval.temporal_sources import (  # noqa: E402
    build_content_profiles,
    build_covisitation,
    classify_targets,
    decayed_popularity,
    recommend_content_knn,
    recommend_covisitation,
    windowed_popularity,
)
from scripts.retrieval.phase3a_sweep import (  # noqa: E402
    SourceLists,
    _ranked_by_scores,
    _union_stats,
)


def day(n: float) -> int:
    return int(n * MS_PER_DAY)


AS_OF = day(100)


def _mk(rows):
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    df["label"] = (df["rating"] >= 3).astype("int8")
    return df


# ---------------------------------------------------------------------------
# windowed / decayed popularity
# ---------------------------------------------------------------------------

def test_windowed_popularity_point_in_time():
    hist = _mk([
        ("U1", "IA", 5, day(95)),    # inside 7d window
        ("U2", "IA", 5, day(80)),    # inside 30d only
        ("U3", "IA", 5, day(5)),     # lifetime only
        ("U4", "IB", 1, day(96)),    # hard negative -> never counted
        ("U5", "IB", 5, day(99)),
    ])
    w7 = windowed_popularity(hist, AS_OF, 7)
    assert w7["IA"] == 1 and w7["IB"] == 1
    w30 = windowed_popularity(hist, AS_OF, 30)
    assert w30["IA"] == 2
    w1000 = windowed_popularity(hist, AS_OF, 1000)
    assert w1000["IA"] == 3


def test_decayed_popularity_exact_values():
    hist = _mk([
        ("U1", "IA", 5, day(90)),    # age 10d
        ("U2", "IA", 5, day(80)),    # age 20d
        ("U3", "IB", 5, day(100 - 40)),  # age 40d
    ])
    d = decayed_popularity(hist, AS_OF, half_life_days=10)
    assert d["IA"] == pytest.approx(0.5 ** 1 + 0.5 ** 2)
    assert d["IB"] == pytest.approx(0.5 ** 4)
    # Ordering: higher decayed mass first.
    assert list(d.index) == ["IA", "IB"]


def test_future_guard_raises_everywhere():
    hist = _mk([("U1", "IA", 5, day(100))])   # exactly at as_of -> future
    with pytest.raises(ValueError, match="leakage"):
        windowed_popularity(hist, AS_OF, 30)
    with pytest.raises(ValueError, match="leakage"):
        decayed_popularity(hist, AS_OF, 30)
    with pytest.raises(ValueError, match="leakage"):
        build_covisitation(hist, AS_OF)


# ---------------------------------------------------------------------------
# co-visitation
# ---------------------------------------------------------------------------

@pytest.fixture
def covis_history():
    # U1: {IA, IB, IC}, U2: {IA, IB}, U3: {IA, IC} (all positives, pre-AS_OF)
    # co-occurrence: (IA,IB)=2, (IA,IC)=2, (IB,IC)=1
    rows = [
        ("U1", "IA", 5, day(10)), ("U1", "IB", 5, day(20)), ("U1", "IC", 5, day(30)),
        ("U2", "IA", 5, day(11)), ("U2", "IB", 5, day(21)),
        ("U3", "IA", 5, day(12)), ("U3", "IC", 5, day(22)),
        ("U4", "ID", 1, day(15)),          # hard negative: excluded from baskets
        ("U4", "IA", 1, day(16)),
    ]
    return _mk(rows)


def test_covisitation_counts_support_and_normalization(covis_history):
    nb_count = build_covisitation(covis_history, AS_OF, min_support=2,
                                  normalization="count")
    # (IB,IC) support 1 < 2 -> pruned; hard negatives never form pairs.
    assert [n for n, _ in nb_count["IA"]] == ["IB", "IC"]
    assert dict(nb_count["IA"])["IB"] == 2.0
    assert "IC" not in dict(nb_count.get("IB", []))
    # cosine: freq IA=3, IB=2, IC=2 -> score(IA,IB) = 2/sqrt(3*2)
    nb_cos = build_covisitation(covis_history, AS_OF, min_support=2,
                                normalization="cosine")
    assert dict(nb_cos["IA"])["IB"] == pytest.approx(2 / np.sqrt(6))


def test_covisitation_row_order_invariance(covis_history):
    shuffled = covis_history.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = build_covisitation(covis_history, AS_OF)
    b = build_covisitation(shuffled, AS_OF)
    assert a == b


def test_covisitation_no_future_pairs(covis_history):
    # A future co-purchase (after AS_OF) must be invisible: guard raises when
    # passed directly; snapshot histories are pre-filtered upstream, so the
    # correct usage is filtering first -- verify the filtered result ignores it.
    future = pd.concat([
        covis_history,
        _mk([("U9", "IB", 5, day(150)), ("U9", "IC", 5, day(151))]),
    ], ignore_index=True)
    with pytest.raises(ValueError):
        build_covisitation(future, AS_OF)
    visible = future[future["timestamp"] < AS_OF]
    nb = build_covisitation(visible, AS_OF, min_support=2, normalization="count")
    assert "IC" not in dict(nb.get("IB", []))   # (IB,IC) still support 1


def test_recommend_covisitation_seeds_decay_seen_and_zero_history(covis_history):
    nb = build_covisitation(covis_history, AS_OF, min_support=1,
                            normalization="count")
    seen = {"U1": {"IA", "IB", "IC"}, "U2": {"IA", "IB"}, "UZ": set()}
    out = recommend_covisitation(covis_history, ["U1", "U2", "UZ"], nb, seen,
                                 max_seeds=2, seed_decay=0.5, k=10)
    # U1 has seen everything -> nothing left after masking.
    assert out["U1"] == []
    # U2 seeds (recent first): IB(day21), IA(day11).
    # score(IC) = 1.0*nb[IB][IC](=1) + 0.5*nb[IA][IC](=2) = 2.0; only IC unseen.
    assert out["U2"] == ["IC"]
    # Zero-history user: empty list, no crash.
    assert out["UZ"] == []


def test_recommend_covisitation_deterministic_tiebreak(covis_history):
    nb = {"IA": [("IZ", 1.0), ("IB", 1.0)]}   # equal scores
    hist = _mk([("U1", "IA", 5, day(50))])
    out = recommend_covisitation(hist, ["U1"], nb, {"U1": {"IA"}}, k=10)
    assert out["U1"] == ["IB", "IZ"]           # alphabetical on ties


# ---------------------------------------------------------------------------
# content i2i: profiles + kNN retrieval
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def content_embeddings():
    """2-d toy embeddings: IA/IB axis-aligned, IC a zero row (item without an
    embedding after alignment), ID unnormalized on purpose."""
    vecs = {"IA": [1.0, 0.0], "IB": [0.0, 1.0],
            "IC": [0.0, 0.0], "ID": [3.0, 4.0]}
    asins = sorted(vecs)
    embs = np.array([vecs[a] for a in asins], dtype=np.float32)
    index = {a: j for j, a in enumerate(asins)}
    return embs, index


def test_content_profile_recency_decay_and_normalization(content_embeddings):
    embs, index = content_embeddings
    hist = _mk([
        ("U1", "IA", 5, day(10)),
        ("U1", "IB", 5, day(20)),        # more recent -> recency rank 0
        ("U1", "IA", 1, day(30)),        # hard negative: never enters
    ])
    prof = build_content_profiles(hist, AS_OF, embs, index,
                                  recency_decay=0.8, max_events=30)
    # p = normalize(1.0 * eIB + 0.8 * eIA)
    assert np.allclose(prof["U1"], _unit([0.8, 1.0]), atol=1e-6)
    assert np.linalg.norm(prof["U1"]) == pytest.approx(1.0)
    # max_events window: only the most recent positive survives.
    prof1 = build_content_profiles(hist, AS_OF, embs, index, max_events=1)
    assert np.allclose(prof1["U1"], [0.0, 1.0], atol=1e-6)


def test_content_profile_normalizes_item_vectors_first(content_embeddings):
    embs, index = content_embeddings
    hist = _mk([("U2", "ID", 5, day(10))])
    prof = build_content_profiles(hist, AS_OF, embs, index)
    # ID = [3, 4] must be unit-normalized before pooling.
    assert np.allclose(prof["U2"], [0.6, 0.8], atol=1e-6)


def test_content_profile_missing_embedding_consumes_recency_slot(
        content_embeddings):
    embs, index = content_embeddings
    # Recency order: IA (r=0), IC (r=1, zero row), IB (r=2).
    hist = _mk([
        ("U3", "IB", 5, day(10)),
        ("U3", "IC", 5, day(20)),
        ("U3", "IA", 5, day(30)),
    ])
    prof = build_content_profiles(hist, AS_OF, embs, index,
                                  recency_decay=0.8)
    # IC contributes nothing but keeps its slot: 1.0*eIA + 0.8^2*eIB,
    # NOT 1.0*eIA + 0.8*eIB.
    assert np.allclose(prof["U3"], _unit([1.0, 0.64]), atol=1e-6)
    assert not np.allclose(prof["U3"], _unit([1.0, 0.8]), atol=1e-3)
    # An item absent from the index entirely behaves the same way.
    hist2 = _mk([
        ("U4", "IB", 5, day(10)),
        ("U4", "IZMISSING", 5, day(20)),
        ("U4", "IA", 5, day(30)),
    ])
    prof2 = build_content_profiles(hist2, AS_OF, embs, index,
                                   recency_decay=0.8)
    assert np.allclose(prof2["U4"], _unit([1.0, 0.64]), atol=1e-6)


def test_content_profile_guard_no_future(content_embeddings):
    embs, index = content_embeddings
    hist = _mk([("U1", "IA", 5, day(100))])      # exactly at as_of -> future
    with pytest.raises(ValueError, match="leakage"):
        build_content_profiles(hist, AS_OF, embs, index)


def test_content_profile_unservable_users_get_no_entry(content_embeddings):
    embs, index = content_embeddings
    hist = _mk([
        ("UNEG", "IA", 1, day(10)),          # negatives only
        ("UZERO", "IC", 5, day(10)),         # only a zero-embedding positive
        ("UOK", "IA", 5, day(10)),
    ])
    prof = build_content_profiles(hist, AS_OF, embs, index)
    assert set(prof) == {"UOK"}


@pytest.fixture
def content_pool():
    items = np.array(["CA", "CB", "CC", "CD"])
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.8]],
                   dtype=np.float32)
    return items, mat


def test_content_knn_ranking_ties_seen_and_unserved(content_pool):
    items, mat = content_pool
    profiles = {"U1": np.array([1.0, 0.0], dtype=np.float32)}
    out, sc = recommend_content_knn(["U1", "U9"], profiles, items, mat,
                                    {}, k=10, return_scores=True)
    # cos: CA=CC=1.0 (tie -> alphabetical), CD=0.6, CB=0.0; raw cosines out.
    assert out["U1"] == ["CA", "CC", "CD", "CB"]
    assert sc["U1"][2] == pytest.approx(0.6, rel=1e-6)
    assert out["U9"] == [] and sc["U9"] == []        # no profile -> fallback
    # Seen-mask contract identical to covis: seen items never emitted.
    out2 = recommend_content_knn(["U1"], profiles, items, mat,
                                 {"U1": {"CA"}}, k=10)
    assert out2["U1"] == ["CC", "CD", "CB"]


def test_content_knn_deterministic_and_partition_matches_full_sort(
        content_pool):
    items, mat = content_pool
    profiles = {"U1": np.array([1.0, 0.0], dtype=np.float32)}
    full = recommend_content_knn(["U1"], profiles, items, mat, {}, k=4,
                                 oversample=64)
    # Rerun -> identical (determinism); tiny oversample forces the
    # argpartition path, whose truncation must agree with the full sort.
    assert recommend_content_knn(["U1"], profiles, items, mat, {}, k=4,
                                 oversample=64) == full
    for k in (1, 2, 3):
        part = recommend_content_knn(["U1"], profiles, items, mat, {}, k=k,
                                     oversample=1)
        assert part["U1"] == full["U1"][:k]


def test_content_knn_loud_contract_asserts(content_pool):
    items, mat = content_pool
    profiles = {"U1": np.array([1.0, 0.0], dtype=np.float32)}
    with pytest.raises(AssertionError, match="sorted"):
        recommend_content_knn(["U1"], profiles, items[::-1], mat[::-1], {})
    bad = mat.copy()
    bad[2] = [3.0, 4.0]                              # unnormalized row
    with pytest.raises(AssertionError, match="normalized"):
        recommend_content_knn(["U1"], profiles, items, bad, {})
    with pytest.raises(AssertionError, match="align"):
        recommend_content_knn(["U1"], profiles, items, mat[:2], {})


# ---------------------------------------------------------------------------
# content i2i inside the candidate builder: 1+cos storage + exact fill
# ---------------------------------------------------------------------------

BUILDER_CAT = "ContentCat"
_USER_FEATS = ["n_reviews_hist", "avg_rating_hist", "std_rating_hist",
               "n_unique_items_hist", "active_days_hist",
               "verified_rate_hist", "days_since_last_interaction",
               "interactions_last_30d", "positives_last_30d"]
_ITEM_FEATS = ["n_reviews_hist", "n_positives_hist", "avg_rating_hist",
               "n_unique_reviewers_hist", "n_features", "n_description",
               "n_categories"]


def _write_builder_fixture(root: Path) -> Path:
    """model_selection snapshot for build_snapshot_candidates + a content npz.

    Universe: eligible items E1/E2/E4/Z0, non-eligible CX. Embeddings (2-d):
    E1=[1,0], E2=[0,1], E4=[0.8,0.6], CX=[0.6,0.8], Z0 zero row (no
    embedding), XX=[0,1] exists only in the npz. UA's only positive is E1 ->
    profile [1,0]; UB is zero-history (no profile). Popularity: Z0 x3, E2 x2,
    the rest x1 -> UA's popularity top-2 is [Z0, E2] (E1 seen), so E2/Z0
    reach UA's rows through popularity, E4 through two-tower + content and CX
    (UA's target) through content ONLY.
    """
    tdir = root / BUILDER_CAT / "temporal_ranker"
    tdir.mkdir(parents=True)
    t0, t1, t2, t3 = day(60), day(80), day(100), day(120)

    hist = _mk([
        ("P1", "E2", 5, day(10)), ("P2", "E2", 5, day(11)),
        ("P3", "E4", 5, day(12)),
        ("P4", "Z0", 5, day(13)), ("P5", "Z0", 5, day(14)),
        ("P6", "Z0", 5, day(15)),
        ("P7", "CX", 5, day(16)),
        ("UA", "E1", 5, day(70)),
        ("UD", "E2", 1, day(41)),          # hard negative in history
    ])
    hist.to_parquet(tdir / "history_model_selection.parquet", index=False)

    gt = pd.DataFrame({
        "user_id": ["UA", "UB"], "parent_asin": ["CX", "E2"],
        "rating": [5, 4], "label": [1, 1],
        "timestamp": [day(85), day(86)],
        "n_history_events": [1, 0], "item_in_history": [1, 1],
    })
    gt.to_parquet(tdir / "groundtruth_model_selection.parquet", index=False)

    users = sorted(hist["user_id"].unique())
    uf = pd.DataFrame({"user_id": users})
    for i, c in enumerate(_USER_FEATS):
        uf[c] = np.linspace(0.5, 2.0, len(users)) + i
    uf.to_parquet(tdir / "user_features_model_selection.parquet", index=False)

    items = ["CX", "E1", "E2", "E4", "Z0"]
    itf = pd.DataFrame({
        "parent_asin": items,
        "main_category": ["C0", "C0", "C1", "C1", "C0"],
        "store": ["S0", "S1", "S0", "S1", np.nan],
        "in_eligible_pool": [0, 1, 1, 1, 1],
        "in_history_catalog": 1,
    })
    for i, c in enumerate(_ITEM_FEATS):
        itf[c] = np.arange(len(items), dtype=float) + i * 0.5
    itf.to_parquet(tdir / "item_features_model_selection.parquet", index=False)

    pd.DataFrame({"user_id": ["UA"], "parent_asin": ["E4"],
                  "score": [0.5], "rank": [1]}).to_parquet(
        tdir / "two_tower_predictions_model_selection.parquet", index=False)

    cutoffs = {"t0_ms": t0, "t1_ms": t1, "t2_ms": t2, "t3_ms": t3}
    json.dump({"cutoffs": cutoffs,
               "snapshots": {"model_selection": {"history_end_ms": t1}}},
              open(tdir / "snapshot_manifest.json", "w"))
    json.dump({"w_store": 1.0, "w_cat": 1.0, "w_pop": 1.0,
               "tuned_on": "ranker_train (T0->T1 development labels)",
               "cutoff_fingerprint": cutoffs},
              open(tdir / "rule_weights.json", "w"))

    # npz deliberately NOT alphabetical: the source must re-sort its pool.
    np.savez(root / BUILDER_CAT / "item_text_embeddings.npz",
             asins=np.array(["Z0", "E4", "CX", "E1", "E2", "XX"],
                            dtype=object),
             embs=np.array([[0.0, 0.0], [0.8, 0.6], [0.6, 0.8],
                            [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                           dtype=np.float32))
    return tdir


CONTENT_SOURCE = [{"name": "content_i2i", "type": "content_i2i",
                   "pool": "catalog"}]


def _build_content_candidates(root: Path, out: Path):
    from scripts.ranker.temporal_candidate_builder import (
        build_snapshot_candidates,
    )
    return build_snapshot_candidates(
        BUILDER_CAT, "model_selection", top_k=2, seed=42,
        processed_dir=root, require_two_tower=True,
        extra_sources=CONTENT_SOURCE, out_dir=out,
    )


def test_builder_content_scores_are_one_plus_cos_and_exact_filled(tmp_path):
    root = tmp_path / "proc"
    _write_builder_fixture(root)
    report = _build_content_candidates(root, tmp_path / "out")
    c = pd.read_parquet(tmp_path / "out" / "candidates_model_selection.parquet")

    ua = c[c["user_id"] == "UA"].set_index("parent_asin")
    # Emitted by the source: stored score is 1 + cosine (shifted to [0,2] so
    # resolve_feature_list's log1p(max(x,0)) never truncates a negative cos).
    assert ua.loc["E4", "source_content_i2i"] == 1
    assert ua.loc["E4", "content_i2i_rank"] == 1
    assert ua.loc["E4", "content_i2i_score"] == pytest.approx(1.8, rel=1e-5)
    assert ua.loc["CX", "source_content_i2i"] == 1
    assert ua.loc["CX", "content_i2i_rank"] == 2
    assert ua.loc["CX", "content_i2i_score"] == pytest.approx(1.6, rel=1e-5)
    # Exact fill: E2 reached UA through popularity only, but UA holds a
    # profile -> 1 + cos([1,0],[0,1]) = 1.0, NOT the blanket 0.0.
    assert ua.loc["E2", "source_content_i2i"] == 0
    assert ua.loc["E2", "content_i2i_score"] == pytest.approx(1.0, rel=1e-6)
    # Item without an embedding row: falls through to 0.0 even for a served
    # user.
    assert ua.loc["Z0", "source_content_i2i"] == 0
    assert ua.loc["Z0", "content_i2i_score"] == 0.0
    # Unserved user (no positive history -> no profile): blanket 0.0.
    ub = c[c["user_id"] == "UB"]
    assert len(ub) > 0
    assert (ub["source_content_i2i"] == 0).all()
    assert (ub["content_i2i_score"] == 0.0).all()
    # The shift keeps every stored value inside [0, 2].
    assert ((c["content_i2i_score"] >= 0.0)
            & (c["content_i2i_score"] <= 2.0)).all()
    assert "content_i2i" in report["sources"]


def test_builder_content_catalog_pool_reaches_non_eligible_target(tmp_path):
    """CX is UA's target, sits OUTSIDE the eligible pool (1 history event)
    and no base source can emit it -- the catalog-pool content source is the
    only reach into it, which is exactly the coverage claim of the wave."""
    root = tmp_path / "proc"
    _write_builder_fixture(root)
    report = _build_content_candidates(root, tmp_path / "out")
    c = pd.read_parquet(tmp_path / "out" / "candidates_model_selection.parquet")
    row = c[(c["user_id"] == "UA") & (c["parent_asin"] == "CX")]
    assert len(row) == 1
    assert row["label"].iloc[0] == 1
    assert row["num_sources"].iloc[0] == 1           # content and nothing else
    assert report["candidate_catalog_size"] == 4     # eligible pool untouched
    # Without the content source the same fixture never reaches CX.
    from scripts.ranker.temporal_candidate_builder import (
        build_snapshot_candidates,
    )
    base = build_snapshot_candidates(
        BUILDER_CAT, "model_selection", top_k=2, seed=42,
        processed_dir=root, require_two_tower=True,
        out_dir=tmp_path / "out_base",
    )
    c0 = pd.read_parquet(
        tmp_path / "out_base" / "candidates_model_selection.parquet")
    assert "CX" not in set(c0["parent_asin"])
    assert base["coverage"]["retrieved_union"] < \
        report["coverage"]["retrieved_union"]


def test_builder_content_run_is_byte_deterministic(tmp_path):
    """Same fixture, same seed -> byte-identical candidate parquet."""
    root = tmp_path / "proc"
    _write_builder_fixture(root)
    _build_content_candidates(root, tmp_path / "out_a")
    _build_content_candidates(root, tmp_path / "out_b")
    a = (tmp_path / "out_a" / "candidates_model_selection.parquet").read_bytes()
    b = (tmp_path / "out_b" / "candidates_model_selection.parquet").read_bytes()
    assert a == b


# ---------------------------------------------------------------------------
# classification: cold vs low_support vs eligible_not_retrieved vs retrieved
# ---------------------------------------------------------------------------

def test_classify_targets_four_way():
    gt = pd.DataFrame({
        "user_id": ["U1", "U2", "U3", "U4"],
        "parent_asin": ["IR", "IE", "IL", "IC"],
    })
    hist_counts = pd.Series({"IR": 10, "IE": 8, "IL": 2})   # IC never seen
    eligible = {"IR", "IE"}                                  # threshold 5
    retrieved = {"U1": {"IR"}, "U2": set(), "U3": set(), "U4": set()}
    s = classify_targets(gt, hist_counts, eligible, retrieved)
    assert s[("U1", "IR")] == "retrieved"
    assert s[("U2", "IE")] == "eligible_not_retrieved"
    assert s[("U3", "IL")] == "low_support"
    assert s[("U4", "IC")] == "cold"
    # One gt row per user: the histogram is exactly the legacy per-user one.
    assert s.value_counts().to_dict() == {"retrieved": 1,
                                          "eligible_not_retrieved": 1,
                                          "low_support": 1, "cold": 1}


def test_classify_targets_keeps_every_positive_of_a_user():
    """A user's targets can hold different statuses -- the classification is a
    property of the item, so keying by user dropped all but the last row."""
    gt = pd.DataFrame({
        "user_id": ["U1", "U1", "U1", "U2"],
        "parent_asin": ["IR", "IE", "IC", "IR"],
    })
    hist_counts = pd.Series({"IR": 10, "IE": 8})            # IC never seen
    eligible = {"IR", "IE"}
    retrieved = {"U1": {"IR"}, "U2": {"IR"}}
    s = classify_targets(gt, hist_counts, eligible, retrieved)
    assert len(s) == 4                                       # not 2 (per user)
    assert s[("U1", "IR")] == "retrieved"
    assert s[("U1", "IE")] == "eligible_not_retrieved"
    assert s[("U1", "IC")] == "cold"
    assert s[("U2", "IR")] == "retrieved"
    # Pre-fix U1 kept only its LAST row ("cold") and the histogram read
    # {"cold": 1, "retrieved": 1} -- 2 entries for 4 groundtruth targets.
    assert s.value_counts().to_dict() == {"retrieved": 2,
                                          "eligible_not_retrieved": 1,
                                          "cold": 1}
    assert list(s.index.names) == ["user_id", "parent_asin"]


# ---------------------------------------------------------------------------
# K-sweep arithmetic + deterministic ranking helpers
# ---------------------------------------------------------------------------

def test_ranked_by_scores_alphabetical_ties_and_seen():
    pool = np.array(["IA", "IB", "IC", "ID"])
    idx = {it: j for j, it in enumerate(pool)}
    scores = np.array([1.0, 2.0, 2.0, 0.5])
    out = _ranked_by_scores(pool, scores, set(), idx, k=4)
    assert out == ["IB", "IC", "IA", "ID"]       # tie IB/IC -> alphabetical
    out2 = _ranked_by_scores(pool, scores, {"IB"}, idx, k=4)
    assert out2 == ["IC", "IA", "ID"]            # seen removed entirely


def test_source_lists_and_union_stats():
    targets = {"U1": {"IA"}, "U2": {"IB"}, "U3": {"IC"}}
    s1 = SourceLists("s1", {"U1": ["IA", "IX"], "U2": ["IX"], "U3": ["IX"]}, targets)
    s2 = SourceLists("s2", {"U1": ["IX"], "U2": ["IX", "IB"], "U3": ["IX"]}, targets)
    assert s1.rank == {"U1": 1, "U2": 0, "U3": 0}
    assert s1.hits_at(1) == {"U1"}
    assert s1.hits_at(2) == {"U1"}
    assert s2.hits_at(2) == {"U2"}
    st = _union_stats([s1, s2], ["U1", "U2", "U3"], [2])["K=2"]
    assert st["union_hits"] == 2
    assert st["union_recall"] == pytest.approx(2 / 3)
    # One target per user -> the micro view is numerically the macro one.
    assert st["union_hits_per_positive"] == 2
    assert st["union_recall_per_positive"] == pytest.approx(2 / 3)
    # Each source uniquely contributes its own hit.
    assert st["per_source_unique_hits"] == {"s1": 1, "s2": 1}
    assert st["avg_unique_candidates_per_user"] == pytest.approx((2 + 2 + 1) / 3)


def test_source_lists_multi_positive_user_vs_positive_denominator():
    # U1 buys three things in the label window, U2 one. User-level recall must
    # count U1 ONCE; the per-positive view must see all four targets.
    targets = {"U1": {"IA", "IB", "IC"}, "U2": {"ID"}}
    s = SourceLists("s", {"U1": ["IB", "IZ", "IA"], "U2": ["IZ"]}, targets)
    assert s.ranks["U1"] == {"IA": 3, "IB": 1, "IC": 0}
    assert s.rank["U1"] == 1                      # best rank, not last-row's
    assert s.hits_at(1) == {"U1"} and s.hits_at(3) == {"U1"}
    assert s.hit_pairs_at(1) == {("U1", "IB")}
    assert s.hit_pairs_at(3) == {("U1", "IA"), ("U1", "IB")}
    st = _union_stats([s], ["U1", "U2"], [3])["K=3"]
    assert st["union_hits"] == 1
    assert st["union_recall"] == pytest.approx(1 / 2)     # users, never rows
    assert st["union_hits_per_positive"] == 2
    assert st["union_recall_per_positive"] == pytest.approx(2 / 4)
    # The K-vs-memory decision input scales with DISTINCT users.
    assert st["est_candidate_rows"] == int(np.mean([3, 1]) * 2)


def test_source_lists_rejects_scalar_target():
    with pytest.raises(TypeError):
        SourceLists("s", {"U1": ["IA"]}, {"U1": "IA"})   # would iterate chars


# ---------------------------------------------------------------------------
# test-lock: Stage A sweep must run WITHOUT groundtruth_test present
# ---------------------------------------------------------------------------

def _write_stage_a_fixture(tdir, gt: pd.DataFrame) -> None:
    """History / item features / manifest / weights / cached two-tower dump for
    a max_k=200 Stage A run. Only `gt` varies between the callers."""
    tdir.mkdir(parents=True)
    hist = _mk([
        ("P1", "IP", 5, day(10)), ("P2", "IP", 5, day(11)),
        ("P3", "IP", 5, day(12)), ("P4", "IP", 5, day(13)),
        ("P5", "IP", 5, day(14)),
        ("P1", "IQ", 5, day(20)), ("P2", "IQ", 5, day(21)),
        ("P3", "IQ", 5, day(22)), ("P4", "IQ", 5, day(23)),
        ("P5", "IQ", 5, day(24)),
        ("UA", "IX", 5, day(30)), ("UA", "IY", 5, day(31)), ("UA", "IZ", 5, day(32)),
    ])
    hist["label_type"] = "positive"
    items = sorted(hist["parent_asin"].unique())
    item_features = pd.DataFrame({
        "parent_asin": items, "main_category": "C",
        "store": [f"S{i%2}" for i in range(len(items))],
        "in_eligible_pool": [1 if hist["parent_asin"].value_counts()[i] >= 5
                             else 0 for i in items],
        "in_history_catalog": 1,
    })
    hist.to_parquet(tdir / "history_model_selection.parquet", index=False)
    gt.to_parquet(tdir / "groundtruth_model_selection.parquet", index=False)
    item_features.to_parquet(tdir / "item_features_model_selection.parquet",
                             index=False)
    # NOTE: NO groundtruth_test / history_test files exist at all.
    json.dump({"snapshots": {"model_selection": {"history_end_ms": day(100)}}},
              open(tdir / "snapshot_manifest.json", "w"))
    json.dump({"w_store": 1.0, "w_cat": 1.0, "w_pop": 1.0},
              open(tdir / "rule_weights.json", "w"))
    # Pre-cache tiny two-tower predictions so no checkpoint is needed.
    pd.DataFrame({"user_id": ["UA"], "parent_asin": ["IQ"],
                  "score": [1.0], "rank": [1]}).to_parquet(
        tdir / "two_tower_predictions_model_selection_k200.parquet", index=False)


def test_stage_a_never_reads_groundtruth_test(tmp_path):
    from scripts.retrieval.phase3a_sweep import run_category

    gt = pd.DataFrame({
        "user_id": ["UA", "UB"],
        "parent_asin": ["IP", "IQ"],
        "rating": [5, 5], "label": [1, 1],
        "timestamp": [day(101), day(102)],
        "n_history_events": [3, 0],
        "item_in_history": [1, 1], "item_in_eligible_pool": [1, 1],
        "is_warm_user": [1, 0],
    })
    _write_stage_a_fixture(tmp_path / "Cat" / "temporal_ranker", gt)

    report = run_category("Cat", max_k=200, skip_covis=False,
                          processed_dir=tmp_path, results_dir=tmp_path / "res")
    # Ran fine with the test window entirely absent -> provably not consulted.
    assert report["test_lock"].startswith("groundtruth_test never")
    # Manual check: UA's target IP is in the eligible pool (5 events) and
    # popularity ranks {IP, IQ}; UA hasn't seen IP -> popularity hits.
    assert report["k_sweep"]["singles"]["popularity"]["K=100"]["hits"] >= 1
    # UB is zero-history: contributes to denominator, never crashes.
    assert report["n_users"] == 2
    assert report["n_positives"] == 2            # one gt row per user


def test_stage_a_denominators_are_users_not_groundtruth_rows(tmp_path):
    """all_positive_labels groundtruth: UA holds two positives, one of them a
    cold item. `sorted(gt["user_id"])` used to make `users` 3 entries long, so
    every recall denominator, every per-user candidate mean and n_users itself
    counted UA twice."""
    from scripts.retrieval.phase3a_sweep import run_category

    gt = pd.DataFrame({
        "user_id": ["UA", "UA", "UB"],
        "parent_asin": ["IP", "ICOLD", "IQ"],
        "rating": [5, 5, 5], "label": [1, 1, 1],
        "timestamp": [day(101), day(103), day(102)],
        "n_history_events": [3, 3, 0],
        "item_in_history": [1, 0, 1], "item_in_eligible_pool": [1, 0, 1],
        "is_warm_user": [1, 1, 0],
    })
    _write_stage_a_fixture(tmp_path / "Cat" / "temporal_ranker", gt)

    report = run_category("Cat", max_k=200, skip_covis=True,
                          processed_dir=tmp_path, results_dir=tmp_path / "res")
    assert report["n_users"] == 2 and report["n_positives"] == 3
    # Popularity ranks [IP, IQ] for both users: UA hits IP (ICOLD is not in the
    # catalog at all), UB hits IQ -> 2/2 users but only 2/3 purchases.
    pop100 = report["k_sweep"]["singles"]["popularity"]["K=100"]
    assert pop100["hits"] == 2
    assert pop100["recall"] == pytest.approx(1.0)
    assert pop100["recall_per_positive"] == pytest.approx(2 / 3)
    union = report["k_sweep"]["all_three"]["K=100"]
    assert union["union_recall"] == pytest.approx(1.0)
    assert union["union_recall_per_positive"] == pytest.approx(2 / 3)
    assert union["est_candidate_rows"] == 4          # 2 candidates x 2 users
    abl = report["eligibility_ablation"]["min_item_history=5"]
    # Macro: UA covers 1 of its 2 targets, UB 1 of 1 -> 0.75; micro -> 2/3.
    assert abl["historical_catalog_coverage"] == pytest.approx(0.75)
    assert abl["historical_catalog_coverage_per_positive"] == pytest.approx(2 / 3)
    assert abl["eligible_pool_coverage"] == pytest.approx(0.75)
    # The status histogram covers every target, not every user.
    assert sum(abl["target_status_counts"].values()) == 3
    assert report["base_cohorts"]["3+"]["n_users"] == 1