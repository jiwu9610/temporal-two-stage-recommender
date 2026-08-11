"""Shared constants for the anchor run scripts (Track C, REBUILD_WAVE_SPEC P1.2).

Importable from BOTH environments (the recbole-anchor venv and the tae conda
env), so: stdlib-only. No pandas / pyarrow / torch / recbole imports here.

Holds the paths, the leg E gate band, and the two memo disclosures the spec
requires recorded next to the results (Track C item 4), plus the writer that
materializes them as results/anchor/ANCHOR_MEMO.md.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_DATA_ROOT = REPO_ROOT / "data" / "anchor"
OFFICIAL_ROOT = ANCHOR_DATA_ROOT / "_official"
ANCHOR_RESULTS_ROOT = REPO_ROOT / "results" / "anchor"

CATEGORIES = ["All_Beauty", "Books", "Electronics", "Video_Games"]

# Leg E gate (REBUILD_WAVE_SPEC §1 Track C / P1.2): official-benchmark SASRec
# Recall@10 inside this band ==> pipeline + protocol healthy. Band synthesized
# from Pctx (arXiv:2510.21276 Table 1: 0.0847, max_len 20) and LLaDA-Rec
# (arXiv:2511.06254 Table 3: 0.0823).
LEG_E_BAND = (0.070, 0.097)

# Top-K dumped per leg: 100 + 1, so that after score_predictions removes the
# user's valid item from a dump (the seen-mask asymmetry fix) at least 100
# ranked items remain for @100 metrics.
TOPK_DUMP = 101

KS = (10, 50, 100)

# Keep in sync with scripts/anchor/export_anchor_splits.py:ISOLATION_DECLARATION
# (not imported: that module needs pyarrow, absent from the recbole venv).
ISOLATION_DECLARATION = (
    "ANCHOR ISOLATION: this split follows the official per-user leave-one-out "
    "protocol, which ignores the global temporal cutoffs of the production "
    "pipeline. Anchor training interactions fall inside the production locked "
    "test window [t2, t3). Every artifact under data/anchor/ and results/anchor/ "
    "(including any anchor two-tower checkpoint) is for protocol/pipeline "
    "health-checking only and MUST NOT enter a production candidate, ranker, or "
    "calibration pipeline. This is isolation, not an exemption: anchor produces "
    "no production metric."
)

# ---- Spec-required memo disclosures (Track C item 4) -------------------------

MEMO_SELECTION_ASYMMETRY = (
    "SELECTION ASYMMETRY (memo, Track C item 4): the two-tower leg D "
    "early-stops / selects its checkpoint on val Recall@100 "
    "(scripts/retrieval/train_two_tower.py), while every RecBole leg selects "
    "on val NDCG@10 (valid_metric). The two legs therefore optimize different "
    "selection criteria; small cross-leg deltas can be an artifact of this "
    "asymmetry, not of model class."
)

MEMO_LIFETIME_ITEM_FEATURES = (
    "LIFETIME-CATALOG AGGREGATES (memo, Track C item 4): leg D's "
    "item_features carry lifetime catalog aggregates (average_rating, "
    "rating_number from the metadata dump). Under per-user leave-one-out these "
    "are a future-information channel — the aggregates summarize ratings that "
    "include each user's held-out interactions. Inherited from phase1 on "
    "purpose (leg D reproduces the phase1 recipe); RecBole legs use no item "
    "features, so this advantage is leg-D-only. Flag sits next to every "
    "anchor-mode leg D row."
)

MEMO_SLICE_CAVEAT = (
    "SLICE CAVEAT: slice legs (our raw-slice 5/5-core) are never quoted next "
    "to published official-benchmark numbers. All_Beauty is footnote-only "
    "(253 users after dedup-first 5/5-core — degenerate scale)."
)

SEEN_MASK_CONVENTIONS = {
    "recbole_general_full_sort": (
        "RecBole general-model full-sort eval masks each user's history "
        "(train items at valid; train+valid items at test) before ranking."
    ),
    "recbole_sequential_full_sort": (
        "RecBole sequential full-sort eval applies NO history masking; the "
        "model ranks the full catalogue including seen items."
    ),
    "two_tower_dump": (
        "The two-tower predictions dump masks train-seen items only — the "
        "valid item is still rankable at test time. score_predictions.py "
        "removes each user's valid item from the dump before test scoring; "
        "only after that removal is the definition identical to the RecBole "
        "general-model convention (spec: '剔除后恒等')."
    ),
}


def leg_output_dir(category: str) -> Path:
    return ANCHOR_RESULTS_ROOT / category


MEMO_FILENAME = "ANCHOR_MEMO.md"


def write_anchor_memo() -> Path:
    """Materialize the spec-required memo lines next to the results.

    Idempotent (fixed content); every anchor run script calls this so the memo
    exists as soon as any result JSON does.
    """
    ANCHOR_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANCHOR_RESULTS_ROOT / MEMO_FILENAME
    lines = [
        "# Anchor results memo (Track C, REBUILD_WAVE_SPEC P1.2)",
        "",
        "Machine-written by scripts/anchor/anchor_common.py; the same text is "
        "embedded in every leg JSON under `protocol` / `anchor_protocol`.",
        "",
        f"- {MEMO_SELECTION_ASYMMETRY}",
        "",
        f"- {MEMO_LIFETIME_ITEM_FEATURES}",
        "",
        f"- {MEMO_SLICE_CAVEAT}",
        "",
        "## Seen-mask conventions per leg family",
        "",
    ]
    for name, text in SEEN_MASK_CONVENTIONS.items():
        lines.append(f"- **{name}**: {text}")
    lines += [
        "",
        f"## Leg E gate",
        "",
        f"- Official-benchmark SASRec Recall@10 in [{LEG_E_BAND[0]}, "
        f"{LEG_E_BAND[1]}] ⇒ pipeline + protocol healthy. Band anchored by "
        "Pctx (arXiv:2510.21276, max_len 20) and LLaDA-Rec (arXiv:2511.06254).",
        "",
        "## Isolation",
        "",
        f"- {ISOLATION_DECLARATION}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


COMPARISON_FILENAME = "COMPARISON.md"

# Display order for the comparison-table skeleton (spec P1.2: "4-way 对比表骨架").
_COMPARISON_LEGS = [
    ("pop", "A"), ("itemknn", "B"), ("sasrec", "C"), ("sasrec_len20", "C-20"),
    ("two_tower_anchor_v1", "D"),
    ("official_sasrec", "E-50"), ("official_sasrec_len20", "E-20"),
    ("official_pop", "F-pop"), ("official_itemknn", "F-knn"),
]


def write_comparison_skeleton() -> Path:
    """Regenerate results/anchor/COMPARISON.md from whatever leg JSONs exist.

    RecBole legs: their own test_metrics (RecBole protocol). Leg D: the
    RecBole-comparable numbers from score_two_tower_anchor_v1.json
    (valid item removed, rating-blind gt) — NOT the phase1-gt numbers in the
    leg's own JSON. Missing cells stay '—' (the skeleton part).
    Idempotent; called after every score_predictions run.
    """
    import json

    def fmt(x):
        return "—" if x is None else f"{x:.4f}"

    lines = [
        "# Anchor 4-way comparison (Track C, P1.2) — skeleton, auto-filled",
        "",
        "Regenerated by scripts/anchor/anchor_common.py:write_comparison_skeleton "
        "on every score_predictions run. RecBole legs show RecBole's own test "
        "metrics; leg D shows score_predictions' valid-removed / rating-blind "
        "numbers (the RecBole-comparable definition). Slice legs are NEVER "
        "quoted next to published numbers; All_Beauty is footnote-only "
        "(253 users, degenerate).",
        "",
    ]
    for cat in CATEGORIES:
        cat_dir = ANCHOR_RESULTS_ROOT / cat
        lines += [f"## {cat}", "",
                  "| leg | model | R@10 | NDCG@10 | R@50 | R@100 | note |",
                  "|---|---|---|---|---|---|---|"]
        for leg, letter in _COMPARISON_LEGS:
            if leg.startswith("official") and cat != "Video_Games":
                continue
            r10 = n10 = r50 = r100 = None
            note = ""
            if leg == "two_tower_anchor_v1":
                model = "two-tower (anchor_v1)"
                p = cat_dir / f"score_{leg}.json"
                if p.exists():
                    m = json.loads(p.read_text())["metrics"]["with_valid_removed"][
                        "official_rating_blind"]
                    r10, n10 = m.get("Recall@10"), m.get("NDCG@10")
                    r50, r100 = m.get("Recall@50"), m.get("Recall@100")
                    note = "valid removed; rating-blind gt"
            else:
                p = cat_dir / f"{leg}.json"
                model = leg
                if p.exists():
                    j = json.loads(p.read_text())
                    model = j.get("model", leg)
                    m = j.get("test_metrics", {})
                    r10, n10 = m.get("recall@10"), m.get("ndcg@10")
                    r50, r100 = m.get("recall@50"), m.get("recall@100")
                    gate = j.get("gate")
                    if gate:
                        note = (f"LEG E GATE {'PASS' if gate.get('passed') else 'FAIL'} "
                                f"band {gate.get('band')}")
            lines.append(f"| {letter} | {model} | {fmt(r10)} | {fmt(n10)} | "
                         f"{fmt(r50)} | {fmt(r100)} | {note} |")
        lines.append("")
    lines += [
        "Footnotes: " + MEMO_SLICE_CAVEAT,
        "",
        MEMO_SELECTION_ASYMMETRY,
        "",
        MEMO_LIFETIME_ITEM_FEATURES,
        "",
    ]
    ANCHOR_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANCHOR_RESULTS_ROOT / COMPARISON_FILENAME
    path.write_text("\n".join(lines))
    return path
