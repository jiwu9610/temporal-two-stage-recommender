# Recommendation System on Amazon Reviews 2023 — temporal evaluation rebuild

Two-stage recommender (retrieval → ranking) on Amazon Reviews 2023 across
four categories (Books, Electronics, Video_Games, All_Beauty), built around
one discipline: **measure honestly under a strict global-time protocol**.

- **Split**: global calendar cutoffs `T0 < T1 < T2 < T3` — train on history
  before T0 (labels [T0,T1)), select models on [T1,T2), and read the test
  window [T2,T3) **once** per declared evaluation. No per-user leave-one-out,
  no future information in features, popularity, or vocabularies.
- **Objective**: cover **every** purchase the user goes on to make in the
  future window (all-positive, multi-positive) — explicitly *not* next-item
  prediction. Label-frame integrity is enforced by hard assertions at
  training time (`assert_all_positive_labels`, bidirectional).
- **Retrieval** (per-source top-500, unioned): popularity, recency popularity,
  store/category rule affinity, two-tower v2 (in-batch softmax + taste
  channel), and a text-embedding **content source** for catalog reach.
- **Ranking**: MLP / Deep&Cross grid on 40+ point-in-time features; wave-3
  adds a **residual sequence module** (`logit = DCN + α·seq`, α=0 ≡ DCN) with
  vanilla/target-aware-pooling/causal/HSTU variants.

## Current results (locked test, all-positive objective — 2026-08)

Source of truth: [`results/REBUILD_WAVE_REPORT.md`](results/REBUILD_WAVE_REPORT.md)
and [`results/SUMMARY.md`](results/SUMMARY.md) §9. Headlines:

| Finding | Evidence |
|---|---|
| **Content retrieval is the one hard win**: Books R@100 **+23.7%**, R@10 +22.3% vs the no-content baseline, driven by candidate-ceiling coverage of zero-history items | one-shot test read; stated **under the metadata-catalog availability assumption** (85.2% of content-only hit items lack pre-T2 interaction evidence — quantified in the report) |
| **Sequence modules: measured negative.** No variant beats the DCN baseline beyond a two-seed noise band on R@100 in any category; not adopted | clean residual ablation, preregistered 3 configs/category, no test read spent |
| Threshold relaxation, identity repair, two-tower v2: no reliable test-time gain (value = correctness/auditability) | winner-vs-reference arms per category |
| The same SASRec that is flat here scores **in the literature band** (R@10 = 0.091) under the standard 5-core LOO protocol | anchor experiment, `results/anchor/COMPARISON.md` — the gap is the protocol and the 1–2-event median sequences, not the implementation |

Frozen production config per category (provenance, sha256, decision rule):
[`results/p4_freeze/frozen_config_ap.json`](results/p4_freeze/frozen_config_ap.json).

**Legacy note**: earlier phases (first-positive labels, leave-last-two
baselines) are preserved for reproducibility in `results/` and SUMMARY §1–§8,
and are superseded — including one disclosed protocol defect (rankers briefly
trained on first-positive labels; found by external review, fixed with hard
guards, everything re-run). See REBUILD_WAVE_REPORT §5 and `MEMO.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (current pipeline)

```bash
# data layer: snapshots + point-in-time feature stores (per category)
python -m scripts.data.temporal_ranker_pipeline --category Video_Games

# retrieval candidates on the all-positive frame (Stage A configs in configs/p4_stageA/)
python -m scripts.ranker.temporal_candidate_builder --category Video_Games \
    --variant A0_ap --label-mode all_positive \
    --config-json configs/p4_stageA/Video_Games_A0.json \
    --snapshots ranker_train,model_selection

# ranker selection (Stage B, test never read)
python -m scripts.ranker.train_temporal_ranker --category Video_Games \
    --variant A0_ap --label-mode all_positive --stage-b

# freeze (pure read, pre-declared rule, full provenance)
python -m scripts.analysis.p4_freeze_ap

# locked test of a frozen winner (exact frozen ranker recipe)
python -m scripts.ranker.train_temporal_ranker --category Books \
    --variant A3_ap --label-mode all_positive \
    --frozen-ranker mlp:0.0003:9 --results-dir results/p5_locked_ap/Books_A3_ap

# wave-3 sequence arm (residual; preregistration in results/wave3_seq/PREREGISTRATION_v2.md)
python -m scripts.ranker.train_temporal_ranker --category Books \
    --variant A3_ap --label-mode all_positive --stage-b \
    --seq-arm --seq-grid seq:hstu:pos=delta
```

Slurm drivers for every stage are under `slurm/`. Tests: `python -m pytest
tests/ --ignore=tests/smoke_ranker_dnn.py` (212 passing as of 2026-08-26).

## Repository layout

- `scripts/data/` — canonicalization (v2), temporal snapshots, feature stores
- `scripts/retrieval/` — sources, two-tower v1/v2, anchor runners
- `scripts/ranker/` — candidate builder, ranker trainer, seq module
- `scripts/evaluation/` — grouped metrics (multi-positive), calibration
- `scripts/analysis/` — Stage-A gate, freeze script, audits
- `results/` — reports (REBUILD_WAVE_REPORT.md is the entry point)
- `MEMO.md` — full lab notebook: decisions, incidents, 16 lessons

## Project spec

Advisor documents in `docs/` (`REBUILD_WAVE_SPEC.md`, `advanced_seq.pdf`,
`Recommendation_system_project_0425.pdf`).
