# Results index

**Current, valid results = the all-positive temporal generation (`_ap`).**
Entry point: [REBUILD_WAVE_REPORT.md](REBUILD_WAVE_REPORT.md). Everything
superseded lives under [`archive/`](../archive/README.md) and backs no
current claim.

## Where each number in the report comes from

| Claim | Artifact |
|---|---|
| Locked-test table (winner + A0 reference per category) | `p5_locked_ap/{cat}_{arm}/{cat}_ranker_{arm}.json` (`final_test.overall`, ceilings, cohort tables incl. 5–9 / 10+ history buckets) |
| Frozen configuration + decision rule + input hashes | `p4_freeze/frozen_config_ap.json`; per-category retrieval configs in `../configs/p4_frozen_ap/` |
| Stage-B selection (grid, per-arm best-epoch R@K/P@K, label-frame check) | `phase3a/{cat}_stageB_{V}_ap.json` |
| Stage-A retrieval gate (union ceilings, per-source R@100/500, content unique-hit share, memory gate) | `p4_stageA_ap/{cat}_gate.json`, built from `phase2_temporal/{cat}_candidates_report_{V}_ap.json` |
| Sequence experiment (3 preregistered variants × 3 categories + seed-43 noise arm) | `wave3_seq_v2/` — per-run reports, `NOISE_BAND.md`, `PREREGISTRATION_v2.md` |
| Calibration (fit on selection, one test application) | `calibration/{cat}_{winner}_ap_{fit,frozen,test}.json` |
| Literature anchor (SASRec et al. under the standard 5-core LOO protocol) | `anchor/COMPARISON.md`, `anchor/ANCHOR_MEMO.md`, per-leg `anchor/{cat}/*.json` |
| Label-frame objective comparison (first vs all positive, horizon windows) | `label_modes/*.json` |
| Content-source go/no-go probes (pessimistic lower bound on poisoned embeddings) | `phase3a/{cat}_content_probe.json` |
| Pre-content retrieval scan that selected the K=500 union | `phase3a/{cat}_stageA.json`, `phase2_temporal/*_C2_k500_recent*.json`, `phase3a/frozen_config.json` (that phase's freeze) |
| First temporal baseline (two-snapshot era) | `phase1_temporal/*.json` |
| Dataset EDA stats used by the data-layer docs | `stream_stats_*.json`, top-level CSVs |

## Reading rules

- The only headline metric is `Recall@K` on the **all** ground-truth-user
  denominator (`overall`); `conditional` and `listed` variants are
  diagnostics and move inversely to quality when retrieval shrinks.
- `candidate_ceiling_macro_recall` is the exact upper bound for macro
  Recall@K; `candidate_ceiling_retrieved_coverage` is any-hit user coverage
  (diagnostic only — with several positives per user it overstates the
  reachable recall).
- Multi-positive Precision@K is `hits@K / K` per user, averaged; it is NOT
  Recall/K.
- AUC/gAUC: standard tie treatment (0.5 credit). Top-K ranking: deterministic
  conservative ties (tied negatives first). Computed separately by design.
- The superseded first-positive freeze lives in
  `archive/results_invalidated_first_positive/frozen_config_first_positive.json`;
  its diff against the live `p4_freeze/frozen_config_ap.json` is the recorded
  footprint of the label-frame incident.
