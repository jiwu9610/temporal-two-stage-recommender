# Archive

Superseded material kept for provenance. Nothing here backs a current claim;
the current results live under `results/` and are indexed by
`results/REBUILD_WAVE_REPORT.md`.

- `results_invalidated_first_positive/` — artifacts from runs whose ranker
  TRAINING labels were on the first-positive frame while evaluation used the
  all-positive frame (a disclosed protocol defect, see the report's incident
  section). **The JSONs in here self-report `"label_mode": "all_positive"`;
  that field describes the evaluation side only and is exactly why the defect
  went unnoticed — do not compare these numbers with anything.** All of them
  were re-run on the corrected pipeline; the valid counterparts carry the
  `_ap` suffix under `results/`.
- `results_legacy_phases/` — earlier project phases (EDA, leave-last-two
  baselines, the first temporal generation, the phase-3A retrieval sweep and
  the two-tower v2 selection arms). Valid for what they were; superseded by
  the temporal all-positive protocol.
- `slurm_legacy/`, `configs_legacy/` — drivers/configs of those phases.
  `configs_legacy/p4_frozen_first_positive/` differs from the current
  `configs/p4_frozen_ap/` by a single integer per category — that delta IS
  the label-frame incident's footprint on the frozen decision.
- `notebooks/`, `figures/` — April-era EDA and rating-model material,
  unrelated to the final two-stage system.
