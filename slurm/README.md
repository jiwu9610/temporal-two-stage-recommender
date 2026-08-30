# Slurm drivers

Site-specific values are placeholders: set `--account` (YOUR_ALLOCATION) and
adjust `--partition` names to your cluster; the conda env is taken from
`${CONDA_ENV:-tae}`. GPU requests use `gpu:1` (any model) deliberately —
pinning a GPU model multiplies queue time for no benefit here.

Current pipeline: `p2_chain` (data layer) → `p4_k500_emit` (two-tower K=500
re-emit, GPU) → `p4_stageA` (retrieval sweep, CPU) → `p4_gate_full` →
`p4_stageB` (ranker selection, GPU) → freeze (`scripts/analysis/p4_freeze_ap`)
→ `p5_locked_test` (one-shot test, consumes the frozen ranker recipe) →
`w3_seq_selection` (sequence-module experiment). Superseded drivers are in
`../archive/slurm_legacy/`.
