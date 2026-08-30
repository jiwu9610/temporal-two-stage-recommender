# Project evolution

A milestone view of how this system was actually built, condensed from the
development history (April → August 2026). The public repository ships the
final, cleaned release; the full granular history — including superseded
experiments and their artifacts — lives in the private research repository,
with the invalidated generations quarantined under [`../archive/`](../archive/README.md).

A sanitized milestone-level development history — 24 real commits with their
original April–August dates, including the superseded first-positive
generation and the label-frame fix — is preserved directly in this
repository's commit history beneath the release commit.

## 2026-04 — First two-stage system
- Amazon Reviews 2023 ingestion, cleaning, and per-category preprocessing.
- Retrieval baselines (popularity, store/category rules, two-tower) unioned
  into per-user candidate sets; MLP and Deep & Cross rankers on a shared
  feature contract.
- Evaluated under a per-user leave-last-two split — later replaced when its
  leakage modes became clear.

## 2026-07 — Temporal evaluation rebuild
- Global walk-forward protocol: calendar cutoffs shared by every user,
  expanding histories, per-snapshot two-tower retraining, and a
  strictly-selection-then-one-shot-test discipline.
- Retrieval coverage diagnosis: candidate ceiling measurement, deeper
  per-source K (top-500), recency-popularity sources; configuration frozen
  before the phase's locked test.
- Evaluation & calibration layer: per-row prediction dumps, grouped
  multi-metric evaluation, Platt/prior calibration fitted on the selection
  snapshot only.

## 2026-08 (early) — Objective migration
- Switched the ground truth from "the user's next positive" to **every
  positively-rated interaction in the future window** (multi-positive), after
  auditing 33 single-positive assumptions across the codebase.
- Both label frames kept emittable so earlier numbers stay reproducible.

## 2026-08 (mid) — Identity repair, anchoring, retrieval/ranking waves
- Canonicalization v2: repaired ~206k over-merged Books item identities
  (title-only merge key → title + store), validated by a sampling audit.
- Literature anchor: SASRec / ItemKNN / popularity under the standard 5-core
  leave-one-out protocol; our SASRec landed inside the published reference
  band, establishing implementation health before any production conclusion.
- Two-tower v2 (in-batch softmax, history "taste" channel) selected and
  frozen per snapshot.
- Full retrieval sweep (6 configurations × 4 domains) → ranker selection →
  per-domain joint freeze → locked test.

## 2026-08-21 — Label-frame defect and re-run
- External review of the repository found that candidate labels were still
  written from the first-positive frame while evaluation used the
  multi-positive frame — later window positives were being trained as
  negatives.
- Fix: the label frame became an explicit builder argument; training now
  refuses to start unless candidate labels match the ground-truth frame
  (bidirectional assertion, logged every run); every downstream stage was
  re-run on the corrected frame. Two of four domains changed their frozen
  decision under the corrected labels.

## 2026-08 (late) — Corrected results and the sequence experiment
- All-positive Stage A/B re-runs, a pure-read freeze script with a
  pre-declared decision rule and hashed provenance, and a declared second
  locked test: **Books content retrieval +23.7% Recall@100**; threshold and
  content variants elsewhere rejected by guardrails or noise bands.
- Sequence modules re-evaluated with a clean residual design
  (`DCN + α·seq`, α₀ = 0), preregistered configurations, and a two-seed
  noise band → a validated negative: **not adopted**.
- Evaluation-semantics hardening: exact frozen-recipe consumption at test
  time, standard Mann–Whitney AUC tie handling, corrected multi-positive
  per-source precision, and release hygiene for this public repository.
