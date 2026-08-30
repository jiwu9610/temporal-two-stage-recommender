# Wave-3 (sequence module at the ranking position) — preregistration v2

Written 2026-08-21 BEFORE any all-positive wave-3 run. Supersedes the 72-cell
sweep (invalid: first-positive training labels; T0-frozen vocab; non-clean
ablation). External review #2 items adopted.

## Model
`logit = DCN(x0) + alpha * g([cand_emb, seq_out, cand_emb*seq_out])`, alpha init 0.
DCN = frozen DeepCrossRanker layout. Test 19 asserts exact equality at alpha=0.
Item vocab: selection phase from ranker_train history; refit from rt+ms
histories (items first seen in [T1,T2) are <unk> at test; OOV rates reported).
Engagement = every pre-cutoff interaction (rating as sparse id), last 20.

## Preregistered configurations (3 per category, no further search)
1. `seq:hstu:pos=delta`          (HSTU block, gap-bucket position emb)
2. `seq:mh_pool:pos=delta:L=2`   (2-layer encoder + target-aware multi-head pooling)
3. `seq:causal:pos=delta`        (SASRec-style causal encoder, last state)
Each at lr {1e-3, 3e-4}; epoch by early stopping on model_selection (all-positive frame).
Categories: Books, Electronics, Video_Games on their all-positive P4.4 winners (`_ap`).
All_Beauty: out of scope (popularity baseline).

## Decision rule (fixed in advance)
Primary: model_selection R@100 macro vs the in-run deep_cross arm.
Guardrail: R@10 no regression vs deep_cross.
Required: lift reported on history cohorts 0 / 1-2 / 3-4 / 5-9 / 10+; a config
is adopted for a category only if overall R@100 >= deep_cross AND the 5-9 and
10+ cohorts do not regress. Otherwise the category keeps DCN (and a gated
hybrid — DCN below a history threshold, DCN+seq above — is evaluated
post-hoc as a diagnostic only, not as the frozen choice).
Locked test: ONE read per adopted config; it follows the all-positive P5.
Noise floor: two seeds of the deep_cross arm define the ±band; differences
inside it are reported as ties.

## Post-audit notes (2026-08-25, review #3)
- Known limitation, fix BEFORE any adopted seq config is ever test-run: the
  refit vocabulary uses rt+ms histories; items first observed in [T1,T2)
  (legal at T2) still map to <unk> at test time.
- Noise-band provenance: results/wave3_seq_v2/NOISE_BAND.md (both seeds'
  reports are committed; later stage-b reports record `seed`).
