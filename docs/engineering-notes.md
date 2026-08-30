# Engineering notes

A condensed, sanitized record of the incidents and lessons that shaped this
project. The point of keeping it public: the headline numbers are only as
good as the protocol, and most of the protocol here was earned the hard way.
(Identifiers, job numbers and operational chatter from the original lab
notebook are removed; the engineering content is unchanged.)

## Incidents that changed the system

**1. Future-popularity tie-breaking (early temporal phase).** The item table
was ordered by 2023 review counts and ties broke by row position, so cold
users' recommendation lists were effectively "future popularity charts".
Video_Games recall was inflated 3.3×. Fix: candidate pools sort
alphabetically before any positional tie-break, and no dump-time aggregate
(price, average rating, rating count) may appear in any feature. Lesson:
*the vocabulary and its ordering are features too* — time-slice them like
everything else.

**2. Over-merged item identities.** The v1 canonicalization merged items by
normalized title alone, folding ~206k cross-store Books groups (~418k ASINs)
into shared ids — different products, one identity. The v2 key is
(normalized title, store), validated by an audit that samples merge groups
and scores their genuineness before the rule is adopted. Net metric effect
of the repair was ≈ 0; the value is that the item space is now defensible.

**3. The label-frame defect (the big one).** The project's objective is
future-window multi-positive coverage (all positively-rated interactions), yet for one generation of the final
pipeline every ranker trained on first-positive labels: the candidate
builder wrote its label column from the single-positive ground-truth file,
and the trainer flag that selected the all-positive frame changed only the
evaluation denominators. Measured on a 3,000-user Books sample, 82% of the
future-positive candidate rows were labelled 0 — the model was taught that
most right answers were wrong. It was caught by external review of the
repository, not by the test suite, because every assertion at the time
checked file *presence*, not label *semantics*. The fix added: an explicit
`--label-mode` on the builder (labels, coverage ceilings and rule-weight
tuning all on one frame), a bidirectional training-time assertion (every
future positive in the table is labelled 1; every label-1 row is in the
ground truth), a printed positives-per-user figure in every log, and a full
re-run. Two of four categories changed their frozen decision under the
corrected labels. Lessons: *never trust a flag name — verify the frame on
disk*; *print the number that would be 1.000 if the bug existed*.

**4. Selection gains that die at test.** Twice, a selection-snapshot
improvement reversed on the locked test: a threshold relaxation
(+16% R@10 at selection, −8% at test) and a sequence-module vocabulary
frozen too early (48% of selection-window label items collapsed to `<unk>`,
69% at test — the model's calibration for "unknown item" broke as the
share drifted). Lessons: *single-metric selection gains of a few percent
are winner's-curse candidates — preregister a small set, define the noise
band with a second seed of the baseline, and never let the guardrail metric
be the one that picks the winner*; *any vocabulary must be rebuilt from the
exact data the training spec allows, no earlier*.

**5. Ablations must change one thing.** The first sequence-module design fed
candidate-id embeddings, a new head and the sequence representation into one
model — when it beat the baseline, nothing could be attributed. The final
design is a residual head (`logit = DCN + α·g(seq, cand)`, α init 0) that
reproduces the baseline exactly at initialisation, asserted by a test.

**6. Provenance is part of the result.** The freeze step is a pure-read
script over the Stage-B reports with a pre-declared decision rule; it
records source paths, hashes and metrics for every input, and fails loudly
if an input is missing. That guard exists because a test-only builder run
once silently overwrote a Stage-A report and a later regeneration recorded
a candidate-row count of zero; reports now merge instead of clobbering.

## Operational lessons (cluster / harness)

1. A job can finish its work and still die (or hang) in teardown — accept
   artifacts by verifying them, not by exit codes.
2. Batch nodes may lack tools the login node has (e.g. `git`); resolve
   HEAD from `.git` files inside jobs.
3. Non-interactive shell initialisation can kill `set -e` scripts; guard the
   sourcing and checkpoint stage boundaries with echoes.
4. Request resources the task needs, not a template: over-asking memory
   queues forever; pinning a specific GPU model queued for 40 hours while
   other models sat idle. Ask for `gpu:1` unless the code needs otherwise.
5. Accounting queries by bare job id can collide with other users' recycled
   ids — always filter by user and submit-time window in automation.
6. Long evaluation phases scale with *user count*, not catalog size; the
   category with the smallest catalog had the slowest jobs.
7. Peak memory extrapolation must divide by the rows the trainer actually
   consumed, not a smaller sibling artifact — the first projection was 7×
   off and mislabeled every configuration as infeasible.
8. Every quantitative claim in a memo should name the file it came from;
   sampled measurements must be labelled as samples. Unverified summaries
   of parallel work are hypotheses, not results.
9. Metrics with inverted incentives (conditional hit rate rises when
   retrieval shrinks) belong in diagnostics, never in headline tables.
10. When two artifact generations coexist, separate them by name (`_ap`
    suffix, distinct weight files) and make the loader refuse mismatches;
    "remember to run the extra step" is not a protocol.

## What we would do differently

Start with the multi-positive objective enforced by assertions from day one;
build the anchor-to-literature experiment before tuning anything; give every
selection experiment a preregistered config list and a seeded noise band
from the start; and treat every "flag that switches the objective" as a
design smell — the objective should be a property of the artifacts, checked
at load time, not of the command line.
