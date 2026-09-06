# Qwen visibility structured-grounding contract

Last updated: 2026-09-06 (Asia/Shanghai)

This document refines ADR-001 section 8. It fixes the V1 supervision semantics
before any Qwen parameters are optimized.

## Purpose and claim boundary

V1 asks whether the 4B VLM can read the accepted physical U tokens. It does not
yet train the Planning Expert and does not use collision, future outcome, or
hidden-actor labels. A Route 151 sparse overfit is permitted only as a plumbing
and capacity check; it is not held-out grounding or safety evidence.

The composite diagnostic answer contains exactly four fields:

```json
{"frontier":"F07","route":"ON_ROUTE","margin":"NEAR","action":"SLOW"}
```

- `frontier` identifies the valid frontier token with maximum
  `frontier_selection_score`;
- `route` thresholds its deterministic `route_weight_mean`;
- `margin` buckets its deterministic stopping margin as `INSIDE`, `NEAR`, or
  `CLEAR`;
- `action` is a conservative longitudinal category, not a direct actuator
  command or a hidden-actor prediction.

The physical U estimator remains task-agnostic. Route and stopping exposure are
computed separately and already exist as inspectable input features. The VLM
is trained to interpret these inputs; the labels do not add semantic knowledge
about an unobservable actor.

The first warm-up now presents `frontier`, `route`, `margin`, and `action` as
four balanced short-answer tasks (`Fxx`, `ON_ROUTE/OFF_ROUTE`,
`INSIDE/NEAR/CLEAR`, and `KEEP/SLOW/STOP`). This was added after the valid V1c
negative result showed that composite JSON cross-entropy learned shared syntax
and majority fields without a true-versus-shuffle gap. The label semantics and
thresholds below are unchanged. Composite JSON remains an evaluation and later
mixed-training target after the four fields are individually grounded.

## Fixed v1 target thresholds

- `ON_ROUTE` when `route_weight_mean >= 0.2`;
- `INSIDE` when normalized stopping margin is at most zero;
- `NEAR` when the positive physical margin is at most 5 m;
- `CLEAR` otherwise;
- `STOP` for an on-route frontier inside the stopping envelope;
- `SLOW` for an on-route near frontier, or an on-route frontier whose local
  maximum urgency is at least 0.1;
- `KEEP` otherwise, including close but route-irrelevant hard negatives.

These thresholds generate supervision and must be reported as such. They are
not learned safety guarantees and do not replace the later trajectory teacher.

## Anti-shortcut permutation

O3 serializes frontier rows in descending selection-score order. Without an
intervention the primary frontier is always `F00`, so an identity task would be
vacuous. Each V1 record therefore stores a seeded permutation mapping new row
index to original row index. The trainer applies it only to valid frontier
rows and updates the label to the new row. Global tokens retain their fixed
spatial raster order, padded rows remain padded, and no physical feature is
changed.

This permutation is training augmentation, not the O3 spatial-shuffle causal
control. Spatial shuffle deliberately misaligns physical content and metric
slots; the anti-shortcut permutation merely changes the arbitrary sequence
order of complete frontier records.

## Initial Route 151 plumbing data

The first manifest joins the five immutable sensor-audit steps with their O3
true-U artifacts. Each record contains three native 1600x900 current RGB views,
the token and image SHA-256 digests, the recorded permutation, the canonical
answer, and the exact numeric evidence used to derive it.

All records are marked `plumbing_overfit_train_eval` and
`reportable_generalization=false`. Zero-U and spatial-shuffle controls are not
optimizer examples; they remain evaluation interventions. The released
Planning Expert is absent from the optimizer.

## V1 optimization and acceptance gates

The first optimization probe must:

1. freeze the vision encoder, embeddings, LM head, lower VLM layers, and the
   complete Planning Expert;
2. train the physical-token projector and LoRA adapters in declared upper VLM
   modules only;
3. compute cross-entropy only on the declared categorical or canonical
   assistant answer, with equal optimizer coverage for all four fields;
4. prove nonzero finite gradients reach both the projector output and at least
   one LoRA tensor before the first optimizer step;
5. save only adaptation weights, optimizer-independent configuration,
   provenance, and an integrity hash—never a copied 4B base checkpoint;
6. evaluate exact one-field answers during the factorized warm-up, then exact
   JSON and per-field accuracy when the composite task is restored; record
   paired zero-U and spatial-shuffle outputs without training either control;
7. keep V1 open if only the Route 151 overfit succeeds. Acceptance as a learned
   consumer requires a separately generated, non-evaluation-route grounding
   set with held-out scenes and an above-chance true-versus-control gap.

### V1d factorized plumbing gate

The V1d probe contains all 20 `(five frame, four field)` examples and takes
exactly 60 optimizer steps, giving every pair three updates. Each true-U field
must reach at least `4/5` exact accuracy. The causal checks reflect what each O3
control actually changes:

- `frontier`: true U must beat the stronger of zero U and spatial shuffle by at
  least `2/5`, because spatial shuffle moves the maximum-score content to a
  different metric slot;
- `margin`: true U must beat zero U by at least `2/5`;
- `action`: true U must beat zero U by at least `2/5`;
- `route`: accuracy must reach `4/5`, but this sparse set cannot support a
  causal route gap because every target is `ON_ROUTE`.

Spatial shuffle moves each frontier's content features together. It should
therefore change the `frontier` sequence identity while preserving the selected
record's route, margin, and action values. Requiring those three values to
degrade under shuffle would reward scientifically incorrect behavior. The
missing `OFF_ROUTE` class remains a declared dataset limitation.

Passing this gate establishes only that the small Route 151 overfit can read
the intended token fields causally. It remains non-reportable plumbing evidence
and does not accept V1, planning behavior, or safety.

V1d produced a protocol-valid negative result: separating the fields removed
JSON-syntax dominance but still allowed fixed image identity, fixed frontier
permutation, and majority-label shortcuts. The next bounded curriculum must
therefore create multiple questions and labels from different real frontier
rows of the same image/token set. Route, margin, and action supervision remains
derived from the unchanged physical features and thresholds above; no hidden
actor label or synthetic final action may be introduced. More epochs on the
unchanged one-image/one-target V1d protocol are not an accepted next probe.

### V1e row-addressed anti-shortcut gate

V1e may reuse the same immutable true-U token artifact under multiple complete
frontier-row permutations. It may ask for the maximum-score row or ask for a
declared row's route, margin, or action field. A row permutation must move all
23 features together; altering individual true-U features would create new
physical evidence and is prohibited.

For each row-addressed field, one fixed query slot receives different real rows
and balanced labels under the same source images. Every selected example must
also have a different label under its paired O3 spatial-shuffle control. The
curriculum stores both control targets for audit, but neither zero U nor
spatial-shuffle may occur in the optimizer schedule.

The bounded capacity gate requires at least 80% exact true-U accuracy per field
and per label, plus at least a 30-point gap between true U and the stronger of
zero U or spatial shuffle for every field. This closes the fixed-image,
fixed-slot, majority-label shortcuts exposed by V1d; it does not establish
held-out grounding, semantic relevance, planning quality, or safety.

V1e produced a protocol-valid negative result. True-U field accuracy was 40%
frontier, 80% route, 50% margin, and 66.7% action; the corresponding gaps over
the stronger zero/spatial-shuffle arm were 0, 10.0, 8.3, and 16.7 percentage
points. Every capacity check failed despite nonzero projector and LoRA updates
on all 360 steps. The balanced labels elicited more diverse answers than V1d,
but `F13` remained dominant, `INSIDE` was never correct, and target-changing
spatial shuffles usually did not change the prediction correctly.

One remaining confound is the deterministic cyclic ordering of all non-queried
rows: every selected row/label is paired with one repeatable surrounding token
order. Any next overfit diagnostic must keep complete physical rows intact,
place the same queried row under multiple independently randomized decoy-row
orders, and reserve unseen orders plus target-changing complete-row swaps for
evaluation. It must retain balanced labels and evaluation-only zero/shuffle
controls. More steps on the unchanged V1e schedule are not a new experiment.

### V1f held-out-order route-readout gate

V1f isolates the simplest local continuous-field readout before changing the
adapter. For each of the five immutable images, one real `ON_ROUTE` row and one
real `OFF_ROUTE` row form a matched pair at `F00`. Pair members contain the same
32 complete rows and differ only by one two-position row swap. Three independent
decoy-row orders per image are optimizer examples; two different orders per
image are evaluation-only. The question, image, query slot, and label balance
therefore cannot identify the answer, and a fixed global ordering cannot
transfer to evaluation.

The 30 training examples receive eight updates each for 240 balanced steps.
The 20 held-out-order examples and their zero/spatial-shuffle arms never enter
the optimizer. Acceptance requires all of the following:

- at least 90% true-U exact accuracy across held-out examples;
- at least 90% true-U accuracy for both `ON_ROUTE` and `OFF_ROUTE`;
- at least 80% of the ten held-out matched pairs correct on both members;
- at least 80% accuracy for the changed spatial-shuffle control target;
- at least a 30-point gap between true-U accuracy and the stronger zero/shuffle
  arm measured against the original true target.

This route-only result is a lower-bound capacity check. Passing does not imply
that frontier argmax, stopping margin, action composition, semantic relevance,
or planning are solved. Failing after a valid run rules out only the current
generic projector plus upper-layer-LoRA recipe under this bounded protocol.

V1f produced a protocol-valid negative result. Held-out-order true-U accuracy
was 75% (`ON_ROUTE` 90%, `OFF_ROUTE` 60%), only 50% of matched pairs were fully
correct, spatial-shuffle changed-target accuracy was 45%, and the true-U causal
gap was 20 points. The training loss fell and both adapter families updated on
every step, so this is not a broken-gradient or checkpoint result.

The unchanged generic whole-row MLP is no longer an accepted first adapter for
formal data generation. A replacement must make physical feature identity and
continuous value structure explicit without predicting semantic relevance or
driving action in the adapter. It must first rerun the held-out-order route
gate above; only after passing may the other grounding fields be restored.

### V1g typed-scalar adapter gate

V1g reuses the V1f matched pairs, unseen row-order split, optimizer schedule,
controls, and numerical gates without modification. Its sole experimental
change is the U projector. The 23 physical fields are not normalized against
one another. Each scalar occupies a deterministic field-specific block with
the fixed basis `[x, x^2, sin(pi*x), cos(pi*x)]` before learned mixing and Qwen
projection. This preserves absolute physical thresholds and exposes field
identity without assigning semantic relevance or an action inside the adapter.

The typed projector has exactly 1,367,040 trainable parameters and seven saved
tensors; Qwen upper-layer LoRA remains 393,216 parameters. All frozen-scope,
checkpoint, split-leakage, matched-pair, zero-U, spatial-shuffle, and claim
boundaries from V1f remain mandatory. A different dataset, extra epoch, changed
LoRA scope, or relaxed gate would invalidate the intended paired comparison.

V1g is a protocol-valid negative result. All true, zero, and spatial-shuffle
held-out answers collapse to `OFF_ROUTE`; true-U accuracy is 50%, matched-pair
accuracy 0%, changed spatial-shuffle target accuracy 50%, and causal gap 0.
Both trainable families update on every step and training loss becomes small,
so the failure is held-out causal readout rather than optimizer connectivity.

Field/value typing alone is not an accepted next adapter. Before another
grounding run, the insertion contract must resolve two separate questions:
whether every continuous row needs an explicit `Gxx/Fxx` identity, and whether
two adapted full-attention layers can route a named local row to the answer.
One experiment may change only one of those factors unless a preceding
read-only/interface test makes the other irrelevant.

### V1h explicit-slot adapter gate

V1h holds the V1g typed scalar basis and two-layer LoRA scope fixed. It appends
a deterministic 48-way one-hot slot code before the hidden projection, with
slots `0..15` mapped to `G00..G15` and `16..47` to `F00..F31`. This adds row
identity but no semantic judgment. The projector must contain exactly
1,391,616 trainable parameters and seven saved tensors.

The immutable V1f curriculum and all held-out-order gates are reused without
change. A passing result isolates slot addressing as necessary in this bounded
probe. A valid failure permits a later attention-scope experiment only if the
slot-typed adapter, data, seed, optimizer, and gates stay fixed.

V1h is a protocol-valid negative result. True-U held-out accuracy is 95%
(`OFF_ROUTE` 100%, `ON_ROUTE` 90%), matched-pair correctness is 90%, and the
true-versus-control gap is 40 points. Those gates pass, but the spatial-shuffle
arm reaches only 45% on its changed target and emits `OFF_ROUTE` for 19/20
examples. Slot identity is therefore accepted as a useful interface feature,
not as proof of causal readout. A V1i attention-scope probe must keep every
other V1h contract item fixed and adapt all eight released full-attention
layers. Until that gate passes, restoring the full frontier/margin/action
curriculum or activating Planning Expert injection is premature.

### V1i full-attention adapter gate

V1i retains the exact V1h slot-typed projector and immutable V1f
matched-pair/held-out-order curriculum. Its sole change is LoRA coverage: all
eight released full-attention layers `[3, 7, 11, 15, 19, 23, 27, 31]` replace
the prior `[27, 31]` scope. Rank, alpha, dropout, target modules, learning
rates, optimizer, clipping, seed, insertion contract, questions, controls, and
five numerical gates remain fixed.

The expected trainable scope is seven projector tensors with 1,391,616 values
and 64 LoRA tensors with 1,572,864 values across 32 wrapped attention modules.
The vision encoder, base VLM, embeddings, LM head, and Planning Expert remain
frozen. A passing result is only a causal route-scalar plumbing lower bound; a
failure is a bounded VLM-interface negative. The Planning Expert fallback
requires a later successful structured-grounding result followed by failed
trajectory conditioning, so V1i cannot activate it.

V1i is a protocol-valid negative. It reaches 100% true-U held-out accuracy,
100% matched-pair correctness, and a 50-point true-versus-control gap, but only
50% spatial-shuffle changed-target accuracy. Eighteen of twenty shuffled
outputs are `OFF_ROUTE`. All trainable tensors update and the expanded
eight-layer scope is independently verified, so repeating or broadening that
scope is not justified.

The V1f-V1i curriculum also shares each target row between training and
held-out-order evaluation; only decoy order is held out. On the ten distinct
true target rows, route labels are perfectly separable by several other
physical fields. The next route probe must therefore reserve target rows—not
only row orders—and include real counterexamples that break route-weight
correlation with urgency, unknown area, observation age, and frontier score.
The adapter must continue to expose physical fields without computing the
answer itself. Gates and causal controls remain fail-closed.

### V1j queried-target-row gate

V1j keeps the exact V1i model-side contract and changes only the immutable
curriculum. Thirteen real ON/OFF row pairs supply 78 training examples under
three decoy orders per pair. Five different ON/OFF row pairs supply 20
evaluation examples under two decoy orders per pair. Queried source rows are
disjoint between the two splits within every frame, and
`route151-step-000200` supplies no optimizer example at all. Evaluation rows
may have appeared only as unqueried decoys in other complete-row permutations;
this is target-row-disjoint plumbing, not unseen-value or broad generalization.

Only true U enters the optimizer. Zero U and spatial shuffle are evaluation
arms. Spatial shuffle must be reported, but its changed-target accuracy is not
a hard V1j gate. Acceptance requires at least 90% held-out true-U accuracy,
90% for each of `ON_ROUTE` and `OFF_ROUTE`, 80% jointly correct matched pairs,
and a 30-percentage-point true-U advantage over the stronger control on the
original target. Curriculum hashes, exact target-row allocation, the held-out
frame, complete permutations, schedule balance, train/evaluation leakage,
trainable scope, gradients, and adaptation-only checkpoint contents remain
fail-closed protocol checks.

V1j is a protocol-valid negative result. True-U accuracy is 10/20, with
`ON_ROUTE` at 4/10 and `OFF_ROUTE` at 6/10; matched-pair correctness is 0/10
and the true-versus-stronger-control gap is zero. The answer stays constant
within each frame when the queried F00 row swaps between ON and OFF. This
rejects the current route-readout recipe as local target-row grounding even
under oracle U. It does not establish that the 4B VLM can never consume U, but
the current structured-grounding prerequisite for trajectory training and the
Planning Expert fallback has not been met.
