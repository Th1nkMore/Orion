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
