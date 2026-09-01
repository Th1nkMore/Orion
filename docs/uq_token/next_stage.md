# Next-Stage Plan

> **Historical document (superseded 2026-08-29).** This plan describes the earlier scalar Density-UQ / explicit-token / pre-LLM adapter line. It is retained as experiment history, not current execution authority. The active status is [../CURRENT_STATE.md](../CURRENT_STATE.md), and the active architecture contract is [../spatial_uq_two_stage_v2.md](../spatial_uq_two_stage_v2.md).

## Current Diagnosis

The experiments establish three facts:

1. Density score is a usable sample-level signal.
2. A continuous UQ token can be read causally by the LLM.
3. Reconstructing score from the waypoint hidden state damages the trajectory
   representation and does not improve open-loop planning.

The remaining problem is not token visibility. It is supervision. The current
expert trajectory does not specify a different desired action when perception
is uncertain. Without outcome-aware or paired-corruption supervision, the
planner can rationally ignore UQ. Forcing score into the waypoint state is not
a valid substitute for this missing behavioral signal.

## Primary Route: Separate Readout and Paired Robustness Training

### Architecture

Use two distinct LLM output positions:

```text
continuous UQ input token -> LLM

<uq_state> hidden state -> score grounding head
<waypoint> hidden state -> unchanged trajectory decoder
```

The grounding head must never consume the waypoint hidden state.

### Training Data

Construct paired views from each valid training frame:

```text
clean image -> original expert trajectory
corrupted image -> same expert trajectory
```

Start with controlled corruptions that affect visual reliability but preserve
scene geometry:

- fog and contrast reduction;
- rain overlay;
- Gaussian blur and motion blur;
- exposure shift;
- partial camera dropout or occlusion.

Keep only corruption settings for which the Density UQ score increases
consistently over the clean view. This avoids assuming that every synthetic
effect is captured by the current density estimator.

### Objectives

```text
L = L_plan(clean)
  + L_plan(corrupted)
  + lambda_pair * distance(
        waypoint(corrupted, correct_uq),
        stopgrad(waypoint(clean, clean_uq))
    )
  + lambda_ground * L_ground(<uq_state>)
  + lambda_cf * L_ground_counterfactual(<uq_state>)
  + lambda_vlm * L_vlm
```

This does not prescribe a hand-designed trajectory modulation. It asks the LLM
to use reliability information to preserve expert behavior when perception is
degraded.

### Phase Gates

#### P0: Plumbing

- add `<uq_state>` extraction;
- verify waypoint output is numerically unchanged with UQ disabled;
- verify grounding gradients reach `<uq_state>`, projector, and LoRA;
- verify the grounding head has no direct path from waypoint state.

#### P1: Semantic and Preservation Pilot

Train 100-300 effective samples with correct/shuffled score pairs.

Proceed only if:

- correct grounding Pearson exceeds shuffled by at least 0.3;
- no-token and disabled-UQ ADE remain within 3% of base ORION;
- correct-token ADE does not exceed base ORION by more than 3%.

#### P2: Paired-Corruption Pilot

Use 500-1,000 clean/corrupted pairs.

Proceed only if:

- correct UQ improves corrupted-view ADE or FDE over zero and shuffled UQ;
- clean-view ADE degradation remains at most 3%;
- improvement is larger in the high-UQ half;
- stopped behavior and trajectory magnitude do not collapse.

#### P3: Scale

Only after P2 passes:

- train on all route-disjoint training frames;
- evaluate normal/adverse weather and UQ quartiles;
- add collision, progress, and comfort metrics;
- run route-bootstrap confidence intervals.

## Behavioral Route After Paired Robustness

Paired expert trajectories teach robustness, not risk-sensitive conservatism.
If the final claim requires the LLM to slow down or change policy under
uncertainty, use outcome-based supervision rather than hand-designed L2
modulation:

```text
collision / safety cost
+ progress reward
+ comfort cost
+ route completion
```

The LLM still decides how UQ affects the waypoint. Progress and comfort terms
are required so collision reduction cannot be achieved by stopping.

This stage should use closed-loop simulation or a teacher planner. Open-loop
expert trajectories alone cannot establish that uncertainty-aware
conservatism is beneficial.

## Fallback A: Textual Reliability Prompt

If continuous tokens remain difficult to align, expose calibrated reliability
through the LLM's native interface:

```text
Visual reliability percentile: 23/100.
Perception confidence is low.
```

Use calibration percentiles rather than arbitrary low/medium/high thresholds.
This sacrifices density direction but provides an immediately interpretable
semantic baseline and tests whether the LLM can use uncertainty expressed in
language.

Acceptance is still based on correct/zero/shuffled prompts and paired
corruptions. A text prompt is not accepted merely because grounding succeeds.

## Fallback B: Pre-LLM Uncertainty Adapter

If explicit LLM tokens fail but UQ remains useful, use FiLM L1 or a small
uncertainty-conditioned cross-attention adapter before the LLM:

```text
EVAViT / QT-Former tokens
  -> UQ-conditioned adapter
  -> LLM
  -> waypoint
```

This does not bypass the LLM. It changes the evidence presented to the LLM,
while the LLM still forms the planning representation. It is scientifically
weaker than explicit semantic tokens, but substantially more defensible than
post-LLM L2 modulation.

## Fallback C: Monitoring-Only UQ

If no conditioning method improves planning under correct-versus-shuffled
controls, retain Density UQ as a calibrated monitor:

- detect distribution shift;
- stratify planning risk;
- trigger fallback policy or human review;
- report uncertainty without claiming end-to-end adaptation.

This remains a valid contribution if the density estimator predicts failure or
adverse conditions reliably. It is preferable to claiming an ineffective
conditioning mechanism.

## Stop Rules

Stop the explicit-token route if either condition holds after two controlled
architecture pilots:

1. correct UQ does not outperform shuffled/zero UQ on corrupted-view planning;
2. clean-view ADE degrades by more than 3% despite preservation losses.

Then run one textual-prompt pilot and one pre-LLM adapter pilot. If neither
passes, use monitoring-only UQ as the final method and report the negative
conditioning result.

## 2026-06-21 Update

The explicit-token route reached the stop condition:

1. consistency-only paired training made correct UQ worse than none and
   shuffled on one-camera-dropout views;
2. correct-versus-shuffled ranking was active during training but correct UQ
   still produced the worst corrupted-view ADE.

Do not launch a larger explicit-token planning run. The immediate behavioral
experiment is now Fallback B: a small score-conditioned adapter before the
LLM, initialized as identity and trained with clean/corrupted trajectory
preservation. R2d remains the separate interpretability result.

### Adapter Pilot Outcome

The 100-step adapter-only pilot passes the minimum gate on the first 50 frames
of Route1115 and on an independent Route504 segment:

- correct UQ outperforms shuffled and none under camera dropout;
- clean ADE does not degrade;
- Route504 clean ADE and FDE both improve.

The harder later section of Route1115 does not improve. The next run must use
route-balanced sampling across all calibration routes instead of the first N
sequential frames.

## 2026-06-24 Mid-Report Note

For the mid-report evidence chain, the newest controlled localization pilot is
recorded in:

```text
reports/risk_qa/embedding_localization_pilot.md
/Users/th1nkmore/th1nkmore_ws/sustech-master-thesis/docs/progress/mid_report_writing_plan.md
```

The result should be treated as a positive semantic-alignment pilot rather than
a completed planning result. Active UQ embedding reaches 61.0% three-column
front-camera localization accuracy on a 100-sample held-out split, above the
45.0% majority baseline, and correct-versus-shuffled UQ changes the answer on
69.4% of eligible samples. Cross-split calibration remains unstable, so the
next claim should focus on local UQ readability and controllability, not final
planning safety.

### 2026-06-23 Route-balanced Update

The route-balanced evaluation has now been run on 10 calibration routes with
50 candidate frames per route and 350 valid planning frames under one-camera
dropout severity 1.

Aggregate corrupted-view planning:

```text
none     ADE 1.4429, FDE 2.5210
zero     ADE 1.4429, FDE 2.5210
shuffled ADE 1.2286, FDE 2.1969
correct  ADE 1.1641, FDE 2.0955
```

This supports the adapter route:

- correct improves over none by 19.3% ADE;
- correct improves over shuffled by 5.3% ADE;
- correct improves over none on 9/10 routes;
- correct improves over shuffled on 8/10 routes.

However, the high/low UQ split did not pass the earlier "high-UQ benefit is
larger" expectation:

```text
high-UQ correct vs none:     +10.2%
high-UQ correct vs shuffled: +1.6%
low-UQ correct vs none:      +16.7%
low-UQ correct vs shuffled:  +3.4%
```

Interpretation:

The score is useful as a conditioning signal, but scalar score magnitude is not
yet calibrated as a monotonic predictor of planning benefit. The immediate
research question shifts from "does the adapter work at all?" to "how do we
make UQ semantics and planning benefit better aligned?"

Clean safety has also been checked:

```text
clean none     ADE 0.8884, FDE 1.3707
clean shuffled ADE 0.9008, FDE 1.4265
clean correct  ADE 0.8971, FDE 1.4248
```

Clean ADE degradation is about 1.0%, which passes the 3% gate. Clean FDE
degradation is about 3.9%, slightly above the strict gate. This is not a stop
condition for the adapter route, but it means the next run must include an
explicit clean preservation term.

Updated next steps:

1. Route-balanced training: train 500-1,000 paired samples sampled across
   routes, not sequential first-N frames.
2. Add clean preservation: keep clean correct/none ADE and FDE within 3% via
   identity regularization or clean consistency loss.
3. Richer conditioning: evaluate score + active density embedding in the
   vision adapter, because score alone may be too coarse to capture which
   visual evidence is degraded.
4. Stratified diagnostics: split by route type and weather, not only by scalar
   UQ median. The current high/low split mixes routes with very different
   baseline difficulty.
5. Keep explicit-token planning stopped. R2d remains the interpretability
   result; planning claims should use the pre-LLM adapter route.

## Immediate Tasks

1. Implement `<uq_state>` token extraction and unit tests.
2. Add deterministic image-corruption pairs to the training pipeline.
3. Audit Density UQ score changes for each corruption and severity.
4. Run P1 on 100-300 samples.
5. Run P2 only if P1 passes.
