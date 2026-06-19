# Next-Stage Plan

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

## Immediate Tasks

1. Implement `<uq_state>` token extraction and unit tests.
2. Add deterministic image-corruption pairs to the training pipeline.
3. Audit Density UQ score changes for each corruption and severity.
4. Run P1 on 100-300 samples.
5. Run P2 only if P1 passes.

