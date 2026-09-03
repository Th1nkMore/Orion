# Decision Log

## 2026-06-20: Replace Hand-Crafted UQ Supervision

Decision:

Use a normal-feature density model instead of training the original UQEstimator
on manually designed pseudo-labels.

Reason:

The old target mixed token heuristics and weather-aware calibration, making the
uncertainty claim difficult to defend. The density model uses only normal
features for fitting and reserves weather labels for evaluation.

Result:

The selected 16-component shrinkage Mahalanobis model achieved route-disjoint
AUROC 0.799 and AUPRC 0.951.

## 2026-06-20: Reject 256 Active Density Components

Decision:

Use 16 active PCA components and preserve the existing 256-D interface by zero
padding.

Reason:

The 256-component model achieved AUROC 0.627. Screening showed that 16
components achieved the strongest route-disjoint result, indicating later
components captured nuisance route/content variation.

## 2026-06-20: Do Not Use FiLM L2 as the Primary Method

Decision:

Retain L2 as a baseline only.

Reason:

L2 modifies the planning representation after the LLM has already produced it.
It does not support the claim that uncertainty information is provided to and
used by the large model during decision formation.

## 2026-06-20: Use Explicit LLM Uncertainty Tokens

Decision:

Project density direction and score into continuous LLM tokens and append them
to the visual token sequence before LLM processing.

Reason:

This makes the scientific claim match the computation path: the LLM receives
perception reliability information before producing the waypoint
representation.

Alternatives:

- FiLM L1: retained as an indirect-conditioning baseline.
- Text labels such as low/medium/high uncertainty: optional interpretability
  baseline, rejected as the main method because of quantization and arbitrary
  thresholds.
- L2: retained as a post-LLM control baseline.

## 2026-06-20: Start with One Token

Decision:

Use `K=1` for the first main implementation.

Reason:

One token adds approximately 2.1M projector parameters, preserves the
lightweight narrative, and gives a clean test of whether explicit uncertainty
conditioning works. Four tokens remain a capacity ablation.

## 2026-06-20: Train LLM LoRA with the Projector

Decision:

The primary model trains the UQ projector and selected LLM LoRA adapters while
freezing base model weights.

Reason:

A completely frozen LLM may ignore a new continuous token. Projector-only
training remains an ablation to measure whether LLM adaptation is necessary.

## 2026-06-20: Reduce the VLM Loss Weight

Decision:

Use initial training weights:

```text
lambda_plan = 1.0
lambda_vae = 0.1
lambda_vlm = 0.001
lambda_consistency = 0.05
```

Reason:

With the inherited VLM weight of `0.01`, the weighted VLM term was approximately
`0.094-0.103`, while planning loss was approximately `0.003-0.010`. The
optimization target was therefore dominated by language modeling. Reducing the
VLM weight to `0.001` puts the weighted VLM term near `0.010`, comparable to the
planning term without removing language supervision entirely.

## 2026-06-20: Require Explicit UQ Grounding

Decision:

Pause full planning training and predict the fixed density score from the LLM
waypoint representation through an auxiliary grounding head.

Reason:

The base LLM weights are frozen, but LoRA adapters are trained and can learn a
functional use for the token. Even so, planning loss reduction alone cannot
show that the representation contains uncertainty information. The model might
use the token as an arbitrary latent variable or benefit from added capacity.

Correct sample-level UQ must outperform no-token, zero-token, and shuffled-token
controls before full training proceeds.

Consequence:

Full 8,316-frame training remains blocked until the grounding pilot passes.

## 2026-06-20: Require Same-Image Counterfactual Grounding

Decision:

Train the grounding head on both the correct score and a deterministically
shuffled score for the same image.

Reason:

Independent no-token, zero-token, shuffled-token, and correct-token models can
learn uncertainty directly from visual content. Their grounding metrics do not
isolate whether the injected token was used. Holding the image fixed and
changing only the score removes this visual shortcut.

Result:

The score-only 100-step pilot achieved Pearson 0.753 with the correct token and
-0.005 after shuffling. This passes the causal-use test.

## 2026-06-20: Ground Score Before Adding Density Direction

Decision:

Use the scalar density score alone for the current grounding gate. Keep the
16-D direction branch disabled until score grounding is calibrated.

Reason:

The score has a clear supervised meaning and supports a direct intervention.
The earlier score+direction run showed strong correct-token correlation but
worse held-out MAE, leaving direction as a possible source of route-specific
shift. Direction remains a later ablation.

Consequence:

The initial 100-step run passed only the causal gate. A 300-step follow-up on
200 calibration frames reduced correct-token MAE to 0.0406, versus 0.1087 for
no token, 0.1129 for zero token, and 0.2940 for shuffled token. Both grounding
gates now pass.

## 2026-06-20: Reject Grounding-Only Weights as Planning Initialization

Decision:

Use the grounding-only run as a causal diagnostic, but start formal joint
training from base ORION.

Reason:

The grounding-only checkpoint achieved excellent score recovery but produced
ADE 1.8923 with the correct token, versus 0.0815 with no token and 0.0803 for
base ORION. It encoded score in a trajectory-destructive direction. A later
50-step planning pilot recovered ADE to 0.0894, confirming that planning
supervision can reshape the representation, but the two-stage transition is
unnecessarily unstable.

Consequence:

Train correct-token planning, correct-token grounding, and same-image shuffled
grounding together from the start. Retain low-UQ consistency. Full-scale
training remains blocked until this joint pilot preserves baseline ADE/FDE and
retains the grounding intervention gap.

## 2026-06-20: Use Grounding as an Auxiliary, Not Dominant, Objective

Decision:

After the initial joint warm-up, reduce `lambda_ground` from `1.0` to `0.2`
while retaining same-image counterfactual grounding.

Reason:

The 100-sample joint pilot retained strong score correlation, but correct-token
ADE/FDE were 0.0941/0.1070 versus base ORION at 0.0803/0.0745. Per-step logs
showed the weighted grounding term was often larger than planning. The model
used the token, but had not learned a beneficial planning response after only
25 optimizer updates.

Consequence:

Continue the same optimizer state to 400 effective samples with planning
dominant. Stop scaling if correct UQ still degrades trajectory metrics.

## 2026-06-20: Stop Waypoint-State Grounding

Decision:

Do not scale the current joint model. Replace grounding from the waypoint
hidden state with grounding from a dedicated LLM UQ readout token.

Reason:

At 400 effective samples, correct-token Pearson remained 0.879 and shuffled
Pearson fell to 0.089, proving causal token use. Planning nevertheless worsened:
correct ADE/FDE reached 0.1046/0.1162 versus base ORION at 0.0803/0.0745.
Lowering the grounding weight did not fix the conflict.

The current auxiliary head explicitly rewards the trajectory decoder's input
for being linearly predictive of uncertainty. This is stronger than the
research claim requires and can distort the planning representation.

Consequence:

Use a separate `<uq_state>` hidden representation for score reconstruction.
Keep the waypoint representation under task supervision. Demonstrate planning
use through correct/zero/shuffled trajectory interventions rather than by
forcing score reconstruction from the waypoint state.
