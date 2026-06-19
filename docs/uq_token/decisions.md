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
