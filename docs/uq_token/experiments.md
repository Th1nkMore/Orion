# Experiment Plan

## Split Protocol

Use route-level splits only. No frame-level random split is permitted.

Current 50-route validation split:

```text
30 routes: training
10 routes: calibration / model selection
10 routes: final test
```

The current split assignment is stored inside:

```text
checkpoints/density_uq/best.pt
```

The final test routes must not be used to select projector architecture,
learning rate, epoch, LoRA rank, token count, or regularization.

For a publishable final result, prefer fitting and training on the official
Bench2Drive train split and using the validation split only once for final
evaluation.

## Stage 2A: Online Density UQ Audit

Purpose: verify that online ORION inference reproduces the offline density model
before adding trainable conditioning.

Required outputs:

- online/offline score agreement;
- score distribution by scene type and Weather ID;
- AUROC and AUPRC on route-disjoint test routes;
- runtime and peak-memory overhead;
- correlation with ADE, collision metrics, braking, and speed;
- high- and low-score qualitative examples.

Acceptance:

- no NaN or Inf;
- online and offline score difference below numerical tolerance;
- output shapes remain `[B, 256]` and `[B, 1]`;
- Density UQ alone does not alter ORION planning output.

## Stage 2B: UQ Token Smoke Training

Purpose: establish that gradients flow through the UQ projector and LLM LoRA.

Configuration:

```text
train routes: small subset of 2-4 routes
epochs: 1
token_count: 1
LoRA: enabled
Density UQ: frozen
```

Checks:

- only intended parameters receive gradients;
- loss decreases over a short run;
- checkpoint reload reproduces output;
- token norm and attention remain finite;
- baseline path is unchanged when UQ tokens are disabled.

This run is an engineering test and must not be reported as a result.

## Stage 2C: UQ Grounding Pilot

Purpose: test whether the LLM waypoint representation encodes the supplied
density score before optimizing full planning behavior.

| ID | Input | Trainable components | Target |
| --- | --- | --- | --- |
| G0 | No UQ token | LoRA + grounding head | Density score |
| G1 | Zero/null UQ token | Projector + LoRA + head | Density score |
| G2 | Shuffled UQ token | Projector + LoRA + head | Original sample score |
| G3 | Correct UQ token | Projector + LoRA + head | Density score |
| G4 | Correct UQ, frozen LLM | Projector + head | Density score |

Required metrics:

- score MAE and SmoothL1;
- Pearson and Spearman correlation;
- normal/adverse AUROC as a secondary diagnostic;
- route-bootstrap confidence intervals.

Minimum condition for proceeding:

```text
G3 clearly outperforms G0, G1, and G2 on calibration routes.
```

If shuffled UQ performs similarly to correct UQ, full training remains blocked.

## Stage 2D: Planning Adaptation

Initial hyperparameters:

```yaml
token_count: 1
projector_hidden_dim: 512
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
learning_rate_projector: 1.0e-4
learning_rate_lora: 2.0e-5
weight_decay: 1.0e-4
epochs: 6
gradient_accumulation: 8
gradient_clip: 1.0
lambda_collision: 1.0
lambda_consistency: 0.05
lambda_plan: 1.0
lambda_vae: 0.1
lambda_vlm: 0.001
lambda_ground: selected by grounding pilot
early_stopping_patience: 2
```

These values are starting points, not fixed conclusions.

The VLM weight was reduced from the inherited `0.01` to `0.001` after a
diagnostic run showed that weighted language loss dominated the planning loss.
With `0.001`, weighted VLM and planning terms are of comparable order on the
observed smoke samples.

## Counterfactual Evaluation

For a fixed image and prompt, change only the UQ input:

```text
score = 0.0, original direction
score = 0.5, original direction
score = 1.0, original direction
normal image + adverse-sample UQ
adverse image + normal-sample UQ
```

Measure predicted grounding score, waypoint hidden-state displacement,
trajectory displacement, braking, speed, and stopped behavior. Aggregate
response should be systematic and stronger than zero/shuffled controls.

## Main Experiment Matrix

| ID | Method | UQ enters LLM | Trainable components | Role |
| --- | --- | --- | --- | --- |
| E0 | ORION baseline | No | None | Reference |
| E1 | Density UQ monitor | No | None | Verify zero behavioral effect |
| E2 | FiLM L2 | No | L2 FiLM | Post-LLM control baseline |
| E3 | FiLM L1 | Indirectly | L1 FiLM | Visual-feature modulation baseline |
| E4 | UQ token, frozen LLM | Yes | Projector | Projector-only ablation |
| E5 | UQ token + LLM LoRA | Yes | Projector + LoRA | Primary method |
| E6 | Shuffled UQ token | Incorrect | Same as E5 | Tests sample-level information |
| E7 | Zero/null UQ token | No effective UQ | Same parameter structure | Extra-parameter control |

## UQ Token Ablations

Run only after E5 is stable:

| Ablation | Values |
| --- | --- |
| Token input | score only / direction only / score + direction |
| Token count | 1 / 4 |
| Gating | none / score gate / learned null token |
| LoRA | off / rank 4 / rank 8 |
| Consistency loss | off / 0.01 / 0.05 |

Do not launch the full grid. Use calibration routes to eliminate clearly weak
settings, then evaluate only selected configurations on final test routes.

## Metrics

Planning quality:

- ADE and FDE at available horizons;
- collision rate;
- route completion or progress where available;
- throttle, brake, and steer MAE;
- mean speed;
- stopped-frame ratio;
- excessive-braking ratio.

Uncertainty behavior:

- score distribution;
- performance by UQ quartile;
- performance by normal/adverse split;
- performance by Weather ID;
- token norm;
- attention paid to the UQ token;
- change in waypoint-token representation relative to baseline.

Efficiency:

- trainable parameter count;
- inference latency;
- peak GPU memory;
- training time.

## Main Acceptance Criteria

The primary model should satisfy all of:

1. Normal-scene ADE degradation no greater than 3% relative to baseline.
2. Adverse-scene collision rate improves without a material rise in stopped
   ratio or excessive braking.
3. Improvement is stronger in high-UQ groups than low-UQ groups.
4. Correct UQ outperforms shuffled and zero/null UQ controls.
5. UQ token + LoRA outperforms projector-only conditioning.
6. The LLM shows measurable attention or representation response to the UQ
   token.
7. Correct-UQ grounding demonstrates that the waypoint representation retains
   sample-level density information.

## Failure Interpretation

- E4 and E5 both fail: token placement or supervision is ineffective.
- E4 fails but E5 succeeds: LLM adaptation is necessary, as expected.
- E5 equals shuffled UQ: gains come from added capacity, not uncertainty.
- E5 improves collision but increases stopped ratio: conservative shortcut.
- L1 succeeds while tokens fail: uncertainty is useful but current LLM token
  interface or training objective is inadequate.
- L2 alone succeeds: direct control is effective, but it remains a weaker
  scientific narrative and should be reported as such.
- Grounding succeeds but planning does not: the LLM receives UQ, but the current
  planning objective or downstream decoder does not use it effectively.
- Planning improves without grounding: treat the result as inconclusive because
  the gain may come from latent capacity rather than uncertainty.
