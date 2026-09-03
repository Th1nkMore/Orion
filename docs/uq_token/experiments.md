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
G3 clearly outperforms G0, G1, and G2 on calibration routes in correlation,
and does not lose to G0 in held-out MAE.
```

If shuffled UQ performs similarly to correct UQ, full training remains blocked.

### Pilot Results

Independent correct/zero/shuffled/no-token training was inconclusive because
the grounding head could predict uncertainty from the image itself. A
counterfactual score-only run was therefore added: each image is trained with
both its correct score and another sample's shuffled score.

The 100-step result on 50 calibration frames was:

| Intervention | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.1158 | 0.4135 | 0.4479 |
| zero | 0.1318 | 0.4048 | 0.4734 |
| shuffled | 0.2742 | -0.0055 | 0.1052 |
| correct | 0.1501 | 0.7531 | 0.6414 |

Interpretation:

- The collapse from correct to shuffled proves sample-level token use.
- Correct UQ substantially improves correlation over no-token and zero-token.
- Correct-UQ MAE remains worse than no-token, so the absolute calibration gate
  has not passed.
- Full Stage 2D planning adaptation remains paused.

The follow-up 300-step run on 200 calibration frames passed the full gate:

| Intervention | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| none | 0.1087 | -0.6553 | -0.5898 |
| zero | 0.1129 | -0.6810 | -0.6208 |
| shuffled | 0.2940 | -0.0517 | -0.0372 |
| correct | **0.0406** | **0.8611** | **0.7253** |

The correct score is now both recoverable and better calibrated than all
controls. This validates the token interface and auxiliary objective, not the
grounding-only checkpoint as a planning model.

## Stage 2D: Planning Adaptation

Formal Stage 2D starts from base ORION and trains planning plus counterfactual
grounding jointly. Do not initialize from the grounding-only checkpoint.

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

First joint pilot overrides:

```yaml
max_steps: 100
gradient_accumulation: 4
learning_rate_projector: 5.0e-5
learning_rate_lora: 5.0e-6
lambda_ground: 1.0
lambda_consistency: 0.1
counterfactual_grounding: true
token_input: score_only
```

The 100-sample pilot performed only 25 optimizer updates. Its 400-sample
follow-up lowered `lambda_ground` to `0.2`, but correct-token ADE/FDE worsened
further. This training design is rejected; do not continue to the full split.

### Joint Pilot Result

Fixed-noise metrics on the same 35 valid calibration frames:

| Checkpoint / mode | ADE | FDE |
| --- | ---: | ---: |
| Base ORION / none | **0.0803** | **0.0745** |
| Joint 100 / correct | 0.0941 | 0.1070 |
| Joint 400 / none | 0.0818 | 0.0839 |
| Joint 400 / shuffled | 0.1874 | 0.2665 |
| Joint 400 / correct | 0.1046 | 0.1162 |

Joint 400 grounding remained strongly score-sensitive:

```text
correct Pearson: 0.879
shuffled Pearson: 0.089
```

The token is causally active, but the learned planning response is harmful.
The next experiment must move grounding to a separate LLM UQ readout token.

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
- Grounding-only planning collapses while joint training recovers: score
  semantics are learnable, but they must be constrained by the trajectory
  decoder during acquisition.
- Planning improves without grounding: treat the result as inconclusive because
  the gain may come from latent capacity rather than uncertainty.
