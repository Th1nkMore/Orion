# Method Design

## Primary Method: LLM Uncertainty Tokens

The density model generates a calibrated magnitude and an abnormality direction:

```text
u_dir:   [B, 16]  unit whitened residual direction
u_score: [B, 1]   calibrated normal-density tail position
```

The full input to the UQ projector is:

```text
u = concat(u_dir, u_score)  # [B, 17]
```

The implemented projector separates score magnitude from degradation
direction:

```text
uq_token =
    null_token
    + score * (score_basis + direction_projector(direction))
```

These tokens are concatenated with ORION's projected visual tokens before the
LLM:

```text
llm_visual_tokens = concat(vision_tokens, uq_tokens)
```

The LLM then produces the waypoint-token hidden state in the original way. No
post-LLM modulation is applied in the primary method.

This explicit score basis is important for grounding: score information has a
direct path into the token and does not have to emerge from an entangled MLP.

## Current Projector Configuration

Start with one uncertainty token:

```text
input_dim: 17
hidden_dim: 512
token_count: 1
llm_dim: 4096

score_basis: [1, 1, 4096]
direction_projector:
  Linear(16, 512)
  GELU
  LayerNorm(512)
  Linear(512, 4096)
```

Approximate parameter count: 2.1M.

Reasons to begin with `K=1`:

- stays inside the original lightweight-parameter narrative;
- minimizes disruption to sequence length and attention;
- provides a clean first test of whether the LLM uses uncertainty;
- avoids assigning unsupported semantics to multiple tokens.

`K=4` is a later capacity ablation, not the default.

## Score and Direction Roles

The projector receives both:

- Score expresses how far the sample is from the normal feature distribution.
- Direction expresses how the sample differs in the whitened density subspace.

A score-only token cannot distinguish different degradation directions, but it
is the cleanest first grounding target. Current experiments therefore ground
the score-only token first. The 16-D direction will be added only after score
grounding and calibration pass, then tested as an ablation rather than assumed
to help.

## Identity Behavior

The primary method should be close to baseline on low-UQ frames. Two mechanisms
will be evaluated:

### Score-Gated Token

```text
uq_token = score * projector(concat(direction, score))
```

At score zero, the added token is a zero vector. This is simple but a zero token
can still occupy an attention position.

### Learned Null Token

```text
uq_token = null_token + score * delta_projector(direction, score)
```

The null token is initialized to zero. This gives the model a stable token
position while gating only the uncertainty-dependent change.

The initial implementation should use the learned-null formulation because the
sequence structure remains constant between low- and high-UQ samples.

The learned null token is trainable, so score zero does not guarantee an
identically zero token after optimization. Low-UQ behavioral identity must be
enforced and measured using the consistency loss and zero/null-token ablation.

## Trainable Parameters

Primary training configuration:

```text
Frozen:
  EVAViT
  QT-Former and detection/map heads
  DensityUQEstimator
  trajectory generator / VAE
  base LLM weights

Trainable:
  UQProjector
  selected LLM LoRA adapters
```

Training the projector without LLM LoRA is retained as an ablation. A fully
frozen LLM may ignore an unfamiliar continuous token, so it is not the expected
best model.

The implemented projector has 2,115,584 trainable parameters. ORION's current
LoRA configuration contributes 16,777,216 trainable parameters.

## How the Token Acquires Meaning

The uncertainty token has no predefined natural-language semantics. Its meaning
must be learned functionally:

```text
planning / language / grounding losses
  -> waypoint hidden state
  -> LLM LoRA
  -> uncertainty token
  -> UQ projector
```

Planning loss can teach how token variation should alter the waypoint
representation, but planning improvement alone does not prove that the LLM has
encoded uncertainty. The model could use the token as an arbitrary latent code
or benefit only from extra parameters.

## UQ Token Grounding

Add a lightweight head on the LLM waypoint representation:

```text
waypoint hidden state [B, 4096]
  -> Linear / small MLP
  -> predicted density score [B, 1]
```

Use the fixed, stop-gradient Density UQ score as the target:

```text
L_ground = SmoothL1(predicted_score, density_score)
```

This does not create another hand-designed uncertainty label. It tests and
encourages whether the LLM representation retains the information supplied by
the density token.

Direction reconstruction is a possible later extension, but score prediction is
the first grounding task because it is simpler and easier to interpret.

## Revised Training Strategy

### Diagnostic Grounding

Grounding-only training is used to test whether the LLM representation can
recover the supplied density score:

```text
L_stage_a = lambda_vlm * L_vlm
          + lambda_ground * L_ground
          + lambda_consistency * L_consistency
```

The current grounding pilot additionally uses a counterfactual pair for every
image. The same image is forwarded once with its correct score and once with a
deterministically shuffled score, and each representation must recover the
score actually supplied:

```text
L_counterfactual =
    0.5 * L_ground(image, correct_score)
  + 0.5 * L_ground(image, shuffled_score)
```

This blocks the grounding head from solving the task from visual content alone.

The resulting checkpoint is diagnostic only. It must not initialize formal
planning training: the representation can encode score in a direction that is
incompatible with the frozen trajectory decoder.

### Joint Grounding and Planning Diagnostic

Formal adaptation starts from base ORION, not from the grounding-only
checkpoint. The correct-token branch receives both planning and grounding
supervision. A same-image shuffled-token branch receives grounding supervision
only:

```text
L_joint = lambda_plan * L_plan(correct_uq)
          + lambda_vae * L_vae
          + lambda_vlm * L_vlm
          + lambda_ground * 0.5 * (
                L_ground(correct_uq, correct_score)
              + L_ground(shuffled_uq, shuffled_score)
            )
          + lambda_consistency * L_consistency
          + lambda_collision * L_collision
```

This forces the LLM to retain score semantics while the planning target
constrains how that information is written into the trajectory-relevant
representation.

The diagnostic failed: stronger score recoverability coincided with worse
correct-token ADE/FDE. The grounding head should therefore not read directly
from the waypoint hidden state in the next architecture.

### Next Architecture: Separate UQ Readout Token

Add a dedicated textual special token such as `<uq_state>` to the LLM sequence:

```text
projected visual tokens + continuous UQ input token
  -> LLM
  -> uq_state hidden representation -> grounding head
  -> waypoint hidden representation -> trajectory decoder
```

The UQ readout token is trained to recover the supplied score under same-image
correct/shuffled interventions. The waypoint token is supervised only by
planning, VLM, collision, and low-UQ behavior-preservation losses.

This separation provides two properties:

1. score semantics are learned inside the LLM without forcing the trajectory
   decoder's input to be linearly predictive of score;
2. whether planning uses the information is tested separately by
   correct/zero/shuffled trajectory interventions.

The existing waypoint-grounding head remains a diagnostic baseline and must
not be used as the primary method.

## Losses

Use the original teacher-forced ORION losses:

```text
L_task = L_vlm + L_plan + lambda_col * L_collision
```

Add low-uncertainty representation consistency:

```text
L_consistency =
    mean((1 - score) * distance(
        ego_feature_with_uq,
        stopgrad(ego_feature_baseline)
    ))
```

Total:

```text
L = L_task + lambda_consistency * L_consistency
```

The first implementation may compute the baseline feature with a second frozen
LLM forward. If this doubles memory or runtime beyond feasibility, use periodic
or cached baseline features rather than silently dropping the constraint.

## Comparison Methods

### FiLM L1

Density embedding modulates QT-Former queries. The LLM receives changed visual
tokens, so it indirectly observes the effect. This is a relevant baseline but
does not expose uncertainty as an explicit concept.

### FiLM L2

Density embedding modifies the LLM-produced `ego_feature` before trajectory
generation. This tests whether direct post-LLM control is easier, but it does
not support the main claim that the LLM uses uncertainty.

### Textual Uncertainty Prompt

Quantize score into text such as low/medium/high reliability. This is highly
interpretable but throws away direction information and introduces arbitrary
thresholds. Keep it as an optional interpretability baseline, not the main
method.

## Expected Attack Surface

The experimental design must answer:

1. Is improvement caused by correct sample-level uncertainty or merely extra
   parameters?
2. Does the LLM actually attend to the uncertainty token?
3. Is the model simply learning weather categories?
4. Does lower collision rate come from stopping or excessive braking?
5. Does the method preserve clear-weather planning quality?
6. Does explicit LLM conditioning outperform indirect L1 and post-LLM L2?
7. Can the waypoint representation recover the supplied density score?
8. Does planning respond to counterfactual UQ-token changes with fixed vision?

## Evidence Required for the Main Claim

The current pilot establishes causal score use because correct-score
correlation is high and correlation collapses under shuffled-score
intervention. It also passes the absolute calibration condition: correct-token
MAE is lower than no-token, zero-token, and shuffled-token controls on 200
held-out calibration frames.

1. Correct UQ grounding outperforms no-token, zero-token, and shuffled-token
   controls.
2. Correct UQ outperforms shuffled UQ in planning, proving sample correspondence.
3. Correct UQ outperforms the same architecture with a zero/null token, controlling
   for added parameters.
4. Fixed-image token interventions produce systematic representation or planning
   changes.
5. Collision improvement is not explained by stopping or excessive braking.
