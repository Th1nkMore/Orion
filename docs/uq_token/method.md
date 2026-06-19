# Method Design

## Primary Method: LLM Uncertainty Tokens

The density model generates a calibrated magnitude and an abnormality direction:

```text
u_dir:   [B, 16]  unit whitened residual direction
u_score: [B, 1]   calibrated normal-density tail position
```

The input to the UQ projector is:

```text
u = concat(u_dir, u_score)  # [B, 17]
```

The projector produces continuous tokens in the LLM embedding space:

```text
UQProjector(u) -> uq_tokens [B, K, 4096]
```

These tokens are concatenated with ORION's projected visual tokens before the
LLM:

```text
llm_visual_tokens = concat(vision_tokens, uq_tokens)
```

The LLM then produces the waypoint-token hidden state in the original way. No
post-LLM modulation is applied in the primary method.

## Initial Projector Configuration

Start with one uncertainty token:

```text
input_dim: 17
hidden_dim: 512
token_count: 1
llm_dim: 4096

Linear(17, 512)
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

A score-only token cannot distinguish different degradation directions. A
direction-only token discards calibrated severity. Both are required in the
main configuration.

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

## Revised Training Stages

### Stage A: Grounding

Train projector, LoRA, and grounding head so the waypoint representation can
recover the fixed density score:

```text
L_stage_a = lambda_vlm * L_vlm
          + lambda_ground * L_ground
          + lambda_consistency * L_consistency
```

### Stage B: Planning Adaptation

Continue from the grounded checkpoint:

```text
L_stage_b = lambda_plan * L_plan
          + lambda_vae * L_vae
          + lambda_vlm * L_vlm
          + lambda_ground * L_ground
          + lambda_consistency * L_consistency
          + lambda_collision * L_collision
```

The grounding term remains active so planning optimization cannot silently
discard the uncertainty information.

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

1. Correct UQ grounding outperforms no-token, zero-token, and shuffled-token
   controls.
2. Correct UQ outperforms shuffled UQ in planning, proving sample correspondence.
3. Correct UQ outperforms the same architecture with a zero/null token, controlling
   for added parameters.
4. Fixed-image token interventions produce systematic representation or planning
   changes.
5. Collision improvement is not explained by stopping or excessive braking.
