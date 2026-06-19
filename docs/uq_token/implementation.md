# Implementation Tracker

## Target Information Flow

```text
EVAViT tokens
  -> DensityUQEstimator
       -> active direction [B, 16]
       -> score [B, 1]
  -> UQProjector
       -> uncertainty token [B, 1, 4096]

QT-Former visual tokens [B, N, 4096]
  + uncertainty token
  -> LLM
  -> waypoint-token ego feature
  -> unchanged trajectory generator
```

## Required Code Changes

### 1. Expose Active Density Features

Owner:

```text
uq_estimator/density.py
```

Required API:

```python
UQOutput.embedding  # compatibility output [B, 256]
UQOutput.score      # [B, 1]
active_embedding    # [B, 16], without zero padding
```

Implemented by extending `UQOutput` with an optional `active_embedding` field.
Legacy UQEstimator callers remain compatible.

Status: complete.

### 2. Implement UQProjector

Suggested owner:

```text
uq_estimator/token_projector.py
```

Responsibilities:

- consume 16-D direction and scalar score;
- produce `[B, K, 4096]`;
- implement learned-null score gating;
- expose token norm diagnostics;
- support `K=1` and `K=4`;
- initialize the uncertainty-dependent branch near zero.

Implemented:

```text
uq_estimator/token_projector.py
17 -> 512 -> 4096
K=1
learned-null score gating
2,115,584 parameters
```

Status: complete.

### 3. Carry UQ Output to Orion

Current `OrionHead` returns:

```text
outs, vlm_memory, uncertainty_embedding, uncertainty_score
```

The primary path needs the active 16-D direction. Options:

1. Return an expanded UQ output object.
2. Return an additional tensor.
3. Let Orion own Density UQ instead of OrionHead.

The `OrionHead` tuple return contract remains unchanged. Its existing
`uq_output` object now exposes `active_embedding`, which Orion consumes before
falling back to the first configured dimensions of the compatibility embedding.

Status: complete.

### 4. Insert Tokens Before LLM

Owners:

```text
mmcv/models/detectors/orion.py
mmcv/utils/llava_arch.py
```

Training and inference paths must both append the same uncertainty token to the
visual token sequence before calling the LLM.

Critical paths:

- teacher-forced training call using `lm_head(...)`;
- inference call using `lm_head.inference_ego(...)`;
- text generation call if used for evaluation.

The token must receive ignored language labels, as visual tokens do.

Implemented through the existing LLaVA multimodal path by appending the
continuous UQ token to `vision_embeded` before it is passed as `images=`.

Covered paths:

- teacher-forced ORION training;
- `inference_ego`;
- ordinary text generation;
- the custom adaptation-training forward.

Status: complete.

### 5. Checkpoint Loading

Checkpoint should contain:

```text
uq_projector state
LLM LoRA state
optimizer and scheduler state
epoch and global step
configuration
base ORION checkpoint identifier
density UQ checkpoint hash or path
git commit
```

Suggested path:

```text
checkpoints/uq_token/
```

Implemented:

- training checkpoints contain projector, LoRA, optimizer, scheduler, epoch,
  step, loss, and CLI configuration;
- training resume uses the same adaptation-state loader;
- `adzoo/orion/test.py` and `scripts/eval_openloop.py` load adaptation weights
  from config or `UQ_TOKEN_CHECKPOINT`.

Status: initial checkpoint path complete.

### 6. Training Script

Suggested owner:

```text
scripts/train_uq_token.py
```

Requirements:

- route-based train/calibration filtering;
- explicit parameter-freeze audit;
- separate learning rates for projector and LoRA;
- gradient accumulation;
- resumable checkpoints;
- token norm and score diagnostics;
- baseline consistency loss;
- no final-test metric during tuning.

Implemented:

```text
scripts/train_uq_token.py
```

Current features:

- route split filtering from the density checkpoint;
- separate projector and LoRA learning rates;
- hard freeze audit;
- first-backward gradient audit;
- resumable optimizer and scheduler state;
- smoke-run sample and step limits.

Still required:

- calibration-route evaluation and early stopping;
- richer diagnostics and periodic checkpoints.

Status: initial training entry complete.

Low-UQ consistency is implemented as a second, no-gradient LLM forward without
the uncertainty token. Its waypoint representation is compared with the
conditioned representation using `(1 - score)`-weighted MSE.

### 7. Evaluation Script

Suggested owner:

```text
scripts/eval_uq_token.py
```

Requirements:

- support E0-E7 experiment modes;
- record frame and route identifiers;
- record UQ score, token norm, planning metrics, and controls;
- support shuffled UQ with a fixed seed;
- support zero/null-token control;
- aggregate by route, weather, scene type, and UQ quartile.

Status: not started.

### 8. UQ Grounding Head

Suggested owner:

```text
uq_estimator/grounding.py
```

Requirements:

- predict fixed density score from the waypoint hidden state;
- use SmoothL1 regression with a detached target;
- report MAE, Pearson, Spearman, and route-bootstrap intervals;
- save/load with projector and LoRA weights;
- support correct, zero, shuffled, and no-token modes.

Status: designed, not implemented.

## Existing Completed Components

| Component | Location | Status |
| --- | --- | --- |
| Density UQ runtime | `uq_estimator/density.py` | Complete |
| Descriptor caching | `scripts/cache_density_descriptors.py` | Complete |
| Density fitting | `scripts/fit_density_uq.py` | Complete |
| Density checkpoint | `checkpoints/density_uq/best.pt` | Complete |
| ORION density loading | `mmcv/models/dense_heads/orion_head.py` | Complete |
| Density tests | `tests/test_density_uq.py` | Complete |

## Verification Checklist

- [x] Projector unit tests.
- [x] Score gating and learned-null behavior test.
- [x] Output token shape test for K=1 and K=4.
- [x] Gradient reaches projector and LoRA.
- [x] No gradient reaches frozen model parameters.
- [x] Teacher-forced and inference paths use identical token construction.
- [x] Checkpoint save/reload equivalence.
- [ ] Shuffled-UQ mode is deterministic.
- [ ] Baseline mode is bitwise or numerically equivalent to pre-token ORION.
- [x] Full ORION one-batch forward succeeds.
- [x] Full ORION construction and real density-feature token append succeed.
- [x] One valid real-data backward step succeeds.
- [x] Low-UQ consistency loss and gradient-isolation tests.
- [x] Multi-step training diagnostics are stored in checkpoints.
- [ ] Grounding head and loss.
- [ ] Correct/zero/shuffled/no-token grounding pilot.
- [ ] Counterfactual token intervention evaluation.
