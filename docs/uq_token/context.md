# Project Context

## Research Problem

ORION uses a frozen EVAViT visual backbone, QT-Former-style visual processing,
an LLM, and a trajectory generator. The project investigates whether explicit
visual uncertainty can improve planning under rain, fog, darkness, and other
perception-degrading conditions without retraining the visual backbone.

The original learned `UQEstimator` produced:

```text
uncertainty_embedding: [B, 256]
uncertainty_score:     [B, 1]
```

It was supervised using hand-designed pseudo-labels derived from token
statistics and weather-aware calibration. This made the score easy to challenge:
the target was manually constructed, and high agreement with weather labels did
not establish principled uncertainty estimation.

## Density UQ Replacement

The learned estimator has been replaced by a normal-feature density model:

```text
EVAViT tokens [B, 6, 1600, 1024]
  -> per-view patch mean and standard deviation
  -> descriptor [B, 12288]
  -> standardization fitted on normal training routes
  -> PCA, 16 active components
  -> Ledoit-Wolf shrinkage Mahalanobis whitening
```

Outputs:

- `score [B, 1]`: empirical CDF of Mahalanobis distance on a held-out normal
  calibration split.
- `embedding [B, 256]`: unit whitened residual direction in the first 16
  dimensions, zero-padded to retain the existing ORION interface.

The weather label is used for evaluation, not to construct the density distance.

## Current Density UQ Result

Validation uses a route-disjoint 60/20/20 split:

| Metric | Result |
| --- | ---: |
| Frames | 12,806 |
| Routes | 50 |
| Test normal frames | 333 |
| Test adverse frames | 1,752 |
| AUROC | 0.799 |
| AUPRC | 0.951 |
| Route-bootstrap AUROC 95% CI | 0.675-0.915 |

The 256-component density model was rejected because AUROC fell to 0.627. The
additional components captured route and scene-content variation that reduced
weather-degradation separation. Sixteen active components gave the strongest
route-disjoint result among the screened PCA dimensions and density methods.

## Available Assets

Local repository:

```text
checkpoints/density_uq/best.pt
configs/density_uq.yaml
reports/density_uq/
uq_estimator/density.py
scripts/cache_density_descriptors.py
scripts/fit_density_uq.py
```

AutoDL server:

```text
Host: root@connect.weste.seetacloud.com
Port: 39408
Repository: /root/autodl-tmp/Orion
Conda environment: /root/autodl-tmp/conda/envs/orion-uq
Assets: /root/autodl-tmp/orion_assets
Features: /root/autodl-tmp/orion_assets/data/features
Descriptor cache: /root/autodl-tmp/orion_assets/data/density_uq/descriptors.pt
```

Feature data:

```text
12,806 files
tokens: [6, 1600, 1024], float16
total storage: approximately 235 GiB
```

## Existing ORION Information Flow

```text
EVAViT
  -> OrionHead / QT-Former visual memory
  -> projected visual tokens [B, N_visual, 4096]
  -> LLM
  -> waypoint-token hidden state, ego_feature [B, 4096]
  -> VAE / trajectory decoder
```

The existing FiLM locations are:

- L1: modifies QT-Former queries before visual tokens reach the LLM.
- L2: modifies `ego_feature` after the LLM and before the trajectory generator.

## Scientific Constraint

The central project narrative is not merely to make planning more conservative.
The intended contribution is to provide additional perception-reliability
information to the large model and let the model use that information during
decision formation.

Therefore:

- L2 is an engineering baseline, not the primary method.
- L1 is a useful indirect-conditioning baseline.
- Explicit uncertainty tokens entering the LLM are the primary method.

## What "LLM Adaptation" Means Here

The primary configuration does modify the large model through LoRA:

```text
Frozen:
  base LLM weights

Trainable:
  LLM attention LoRA adapters
  UQ token projector
```

It is therefore inaccurate to describe the method as injecting a token into a
completely frozen LLM. The base weights remain frozen, but LoRA lets the LLM
learn how the new continuous token should affect its waypoint representation.

Without LoRA or another adaptation mechanism, the pretrained LLM has no reason
to assign a useful meaning to the token. Projector-only conditioning remains an
ablation rather than the assumed main solution.

## Operational Constraints

- EVAViT remains frozen.
- Density UQ remains fixed after fitting.
- The main ORION checkpoint is approximately 36 GB.
- The QFormer/LLM assets are approximately 14 GB.
- Training and evaluation must use route-disjoint splits.
- Normal-scene performance cannot be sacrificed merely to reduce collisions.
- Reduced collision rate caused by stopping or excessive braking is not a valid
  success.
