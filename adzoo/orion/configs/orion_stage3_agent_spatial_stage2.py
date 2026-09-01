"""ORION config for the new spatial Stage-1 -> VLM Stage-2 mainline.

The historical ``orion_stage3_agent_uq.py`` is retained only so completed
oracle runs remain byte-for-byte auditable.  New jobs must use this config,
which never constructs the legacy Density estimator, scalar UQ token, scalar
vision adapter, FiLM, BEV cost, or speed governor.
"""

import os


_base_ = ['./orion_stage3_agent.py']

legacy_density = os.environ.get('ORION_ENABLE_LEGACY_DENSITY_UQ', '0').strip().lower()
legacy_conditioning = os.environ.get('ORION_CLOSEDLOOP_CONDITIONING', 'none').strip().lower()
if legacy_density not in {'0', 'false', 'no', 'off'}:
    raise RuntimeError('Legacy Density UQ is retired from the spatial Stage-2 mainline')
if legacy_conditioning != 'none':
    raise RuntimeError('Legacy Density token/vision conditioning is retired')

stage2_source = os.environ.get('ORION_STAGE2_SPATIAL_UQ_SOURCE', 'disabled').strip()
if stage2_source not in {'disabled', 'external_oracle', 'learned_adapter'}:
    raise RuntimeError('ORION_STAGE2_SPATIAL_UQ_SOURCE is invalid')
stage2_enabled = stage2_source != 'disabled'

stage1_checkpoint = os.environ.get('ORION_STAGE1_SPATIAL_UQ_CHECKPOINT', '').strip()
stage1_checkpoint_sha256 = os.environ.get(
    'ORION_STAGE1_SPATIAL_UQ_CHECKPOINT_SHA256', ''
).strip()
stage2_checkpoint = os.environ.get('ORION_STAGE2_TASK_CHECKPOINT', '').strip()
stage2_checkpoint_sha256 = os.environ.get(
    'ORION_STAGE2_TASK_CHECKPOINT_SHA256', ''
).strip()
if stage2_source == 'learned_adapter' and not stage1_checkpoint:
    raise RuntimeError('learned_adapter requires ORION_STAGE1_SPATIAL_UQ_CHECKPOINT')
if stage2_source != 'learned_adapter' and stage1_checkpoint:
    raise RuntimeError('only learned_adapter may load the Stage-1 checkpoint')
if not stage2_enabled and stage2_checkpoint:
    raise RuntimeError('disabled Stage-2 path cannot load a task checkpoint')

model = dict(
    # Every legacy scalar/global mechanism is structurally absent.
    use_uncertainty_l2=False,
    use_uq_token=False,
    uq_token_checkpoint='',
    use_uq_vision_adapter=False,
    use_bev_uncertainty=False,
    use_stage2_spatial_uq=stage2_enabled,
    stage2_spatial_uq_source=stage2_source,
    stage1_spatial_uq_checkpoint=stage1_checkpoint,
    stage1_spatial_uq_checkpoint_sha256=stage1_checkpoint_sha256,
    stage1_spatial_uq_warmup_frames=int(
        os.environ.get('ORION_STAGE1_SPATIAL_UQ_WARMUP_FRAMES', '60')
    ),
    pts_bbox_head=dict(
        use_uncertainty=False,
        uq_checkpoint='',
        use_stage2_spatial_uq=stage2_enabled,
        stage2_spatial_uq_checkpoint=stage2_checkpoint,
        stage2_spatial_uq_checkpoint_sha256=stage2_checkpoint_sha256,
        stage2_spatial_uq_tokens_per_view=int(
            os.environ.get('ORION_STAGE2_SPATIAL_UQ_TOKENS_PER_VIEW', '8')
        ),
        stage2_spatial_uq_hidden_dim=int(
            os.environ.get('ORION_STAGE2_SPATIAL_UQ_HIDDEN_DIM', '256')
        ),
        stage2_spatial_uq_num_heads=int(
            os.environ.get('ORION_STAGE2_SPATIAL_UQ_NUM_HEADS', '8')
        ),
        transformer=dict(use_uncertainty=False),
    ),
)

# MMCV 1.x deep-copies public config values; module objects are not copyable.
del os
