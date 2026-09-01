"""Closed-loop ORION agent with legacy Density UQ permanently retired."""

import os


_base_ = ['./orion_stage3_agent.py']

legacy_density_flag = os.environ.get(
    'ORION_ENABLE_LEGACY_DENSITY_UQ', '0'
).strip().lower()
if legacy_density_flag not in {'0', 'false', 'no', 'off'}:
    raise RuntimeError(
        'Legacy Density UQ is retired from the current closed-loop config; '
        'use the frozen spatial observation-UQ adapter instead'
    )
qformer_path = os.environ.get(
    'ORION_QFORMER_PATH', 'ckpts/pretrain_qformer/'
)

# MMCV 1.x deep-copies every public top-level config value.  Module objects are
# not copyable, so keep only the resolved strings in the resulting config.
del os

model = dict(
    tokenizer=qformer_path,
    lm_head=qformer_path,
    use_uq_token=True,
    use_uq_vision_adapter=True,
    uq_token_checkpoint='',
    pts_bbox_head=dict(
        use_uncertainty=False,
        uq_checkpoint='',
        transformer=dict(use_uncertainty=False),
    ),
)
