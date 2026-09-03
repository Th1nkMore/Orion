# ORION final-head decode adapter integration

> Date: 2026-08-26
>
> Status: dependency-light adapter and CPU fixtures complete; frozen ORION has
> **not** been run through this adapter.

## Audited repository contract

The adapter in `uq_estimator/orion_decode_adapter.py` was checked against:

- `mmcv/models/dense_heads/orion_head.py`;
- `mmcv/core/bbox/coder/fut_nms_free_coder.py`;
- `mmcv/core/bbox/util.py::denormalize_bbox`;
- `adzoo/orion/configs/orion_stage3_agent.py`.

For the Stage-3 agent configuration, the final evaluation tensors are expected
to have this dynamic contract:

| Tensor | Shape | Retained meaning |
|---|---|---|
| `all_cls_scores` | `[L,B,Q,9]` | raw class logits |
| `all_bbox_preds` | `[L,B,Q,10]` | ORION coder-format boxes; xyz already restored to point-cloud coordinates |
| `all_traj_preds` | `[L,B,Q,6,12]` | six motion modes, six 2-D displacement-delta steps |
| `all_traj_cls_scores` | `[L,B,Q,6]` | raw motion-mode logits |
| `all_traffic_states` | `[L,B,Q,4]` | query-level traffic-state/affect logits |

`L=6` in the checked head. In evaluation, `Q` is dynamic rather than assumed
from a stale comment: the configured 600 learned queries are temporally aligned
with 300 propagated queries, so the normal inference expectation is `Q=900`.
Training denoising can change intermediate counts; actual-target export must be
performed in evaluation mode and must trust the mutually checked tensor shapes,
not a hard-coded `Q`.

## Exact coder selection and retained information

`CustomNMSFreeCoder` does not perform conventional NMS in `decode_single`.
Its selection is:

1. sigmoid the last decoder layer's class logits;
2. flatten `[Q,C]` and take a global top-k (`max_num=300` in Stage 3);
3. recover `label = flat_index % C` and
   `bbox_index = flat_index // C`;
4. gather the query box and trajectory tensors;
5. denormalize boxes;
6. apply inclusive `post_center_range`, plus the coder's optional score mask
   and its strict-first/adaptive-fallback threshold behavior.

The adapter reproduces that order and retains the otherwise dropped
`source_query_index`. Because top-k operates on query × class, one source query
can legally occur more than once with different selected classes. Those rows
are not deduplicated. The output schema checks that every query-level value is
identical across duplicates: full decoded box, complete class sigmoid vector,
traffic logits, all trajectory modes, raw mode logits, selected mode, and
selected-mode occupancy.

Trajectory mode selection is the argmax of raw `all_traj_cls_scores`. ORION's
other code sometimes applies sigmoid before consuming these values, but sigmoid
does not change the argmax. Preserving the raw logits avoids an irreversible
transform while keeping the selected mode at parity with ORION.

Traffic logits are query-level. The adapter gathers them with the same retained
`source_query_index` **after** top-k and range/score masking. It never indexes
traffic state by decoded row position alone.

## Light-state stop condition

The checked Stage-3 head has `pred_traffic_light_state=True`, but the current
`orion_stage3_agent.py` test pipeline loads `LoadAnnotations3D` without
`with_light_state=True`. A head prediction existing is not enough for the
actual-target traffic error: privileged GT state and validity must also be
loaded.

Accordingly, the adapter configuration fails unless
`with_light_state=True`, and it fails if `all_traffic_states` is absent or
misaligned. The real offline exporter must use a dedicated data pipeline that
loads light state explicitly; this module does not silently edit the existing
agent config.

## Occupancy boundary

ORION trajectories are displacement deltas. The adapter selects the predicted
mode but does not invent a rasterizer. A caller must pass an audited function
that consumes `SelectedMotionRasterInputV1` and returns `[N,T,H,W]` occupancy.
That function owns cumulative summation, box geometry, and BEV grid convention,
and its immutable ID is stored in `DecodedORIONFrameV1`.

This is deliberate: the CPU fixture rasterizer is only a shape/alignment test.
It is not PlanningMetric-compatible and cannot pass the actual-target G1 gate.

## Minimal real hook (not yet executed)

After `pts_bbox_head.forward` returns `preds_dicts`, the frozen-model exporter
should call:

```python
decoded_batch = adapt_orion_head_outputs_v1(
    preds_dicts,
    config=ORIONDecodeAdapterConfigV1(
        num_classes=9,
        max_num=300,
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        class_mapping_id="<frozen class-map hash/id>",
        occupancy_rasterizer_id="<audited rasterizer hash/id>",
        with_light_state=True,
    ),
    occupancy_rasterizer=audited_selected_mode_rasterizer,
)
```

The caller should save both each `DecodedORIONFrameV1` and its
`CustomNMSFreeParityAuditV1`. The audit retains the flattened top-k indices,
source queries, labels, scores, post-center mask, score mask, final mask,
effective fallback threshold, and duplicate-query flag.

This hook still needs one chronological, bounded frozen-ORION smoke. Until
that happens, it is interface evidence only: it does not establish real model
execution, evaluator occupancy parity, or valid actual failure targets.

## CPU verification

```bash
.venv/bin/python -m pytest -q tests/test_orion_decode_adapter.py
```

The fixtures cover query × class duplicate selection, spatial masking,
adaptive score fallback, exact bbox field order, query-aligned traffic and
motion tensors, mode selection, required light-state provenance, final-layer
shape failures, and occupancy contract failures.
