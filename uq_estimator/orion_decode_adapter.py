"""Adapt final-layer ORION head tensors without importing MMCV.

This module mirrors the selection performed by
``CustomNMSFreeCoder.decode_single`` while retaining query-level tensors that
the repository coder currently discards.  It deliberately does not change or
monkey-patch the production coder.

The adapter stops at an audited decoded-frame boundary.  Motion occupancy is
provided by an explicit caller-owned rasterizer so that a cheap placeholder
cannot be mistaken for the evaluator-compatible occupancy used by actual
failure targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Optional, Tuple

import torch

from uq_estimator.decoded_actual_target_export import (
    DecodedORIONFrameV1,
    MOTION_MODE_POLICY,
)


ADAPTER_SCHEMA_VERSION = "orion.head-to-decoded-adapter/v1"
DECODER_POLICY_ID = "custom-nms-free-coder-exact-v1"
FLATTEN_POLICY = "sigmoid_query_class_topk"


class ORIONDecodeAdapterError(ValueError):
    """Raised when adapting a head output would require guessing."""


@dataclass(frozen=True)
class ORIONDecodeAdapterConfigV1:
    """Frozen decoder and provenance settings for one export run."""

    num_classes: int
    max_num: int
    post_center_range: Tuple[float, float, float, float, float, float]
    class_mapping_id: str
    occupancy_rasterizer_id: str
    with_light_state: bool
    score_threshold: Optional[float] = None
    traffic_probability_transform: str = "sigmoid"
    decoder_policy_id: str = DECODER_POLICY_ID
    schema_version: str = ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_SCHEMA_VERSION:
            raise ORIONDecodeAdapterError(
                f"unsupported adapter schema {self.schema_version!r}"
            )
        if self.decoder_policy_id != DECODER_POLICY_ID:
            raise ORIONDecodeAdapterError(
                f"decoder_policy_id must be {DECODER_POLICY_ID!r}"
            )
        if isinstance(self.num_classes, bool) or self.num_classes <= 0:
            raise ORIONDecodeAdapterError("num_classes must be positive")
        if isinstance(self.max_num, bool) or self.max_num <= 0:
            raise ORIONDecodeAdapterError("max_num must be positive")
        if len(self.post_center_range) != 6 or not all(
            math.isfinite(float(value)) for value in self.post_center_range
        ):
            raise ORIONDecodeAdapterError(
                "post_center_range must contain six finite values"
            )
        lower = self.post_center_range[:3]
        upper = self.post_center_range[3:]
        if any(float(lo) > float(hi) for lo, hi in zip(lower, upper)):
            raise ORIONDecodeAdapterError(
                "post_center_range lower bounds must not exceed upper bounds"
            )
        if not str(self.class_mapping_id).strip():
            raise ORIONDecodeAdapterError("class_mapping_id must be non-empty")
        if not str(self.occupancy_rasterizer_id).strip():
            raise ORIONDecodeAdapterError(
                "occupancy_rasterizer_id must be non-empty"
            )
        if self.with_light_state is not True:
            raise ORIONDecodeAdapterError(
                "actual-target decoding requires with_light_state=True"
            )
        if self.score_threshold is not None:
            if not math.isfinite(float(self.score_threshold)):
                raise ORIONDecodeAdapterError("score_threshold must be finite")
            if float(self.score_threshold) < 0:
                raise ORIONDecodeAdapterError(
                    "score_threshold must be non-negative"
                )
        if self.traffic_probability_transform != "sigmoid":
            raise ORIONDecodeAdapterError(
                "ORION v1 traffic-state logits require sigmoid focal-loss semantics"
            )


@dataclass(frozen=True)
class SelectedMotionRasterInputV1:
    """Inputs passed to the audited selected-mode occupancy rasterizer.

    ORION predicts per-timestep displacement deltas.  ``selected_deltas`` is
    intentionally not cumulatively summed here: the evaluator-compatible
    rasterizer must own that transformation together with its box geometry and
    grid convention.
    """

    decoded_boxes_lidar: torch.Tensor
    selected_deltas: torch.Tensor
    source_query_index: torch.Tensor
    selected_mode_index: torch.Tensor
    batch_index: int
    trajectories_are_step_deltas: bool = True


OccupancyRasterizer = Callable[[SelectedMotionRasterInputV1], torch.Tensor]


@dataclass(frozen=True)
class CustomNMSFreeParityAuditV1:
    """Selection trace sufficient to compare against the repository coder."""

    batch_index: int
    decoder_layer: int
    flattened_topk_index: torch.Tensor
    topk_source_query_index: torch.Tensor
    topk_class_index: torch.Tensor
    topk_scores: torch.Tensor
    post_center_mask: torch.Tensor
    score_threshold_mask: torch.Tensor
    final_mask: torch.Tensor
    effective_score_threshold: Optional[float]
    duplicate_source_queries_present: bool
    schema_version: str = ADAPTER_SCHEMA_VERSION


@dataclass(frozen=True)
class AdaptedORIONBatchV1:
    frames: Tuple[DecodedORIONFrameV1, ...]
    audits: Tuple[CustomNMSFreeParityAuditV1, ...]
    schema_version: str = ADAPTER_SCHEMA_VERSION


def _require_float_tensor(
    value: object,
    name: str,
    *,
    ndim: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ORIONDecodeAdapterError(f"{name} must be a tensor")
    if value.ndim != ndim:
        raise ORIONDecodeAdapterError(
            f"{name} must have {ndim} dimensions, got {value.ndim}"
        )
    if not value.is_floating_point():
        raise ORIONDecodeAdapterError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ORIONDecodeAdapterError(f"{name} must be finite")
    return value


def _final_layer_tensors(
    preds_dicts: Mapping[str, torch.Tensor],
    config: ORIONDecodeAdapterConfigV1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = (
        "all_cls_scores",
        "all_bbox_preds",
        "all_traj_preds",
        "all_traj_cls_scores",
        "all_traffic_states",
    )
    missing = [key for key in required if key not in preds_dicts]
    if missing:
        raise ORIONDecodeAdapterError(
            "missing final-head tensors: " + ", ".join(missing)
        )

    cls = _require_float_tensor(
        preds_dicts["all_cls_scores"], "all_cls_scores", ndim=4
    )
    bbox = _require_float_tensor(
        preds_dicts["all_bbox_preds"], "all_bbox_preds", ndim=4
    )
    traj = _require_float_tensor(
        preds_dicts["all_traj_preds"], "all_traj_preds", ndim=5
    )
    traj_cls = _require_float_tensor(
        preds_dicts["all_traj_cls_scores"],
        "all_traj_cls_scores",
        ndim=4,
    )
    traffic = _require_float_tensor(
        preds_dicts["all_traffic_states"], "all_traffic_states", ndim=4
    )

    layer_count, batch_size, query_count, class_count = cls.shape
    if layer_count <= 0 or batch_size <= 0 or query_count <= 0:
        raise ORIONDecodeAdapterError(
            "ORION head tensors must contain layers, batches, and queries"
        )
    if class_count != config.num_classes:
        raise ORIONDecodeAdapterError(
            "all_cls_scores class dimension does not match coder num_classes"
        )
    if bbox.shape[:3] != cls.shape[:3] or bbox.shape[-1] != 10:
        raise ORIONDecodeAdapterError(
            "all_bbox_preds must have ORION shape [L,B,Q,10]"
        )
    if traj.shape[:3] != cls.shape[:3] or traj.shape[3] <= 0:
        raise ORIONDecodeAdapterError(
            "all_traj_preds must have shape [L,B,Q,M,2T]"
        )
    if traj.shape[-1] <= 0 or traj.shape[-1] % 2:
        raise ORIONDecodeAdapterError(
            "all_traj_preds last dimension must be a positive even 2T"
        )
    if traj_cls.shape != traj.shape[:4]:
        raise ORIONDecodeAdapterError(
            "all_traj_cls_scores must have shape [L,B,Q,M]"
        )
    if traffic.shape[:3] != cls.shape[:3] or traffic.shape[-1] <= 0:
        raise ORIONDecodeAdapterError(
            "all_traffic_states must have shape [L,B,Q,S]"
        )
    devices = (bbox.device, traj.device, traj_cls.device, traffic.device)
    if any(device != cls.device for device in devices):
        raise ORIONDecodeAdapterError("all head tensors must share one device")
    if config.max_num > query_count * class_count:
        # This matches the effective constraint of Tensor.topk used by the
        # repository coder, but produces a useful error instead of a backend
        # exception.
        raise ORIONDecodeAdapterError(
            "max_num exceeds flattened query×class candidate count"
        )

    return cls[-1], bbox[-1], traj[-1], traj_cls[-1], traffic[-1]


def denormalize_orion_bbox_exact(bbox_preds: torch.Tensor) -> torch.Tensor:
    """Dependency-light copy of ``mmcv.core.bbox.util.denormalize_bbox``.

    The ORION head has already restored ``x, y, z`` to its point-cloud range.
    The coder's ``pc_range`` parameter is therefore unused by the repository
    helper as well.
    """

    if bbox_preds.ndim != 2 or bbox_preds.shape[-1] not in (8, 10):
        raise ORIONDecodeAdapterError("bbox_preds must have shape [N,8 or 10]")
    rot = torch.atan2(bbox_preds[..., 6:7], bbox_preds[..., 7:8])
    parts = (
        bbox_preds[..., 0:1],
        bbox_preds[..., 1:2],
        bbox_preds[..., 4:5],
        bbox_preds[..., 2:3].exp(),
        bbox_preds[..., 3:4].exp(),
        bbox_preds[..., 5:6].exp(),
        rot,
    )
    if bbox_preds.shape[-1] > 8:
        return torch.cat(parts + (bbox_preds[..., 8:10],), dim=-1)
    return torch.cat(parts, dim=-1)


def _custom_coder_score_mask(
    scores: torch.Tensor,
    score_threshold: Optional[float],
) -> Tuple[torch.Tensor, Optional[float]]:
    """Mirror the coder's strict-first, adaptive-fallback threshold logic."""

    if score_threshold is None or not bool(score_threshold):
        return torch.ones_like(scores, dtype=torch.bool), None

    threshold = float(score_threshold)
    threshold_mask = scores > threshold
    effective = threshold
    while int(threshold_mask.sum().item()) == 0:
        effective *= 0.9
        if effective < 0.01:
            threshold_mask = scores > -1
            break
        # The repository switches from strict > to >= after the first failed
        # attempt; this asymmetry is preserved for exact parity.
        threshold_mask = scores >= effective
    return threshold_mask, effective


def _validate_occupancy(
    occupancy: object,
    *,
    count: int,
    timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    value = _require_float_tensor(
        occupancy, "selected_motion_occupancy", ndim=4
    )
    if value.shape[0] != count or value.shape[1] != timesteps:
        raise ORIONDecodeAdapterError(
            "selected_motion_occupancy must have shape [N,T,H,W]"
        )
    if value.shape[2] <= 0 or value.shape[3] <= 0:
        raise ORIONDecodeAdapterError("occupancy raster dimensions must be positive")
    if value.device != device:
        raise ORIONDecodeAdapterError(
            "occupancy rasterizer must return a tensor on the input device"
        )
    if torch.any(value < 0) or torch.any(value > 1):
        raise ORIONDecodeAdapterError(
            "selected_motion_occupancy must lie in [0, 1]"
        )
    return value


def adapt_orion_head_outputs_v1(
    preds_dicts: Mapping[str, torch.Tensor],
    *,
    config: ORIONDecodeAdapterConfigV1,
    occupancy_rasterizer: OccupancyRasterizer,
) -> AdaptedORIONBatchV1:
    """Decode a frozen ORION head output into audited per-frame objects.

    Selection order is identical to ``CustomNMSFreeCoder``:

    1. sigmoid all final-layer class logits;
    2. flatten query×class and take global top-k;
    3. gather query-level box/trajectory/state tensors by ``bbox_index``;
    4. denormalize boxes;
    5. apply inclusive post-center range and the coder's optional adaptive
       score threshold.

    No NMS or ``OrionHead.get_motion_bboxes`` debug-only post-filter is added.
    """

    if not callable(occupancy_rasterizer):
        raise ORIONDecodeAdapterError("occupancy_rasterizer must be callable")
    cls, bbox, traj, traj_cls, traffic = _final_layer_tensors(preds_dicts, config)
    decoder_layer = int(preds_dicts["all_cls_scores"].shape[0] - 1)
    # The repository coder constructs this with torch.tensor rather than
    # scores.new_tensor; preserve its default floating dtype as well as device.
    post_range = torch.tensor(config.post_center_range, device=cls.device)

    frames = []
    audits = []
    for batch_index in range(cls.shape[0]):
        class_probabilities_by_query = cls[batch_index].sigmoid()
        topk_scores, flat_index = class_probabilities_by_query.reshape(-1).topk(
            config.max_num
        )
        labels = flat_index % config.num_classes
        source_query_index = torch.div(
            flat_index, config.num_classes, rounding_mode="floor"
        )

        gathered_bbox = bbox[batch_index][source_query_index]
        decoded_boxes = denormalize_orion_bbox_exact(gathered_bbox)
        if not torch.isfinite(decoded_boxes).all():
            raise ORIONDecodeAdapterError(
                "denormalized boxes must remain finite for actual-target export"
            )
        centers = decoded_boxes[..., :3]
        post_center_mask = (
            (centers >= post_range[:3]).all(dim=1)
            & (centers <= post_range[3:]).all(dim=1)
        )
        score_mask, effective_threshold = _custom_coder_score_mask(
            topk_scores, config.score_threshold
        )
        final_mask = post_center_mask & score_mask

        kept_query = source_query_index[final_mask]
        kept_labels = labels[final_mask]
        kept_scores = topk_scores[final_mask]
        kept_boxes = decoded_boxes[final_mask]
        full_class_sigmoid = class_probabilities_by_query[kept_query]

        query_modes = traj[batch_index].reshape(
            traj.shape[1], traj.shape[2], traj.shape[3] // 2, 2
        )
        all_modes = query_modes[kept_query]
        all_mode_scores = traj_cls[batch_index][kept_query]
        selected_mode_index = all_mode_scores.argmax(dim=-1)
        row_index = torch.arange(
            kept_query.shape[0], device=kept_query.device, dtype=torch.long
        )
        selected_deltas = all_modes[row_index, selected_mode_index]
        selected_occupancy = _validate_occupancy(
            occupancy_rasterizer(
                SelectedMotionRasterInputV1(
                    decoded_boxes_lidar=kept_boxes,
                    selected_deltas=selected_deltas,
                    source_query_index=kept_query,
                    selected_mode_index=selected_mode_index,
                    batch_index=batch_index,
                )
            ),
            count=kept_query.shape[0],
            timesteps=all_modes.shape[2],
            device=cls.device,
        )

        traffic_logits = traffic[batch_index][kept_query]
        frame = DecodedORIONFrameV1(
            centers_lidar=kept_boxes[..., :3],
            boxes_lidar=kept_boxes,
            classes=kept_labels,
            scores=kept_scores,
            source_query_index=kept_query,
            class_probabilities=full_class_sigmoid,
            selected_motion_occupancy=selected_occupancy,
            traffic_state_logits=traffic_logits,
            all_trajectory_modes=all_modes,
            trajectory_mode_scores=all_mode_scores,
            selected_motion_mode_index=selected_mode_index,
            occupancy_rasterizer_id=config.occupancy_rasterizer_id,
            decoder_layer=decoder_layer,
            decoder_policy_id=config.decoder_policy_id,
            class_mapping_id=config.class_mapping_id,
            with_light_state=config.with_light_state,
            traffic_probability_transform=config.traffic_probability_transform,
            decoder_flatten_policy=FLATTEN_POLICY,
            decoder_topk=config.max_num,
            motion_mode_policy=MOTION_MODE_POLICY,
        )
        duplicate_present = (
            kept_query.unique().numel() != kept_query.numel()
        )
        if frame.duplicate_source_queries_present != duplicate_present:
            raise ORIONDecodeAdapterError("duplicate-query parity audit failed")

        frames.append(frame)
        audits.append(
            CustomNMSFreeParityAuditV1(
                batch_index=batch_index,
                decoder_layer=decoder_layer,
                flattened_topk_index=flat_index,
                topk_source_query_index=source_query_index,
                topk_class_index=labels,
                topk_scores=topk_scores,
                post_center_mask=post_center_mask,
                score_threshold_mask=score_mask,
                final_mask=final_mask,
                effective_score_threshold=effective_threshold,
                duplicate_source_queries_present=duplicate_present,
            )
        )

    return AdaptedORIONBatchV1(frames=tuple(frames), audits=tuple(audits))


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "DECODER_POLICY_ID",
    "FLATTEN_POLICY",
    "AdaptedORIONBatchV1",
    "CustomNMSFreeParityAuditV1",
    "ORIONDecodeAdapterConfigV1",
    "ORIONDecodeAdapterError",
    "OccupancyRasterizer",
    "SelectedMotionRasterInputV1",
    "adapt_orion_head_outputs_v1",
    "denormalize_orion_bbox_exact",
]
