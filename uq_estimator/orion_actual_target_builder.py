"""Concrete production branch-target builder for frozen ORION replay.

This module consumes the audited context assembled by
``orion_actual_target_runner``.  It owns no model or data-loader construction;
it deterministically turns one decoded frame plus privileged B2D labels into
``BuiltBranchTargetV1`` using the frozen production raster/projection hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from uq_estimator.bev_target_rasterizer import (
    GT_RASTERIZER_ID,
    PAIRWISE_BEV_IOU_POLICY_ID,
    SELECTED_MODE_RASTERIZER_ID,
)
from uq_estimator.decoded_actual_target_export import (
    ActualTargetBranchBundleV1,
    DecodedORIONFrameV1,
    FailureEventPolicyV1,
    FrameChronologyV1,
    ObservationConditionV1,
    PrivilegedGroundTruthFrameV1,
    build_actual_target_branch,
)
from uq_estimator.object_failure_targets import TargetProvenanceV1
from uq_estimator.orion_actual_target_runner import (
    BOX_Z_ORIGIN_POLICY_ID,
    DECODED_BOX_Z_ORIGIN,
    GT_BOX_Z_ORIGIN,
    PILOT_CALIBRATION_POLICY_ID,
    PILOT_MATCH_POLICY_ID,
    PILOT_MAXIMUM_CENTER_DISTANCE_M,
    PILOT_MINIMUM_PREDICTION_SCORE,
    TRAFFIC_SEMANTICS_ID,
    BuiltBranchTargetV1,
    CanonicalORIONBatchV1,
    OrionActualTargetRunnerError,
)
from uq_estimator.projected_visible_support import (
    ORION_CAMERA_ORDER,
    VISIBLE_SUPPORT_PROJECTION_VERSION,
)


BUILDER_SCHEMA_VERSION = "orion.actual-target-production-branch-builder/v1"
ELIGIBILITY_POLICY_ID = "safety-actors-plus-affecting-traffic-light/v1"


class OrionActualTargetBuilderError(RuntimeError):
    """Raised when target construction would require guessing provenance."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrionActualTargetBuilderError("%s must be non-empty" % name)
    return value


def _sha256(value: str, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise OrionActualTargetBuilderError(
            "%s must be a lowercase SHA-256 digest" % name
        )
    return result


@dataclass(frozen=True)
class ProductionBranchTargetConfigV1:
    base_checkpoint_sha256: str
    inference_config_sha256: str
    git_revision: str
    route_id: str
    town: str
    class_mapping_id: str
    decoder_policy_id: str
    image_transform_id: str
    observed_corruption_family: str
    observed_severity: float
    observed_seed: int
    event_window_frames: Tuple[int, int]
    gt_box_z_origin: str = GT_BOX_Z_ORIGIN
    decoded_box_z_origin: str = DECODED_BOX_Z_ORIGIN
    schema_version: str = BUILDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BUILDER_SCHEMA_VERSION:
            raise OrionActualTargetBuilderError("unsupported builder schema")
        _sha256(self.base_checkpoint_sha256, "base_checkpoint_sha256")
        _sha256(self.inference_config_sha256, "inference_config_sha256")
        for name in (
            "git_revision",
            "route_id",
            "town",
            "class_mapping_id",
            "decoder_policy_id",
            "image_transform_id",
            "observed_corruption_family",
        ):
            _text(getattr(self, name), name)
        if self.observed_corruption_family == "none":
            raise OrionActualTargetBuilderError(
                "observed_corruption_family must not be 'none'"
            )
        if not math.isfinite(float(self.observed_severity)) or self.observed_severity <= 0:
            raise OrionActualTargetBuilderError("observed_severity must be positive")
        if (
            isinstance(self.observed_seed, bool)
            or not isinstance(self.observed_seed, int)
            or self.observed_seed < 0
        ):
            raise OrionActualTargetBuilderError("observed_seed must be non-negative int")
        if len(self.event_window_frames) != 2:
            raise OrionActualTargetBuilderError("event_window_frames must have two values")
        start, end = self.event_window_frames
        if start < 0 or end < start:
            raise OrionActualTargetBuilderError("event window is invalid")
        if (
            self.gt_box_z_origin != GT_BOX_Z_ORIGIN
            or self.decoded_box_z_origin != DECODED_BOX_Z_ORIGIN
        ):
            raise OrionActualTargetBuilderError(
                "production origins must be GT bottom and decoded center"
            )


def _cpu_tensor(value: Any, name: str, *, floating: bool = False) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
    else:
        try:
            result = torch.as_tensor(value).detach().cpu().clone()
        except Exception as exc:
            raise OrionActualTargetBuilderError("%s is not tensor-like" % name) from exc
    if floating:
        result = result.to(torch.float32)
    return result


def _gt_box_tensor(value: Any) -> torch.Tensor:
    tensor = getattr(value, "tensor", value)
    result = _cpu_tensor(tensor, "gt_bboxes_3d", floating=True)
    if result.ndim != 2 or result.shape[1] < 7:
        raise OrionActualTargetBuilderError("gt_bboxes_3d must have shape [N,D>=7]")
    if not torch.isfinite(result).all() or bool(torch.any(result[:, 3:6] <= 0)):
        raise OrionActualTargetBuilderError("GT boxes must be finite with positive size")
    return result


def _decoded_cpu_xy(decoded: DecodedORIONFrameV1) -> DecodedORIONFrameV1:
    """Move an audited decoded frame to CPU and use XY for center matching."""

    boxes = decoded.boxes_lidar.detach().cpu().float()
    return DecodedORIONFrameV1(
        centers_lidar=boxes[:, :2],
        boxes_lidar=boxes,
        classes=decoded.classes.detach().cpu(),
        scores=decoded.scores.detach().cpu().float(),
        source_query_index=decoded.source_query_index.detach().cpu(),
        class_probabilities=decoded.class_probabilities.detach().cpu().float(),
        selected_motion_occupancy=(
            decoded.selected_motion_occupancy.detach().cpu().float()
        ),
        traffic_state_logits=decoded.traffic_state_logits.detach().cpu().float(),
        all_trajectory_modes=decoded.all_trajectory_modes.detach().cpu().float(),
        trajectory_mode_scores=decoded.trajectory_mode_scores.detach().cpu().float(),
        selected_motion_mode_index=(
            decoded.selected_motion_mode_index.detach().cpu()
        ),
        occupancy_rasterizer_id=decoded.occupancy_rasterizer_id,
        decoder_layer=decoded.decoder_layer,
        decoder_policy_id=decoded.decoder_policy_id,
        class_mapping_id=decoded.class_mapping_id,
        with_light_state=decoded.with_light_state,
        traffic_probability_transform=decoded.traffic_probability_transform,
        decoder_flatten_policy=decoded.decoder_flatten_policy,
        decoder_topk=decoded.decoder_topk,
        motion_mode_policy=decoded.motion_mode_policy,
        coordinate_frame=decoded.coordinate_frame,
        schema_version=decoded.schema_version,
    )


def _scalar_timestamp(value: Any) -> float:
    tensor = _cpu_tensor(value, "timestamp", floating=True).reshape(-1)
    if tensor.numel() != 1 or not torch.isfinite(tensor).all():
        raise OrionActualTargetBuilderError("timestamp must contain one finite value")
    result = float(tensor.item())
    if result < 0:
        raise OrionActualTargetBuilderError("timestamp must be non-negative")
    return result


def _require_context(context: Mapping[str, Any], name: str) -> Any:
    if name not in context:
        raise OrionActualTargetBuilderError("runner context missing %s" % name)
    return context[name]


class ProductionActualTargetBranchBuilderV1:
    """Callable concrete builder accepted by production runtime hooks."""

    production_hook_id = BUILDER_SCHEMA_VERSION

    def __init__(self, config: ProductionBranchTargetConfigV1) -> None:
        self.config = config

    def __call__(
        self,
        decoded: DecodedORIONFrameV1,
        canonical: CanonicalORIONBatchV1,
        context: Mapping[str, Any],
    ) -> BuiltBranchTargetV1:
        if not isinstance(decoded, DecodedORIONFrameV1):
            raise OrionActualTargetBuilderError("decoded frame has the wrong type")
        if not isinstance(canonical, CanonicalORIONBatchV1):
            raise OrionActualTargetBuilderError("canonical batch has the wrong type")
        branch = _require_context(context, "branch")
        if branch not in ("clean", "observed"):
            raise OrionActualTargetBuilderError("branch must be clean or observed")
        if canonical.scene_token != self.config.route_id:
            raise OrionActualTargetBuilderError("canonical route does not match builder config")
        if decoded.class_mapping_id != self.config.class_mapping_id:
            raise OrionActualTargetBuilderError("decoded class mapping disagrees")
        if decoded.decoder_policy_id != self.config.decoder_policy_id:
            raise OrionActualTargetBuilderError("decoded policy disagrees")
        if decoded.occupancy_rasterizer_id != SELECTED_MODE_RASTERIZER_ID:
            raise OrionActualTargetBuilderError("decoded rasterizer ID is not production")
        if canonical.camera_order != ORION_CAMERA_ORDER:
            raise OrionActualTargetBuilderError("canonical camera order disagrees")

        expected_context = {
            "object_matching_policy_id": PILOT_MATCH_POLICY_ID,
            "minimum_prediction_score": PILOT_MINIMUM_PREDICTION_SCORE,
            "maximum_center_distance_m": PILOT_MAXIMUM_CENTER_DISTANCE_M,
            "traffic_semantics_id": TRAFFIC_SEMANTICS_ID,
            "gt_occupancy_rasterizer_id": GT_RASTERIZER_ID,
            "predicted_occupancy_rasterizer_id": SELECTED_MODE_RASTERIZER_ID,
            "pairwise_bev_iou_policy_id": PAIRWISE_BEV_IOU_POLICY_ID,
            "support_projector_id": VISIBLE_SUPPORT_PROJECTION_VERSION,
            "gt_box_z_origin": GT_BOX_Z_ORIGIN,
            "decoded_box_z_origin": DECODED_BOX_Z_ORIGIN,
            "box_z_origin_policy_id": BOX_Z_ORIGIN_POLICY_ID,
            "gt_target_eligibility_policy": ELIGIBILITY_POLICY_ID,
        }
        for name, expected in expected_context.items():
            if _require_context(context, name) != expected:
                raise OrionActualTargetBuilderError(
                    "runner context %s disagrees with production policy" % name
                )
        policy = _require_context(context, "failure_event_policy")
        if (
            not isinstance(policy, FailureEventPolicyV1)
            or policy.calibration_policy_id != PILOT_CALIBRATION_POLICY_ID
        ):
            raise OrionActualTargetBuilderError(
                "failure-event policy must be the preregistered pilot policy"
            )

        gt_boxes = _gt_box_tensor(canonical.data["gt_bboxes_3d"])
        gt_classes = _cpu_tensor(canonical.data["gt_labels_3d"], "gt_labels_3d")
        gt_attr = _cpu_tensor(
            canonical.data["gt_attr_labels"], "gt_attr_labels", floating=True
        )
        traffic_state = _cpu_tensor(canonical.data["traffic_state"], "traffic_state")
        traffic_mask = _cpu_tensor(
            canonical.data["traffic_state_mask"], "traffic_state_mask"
        ).to(torch.bool)
        actor_ids_tensor = _cpu_tensor(
            canonical.data["gt_actor_ids"], "gt_actor_ids"
        )
        actor_ids: Sequence[Any] = actor_ids_tensor.tolist()
        if any(value.shape[0] != gt_boxes.shape[0] for value in (gt_classes, gt_attr, traffic_state, traffic_mask, actor_ids_tensor)):
            raise OrionActualTargetBuilderError("privileged GT axes are not aligned")

        projector = _require_context(context, "project_visible_support")
        eligibility_filter = _require_context(context, "gt_target_eligibility_filter")
        gt_rasterizer = _require_context(context, "gt_occupancy_rasterizer")
        pairwise_iou_fn = _require_context(context, "pairwise_bev_iou")
        for function, name in (
            (projector, "project_visible_support"),
            (eligibility_filter, "gt_target_eligibility_filter"),
            (gt_rasterizer, "gt_occupancy_rasterizer"),
            (pairwise_iou_fn, "pairwise_bev_iou"),
        ):
            if not callable(function):
                raise OrionActualTargetBuilderError("%s must be callable" % name)

        matrices = _cpu_tensor(
            _require_context(context, "post_augmentation_lidar2img_cpu"),
            "post_augmentation_lidar2img_cpu",
            floating=True,
        )
        canonical_matrices = canonical.data["lidar2img"][0].detach().cpu().float()
        if not torch.allclose(matrices, canonical_matrices):
            raise OrionActualTargetBuilderError(
                "context matrices differ from canonical post-augmentation lidar2img"
            )
        image_rows = _require_context(context, "processed_image_hw_by_view")
        if tuple(tuple(row) for row in image_rows) != (
            canonical.processed_image_hw,
        ) * len(ORION_CAMERA_ORDER):
            raise OrionActualTargetBuilderError("processed image shape context disagrees")

        gt_projection = projector(
            gt_boxes,
            matrices,
            image_rows,
            box_source="privileged_gt",
            image_transform_id=self.config.image_transform_id,
        )
        try:
            eligibility = eligibility_filter(
                boxes=gt_boxes,
                classes=gt_classes,
                gt_attr=gt_attr,
                traffic_state=traffic_state,
                traffic_state_mask=traffic_mask,
                actor_ids=actor_ids,
                projected_support=gt_projection.support,
                support_actor_ids=actor_ids,
            )
        except OrionActualTargetRunnerError as exc:
            raise OrionActualTargetBuilderError(str(exc)) from exc
        eligible = eligibility.axes
        eligible_boxes = eligible["boxes"].detach().cpu().float()
        eligible_classes = eligible["classes"].detach().cpu().to(torch.long)
        eligible_attr = eligible["gt_attr"].detach().cpu().float()
        if not torch.equal(
            eligible_attr[:, 27], eligible_classes.to(eligible_attr.dtype)
        ):
            raise OrionActualTargetBuilderError(
                "eligible PlanningMetric category axis diverges from classes"
            )
        gt_raster = gt_rasterizer(
            eligible_boxes, eligible_attr.unsqueeze(0), include_union=False
        )
        if gt_raster.rasterizer_id != GT_RASTERIZER_ID:
            raise OrionActualTargetBuilderError("GT rasterizer returned wrong ID")

        decoded_cpu = _decoded_cpu_xy(decoded)
        pred_projection = projector(
            decoded_cpu.boxes_lidar,
            matrices,
            image_rows,
            box_source="decoded_orion",
            image_transform_id=self.config.image_transform_id,
        )
        if gt_projection.support_provenance != pred_projection.support_provenance:
            raise OrionActualTargetBuilderError("GT/pred support provenance disagrees")
        if not torch.equal(
            gt_projection.valid_patch_mask, pred_projection.valid_patch_mask
        ):
            raise OrionActualTargetBuilderError("GT/pred patch-valid masks disagree")
        pairwise_iou = pairwise_iou_fn(
            eligible_boxes, decoded_cpu.boxes_lidar
        )

        traffic_labels = eligible["traffic_state_labels"].detach().cpu().long()
        traffic_valid = eligible["traffic_state_valid"].detach().cpu().to(torch.bool)
        # The exporter requires non-negative integer storage. Invalid traffic
        # remains semantically absent through the validity mask, never a label.
        traffic_storage = torch.where(
            traffic_valid, traffic_labels, torch.zeros_like(traffic_labels)
        )
        ground_truth = PrivilegedGroundTruthFrameV1(
            centers_lidar=eligible_boxes[:, :2],
            boxes_lidar=eligible_boxes,
            classes=eligible_classes,
            motion_occupancy=gt_raster.per_object.detach().cpu().float(),
            motion_valid_mask=gt_raster.valid_mask.detach().cpu(),
            traffic_states=traffic_storage,
            traffic_state_valid=traffic_valid,
            occupancy_rasterizer_id=GT_RASTERIZER_ID,
        )

        frame_idx = canonical.frame_idx
        paired_replay_id = _text(
            _require_context(context, "paired_replay_id"), "paired_replay_id"
        )
        history_id = _text(
            _require_context(context, "branch_history_id"), "branch_history_id"
        )
        previous = _require_context(context, "previous_frame_idx")
        target_provenance = TargetProvenanceV1(
            base_checkpoint_sha256=self.config.base_checkpoint_sha256,
            inference_config_sha256=self.config.inference_config_sha256,
            git_revision=self.config.git_revision,
            route_id=self.config.route_id,
            town=self.config.town,
            frame_idx=frame_idx,
            observation_branch=branch,
            temporal_history_id=history_id,
            paired_history_protocol_id=paired_replay_id,
            class_mapping_id=self.config.class_mapping_id,
            decoder_policy_id=self.config.decoder_policy_id,
            camera_order=ORION_CAMERA_ORDER,
            image_transform_id=self.config.image_transform_id,
        )
        chronology = FrameChronologyV1(
            route_id=self.config.route_id,
            frame_idx=frame_idx,
            sequence_index=frame_idx,
            source_timestamp_s=_scalar_timestamp(canonical.data["timestamp"]),
            history_start_frame_idx=0,
            branch_history_id=history_id,
            paired_replay_id=paired_replay_id,
            history_content=branch,
            previous_frame_idx=previous,
        )
        if branch == "clean":
            observation = ObservationConditionV1(
                corruption_family="none",
                severity=0.0,
                seed=None,
                event_window_frames=None,
                active_at_frame=False,
            )
        else:
            start, end = self.config.event_window_frames
            observation = ObservationConditionV1(
                corruption_family=self.config.observed_corruption_family,
                severity=float(self.config.observed_severity),
                seed=self.config.observed_seed,
                event_window_frames=self.config.event_window_frames,
                active_at_frame=start <= frame_idx <= end,
            )

        branch_bundle: ActualTargetBranchBundleV1 = build_actual_target_branch(
            decoded_cpu,
            ground_truth,
            pairwise_iou,
            eligible["projected_support"].detach().cpu().float(),
            pred_projection.support.detach().cpu().float(),
            gt_projection.valid_patch_mask.detach().cpu(),
            support_provenance=gt_projection.support_provenance,
            target_provenance=target_provenance,
            chronology=chronology,
            observation=observation,
            failure_event_policy=policy,
            bev_iou_policy_id=PAIRWISE_BEV_IOU_POLICY_ID,
            max_center_distance=PILOT_MAXIMUM_CENTER_DISTANCE_M,
            minimum_prediction_score=PILOT_MINIMUM_PREDICTION_SCORE,
        )
        if (
            branch_bundle.occupancy_rasterizer_id != SELECTED_MODE_RASTERIZER_ID
            or branch_bundle.gt_occupancy_rasterizer_id != GT_RASTERIZER_ID
        ):
            raise OrionActualTargetBuilderError(
                "exporter did not retain distinct production rasterizer IDs"
            )
        audit = dict(eligibility.audit)
        audit.update(
            {
                "builder_schema_version": BUILDER_SCHEMA_VERSION,
                "branch": branch,
                "frame_idx": frame_idx,
                "traffic_invalid_storage_placeholder": 0,
                "traffic_invalid_semantics": "masked_not_negative",
                "predicted_object_count": int(decoded_cpu.classes.numel()),
                "gt_projection_box_z_origin": (
                    gt_projection.projection_provenance.box_z_origin
                ),
                "pred_projection_box_z_origin": (
                    pred_projection.projection_provenance.box_z_origin
                ),
                "gt_occupancy_rasterizer_id": GT_RASTERIZER_ID,
                "predicted_occupancy_rasterizer_id": SELECTED_MODE_RASTERIZER_ID,
                "pairwise_bev_iou_policy_id": PAIRWISE_BEV_IOU_POLICY_ID,
                "support_projector_id": VISIBLE_SUPPORT_PROJECTION_VERSION,
                "support_actor_ids_sha256": audit[
                    "post_filter_actor_ids_sha256"
                ],
            }
        )
        return BuiltBranchTargetV1(
            bundle=branch_bundle,
            eligibility_audit=audit,
        )


__all__ = [
    "BUILDER_SCHEMA_VERSION",
    "ELIGIBILITY_POLICY_ID",
    "OrionActualTargetBuilderError",
    "ProductionActualTargetBranchBuilderV1",
    "ProductionBranchTargetConfigV1",
]
