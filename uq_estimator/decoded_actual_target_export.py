"""Dependency-light exporter for decoded frozen-ORION failure targets.

This is an adapter boundary, not a real ORION inference hook.  Callers must
decode the frozen model, rasterize selected-mode motion, compute audited BEV
IoU, and project visible GT/predicted object supports with post-augmentation
calibration before calling this module.  The module validates those inputs,
builds a compact actual-target bundle, and can bridge it into the Stage-1 v2
paired-feature record contract without importing ORION, MMCV, or CARLA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from uq_estimator.object_failure_targets import (
    ACTUAL_TARGET_PROVENANCE,
    BEV_OCCUPANCY_SIDECAR_VERSION,
    MOTION_MODE_POLICY,
    OBJECT_FAILURE_TARGET_VERSION,
    BEVOccupancyErrorSidecarV1,
    ObjectFailureTargetError,
    PatchSupportProvenanceV1,
    TargetProvenanceV1,
    aggregate_projected_visible_support,
    class_aware_distance_gated_match,
    compute_object_failure_components,
    make_bev_occupancy_error_sidecar,
)
from uq_estimator.spatial_training import PairedSpatialFeatureRecord


DECODED_INPUT_SCHEMA_VERSION = "orion.decoded-perception-frame/v1"
GT_INPUT_SCHEMA_VERSION = "b2d.privileged-object-frame/v1"
CHRONOLOGY_SCHEMA_VERSION = "orion.actual-target-chronology/v1"
OBSERVATION_SCHEMA_VERSION = "orion.actual-target-observation/v1"
FAILURE_EVENT_POLICY_VERSION = "orion.actual-failure-event-policy/v1"
BRANCH_BUNDLE_SCHEMA_VERSION = "orion.actual-target-branch-bundle/v2"
PAIRED_BUNDLE_SCHEMA_VERSION = "orion.actual-target-paired-bundle/v2"
GT_PROVENANCE = "privileged_b2d_3d_boxes_future_state"
MATCH_POLICY_ID = "class-score-distance-gated-one-to-one/v1"


class ActualTargetExportError(ValueError):
    """Raised when decoded target export would require guessing."""


def _text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ActualTargetExportError(f"{name} must be non-empty")
    return result


def _float_tensor(
    value: torch.Tensor,
    name: str,
    *,
    ndim: Optional[int] = None,
    unit_interval: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ActualTargetExportError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ActualTargetExportError(f"{name} must have {ndim} dimensions")
    if not value.is_floating_point():
        raise ActualTargetExportError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ActualTargetExportError(f"{name} must be finite")
    if unit_interval and (torch.any(value < 0) or torch.any(value > 1)):
        raise ActualTargetExportError(f"{name} must lie in [0, 1]")


def _integer_vector(value: torch.Tensor, name: str, length: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.shape[0] != length
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
    ):
        raise ActualTargetExportError(
            f"{name} must be an integer tensor with shape [{length}]"
        )


def _bool_tensor(value: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != shape
    ):
        raise ActualTargetExportError(
            f"{name} must be a boolean tensor with shape {shape}"
        )


@dataclass(frozen=True)
class DecodedORIONFrameV1:
    """Already-decoded final-layer object/motion/state outputs."""

    centers_lidar: torch.Tensor
    boxes_lidar: torch.Tensor
    classes: torch.Tensor
    scores: torch.Tensor
    source_query_index: torch.Tensor
    class_probabilities: torch.Tensor
    selected_motion_occupancy: torch.Tensor
    traffic_state_logits: torch.Tensor
    all_trajectory_modes: torch.Tensor
    trajectory_mode_scores: torch.Tensor
    selected_motion_mode_index: torch.Tensor
    occupancy_rasterizer_id: str
    decoder_layer: int
    decoder_policy_id: str
    class_mapping_id: str
    with_light_state: bool
    # ORION trains the three light-state logits with sigmoid focal loss.  The
    # fourth query-aligned logit (affects_ego) is retained for audit but is not
    # folded into the v1 light-state component.
    traffic_probability_transform: str = "sigmoid"
    decoder_flatten_policy: str = "sigmoid_query_class_topk"
    decoder_topk: int = 300
    motion_mode_policy: str = MOTION_MODE_POLICY
    coordinate_frame: str = "lidar"
    schema_version: str = DECODED_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DECODED_INPUT_SCHEMA_VERSION:
            raise ActualTargetExportError(
                f"unsupported decoded input schema {self.schema_version!r}"
            )
        if self.coordinate_frame != "lidar":
            raise ActualTargetExportError("decoded centers must be in lidar coordinates")
        if self.motion_mode_policy != MOTION_MODE_POLICY:
            raise ActualTargetExportError(
                "primary target requires ORION-selected motion mode, not oracle-best mode"
            )
        if self.with_light_state is not True:
            raise ActualTargetExportError(
                "traffic-state export requires the frozen ORION path with_light_state=True"
            )
        if self.decoder_flatten_policy != "sigmoid_query_class_topk":
            raise ActualTargetExportError(
                "decoder_flatten_policy must preserve CustomNMSFreeCoder query×class top-k"
            )
        if isinstance(self.decoder_topk, bool) or not isinstance(self.decoder_topk, int) or self.decoder_topk <= 0:
            raise ActualTargetExportError("decoder_topk must be a positive integer")
        if isinstance(self.decoder_layer, bool) or not isinstance(self.decoder_layer, int) or self.decoder_layer < 0:
            raise ActualTargetExportError("decoder_layer must be a non-negative integer")
        _text(self.decoder_policy_id, "decoder_policy_id")
        _text(self.class_mapping_id, "class_mapping_id")
        _text(self.occupancy_rasterizer_id, "occupancy_rasterizer_id")
        _float_tensor(self.centers_lidar, "centers_lidar", ndim=2)
        if self.centers_lidar.shape[1] not in (2, 3):
            raise ActualTargetExportError("centers_lidar must have shape [N,2 or 3]")
        count = self.centers_lidar.shape[0]
        _float_tensor(self.boxes_lidar, "boxes_lidar", ndim=2)
        if self.boxes_lidar.shape[0] != count or self.boxes_lidar.shape[1] < 7:
            raise ActualTargetExportError(
                "boxes_lidar must have shape [N,D] with D>=7"
            )
        center_dims = self.centers_lidar.shape[1]
        if not torch.allclose(
            self.boxes_lidar[:, :center_dims],
            self.centers_lidar,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ActualTargetExportError(
                "boxes_lidar center fields must equal centers_lidar"
            )
        _integer_vector(self.classes, "classes", count)
        if torch.any(self.classes < 0):
            raise ActualTargetExportError("decoded classes must be non-negative")
        _float_tensor(self.scores, "scores", ndim=1, unit_interval=True)
        if self.scores.shape != (count,):
            raise ActualTargetExportError(f"scores must have shape [{count}]")
        _integer_vector(self.source_query_index, "source_query_index", count)
        if torch.any(self.source_query_index < 0):
            raise ActualTargetExportError("source_query_index must be non-negative")
        _float_tensor(
            self.class_probabilities,
            "class_probabilities",
            ndim=2,
            unit_interval=True,
        )
        if self.class_probabilities.shape[0] != count or self.class_probabilities.shape[1] <= 0:
            raise ActualTargetExportError(
                "class_probabilities must have shape [N, positive_classes]"
            )
        if count and int(self.classes.max().item()) >= self.class_probabilities.shape[1]:
            raise ActualTargetExportError("decoded class ID exceeds full sigmoid vector")
        if count:
            row = torch.arange(count, device=self.scores.device)
            selected_class_scores = self.class_probabilities[row, self.classes.long()]
            if not torch.allclose(self.scores, selected_class_scores, atol=1e-6, rtol=1e-5):
                raise ActualTargetExportError(
                    "decoded scores must equal the selected entry of the full class sigmoid vector"
                )
        _float_tensor(
            self.selected_motion_occupancy,
            "selected_motion_occupancy",
            unit_interval=True,
        )
        if self.selected_motion_occupancy.ndim < 4 or self.selected_motion_occupancy.shape[0] != count:
            raise ActualTargetExportError(
                "selected_motion_occupancy must have shape [N,T,H,W,...]"
            )
        _float_tensor(self.traffic_state_logits, "traffic_state_logits", ndim=2)
        if self.traffic_state_logits.shape != (count, 4):
            raise ActualTargetExportError(
                "ORION v1 traffic_state_logits must have shape [N,4]"
            )
        if self.traffic_probability_transform != "sigmoid":
            raise ActualTargetExportError(
                "ORION v1 traffic-state logits require sigmoid focal-loss semantics"
            )
        _float_tensor(self.all_trajectory_modes, "all_trajectory_modes", ndim=4)
        if self.all_trajectory_modes.shape[0] != count or self.all_trajectory_modes.shape[-1] != 2:
            raise ActualTargetExportError(
                "all_trajectory_modes must have shape [N,M,T,2]"
            )
        _float_tensor(self.trajectory_mode_scores, "trajectory_mode_scores", ndim=2)
        if self.trajectory_mode_scores.shape != self.all_trajectory_modes.shape[:2]:
            raise ActualTargetExportError(
                "trajectory_mode_scores must have shape [N,M]"
            )
        _integer_vector(
            self.selected_motion_mode_index,
            "selected_motion_mode_index",
            count,
        )
        modes = self.all_trajectory_modes.shape[1]
        if modes <= 0 or torch.any(self.selected_motion_mode_index < 0) or torch.any(
            self.selected_motion_mode_index >= modes
        ):
            raise ActualTargetExportError("selected motion mode index is out of range")
        if count and not torch.equal(
            self.selected_motion_mode_index.long(),
            self.trajectory_mode_scores.argmax(dim=1),
        ):
            raise ActualTargetExportError(
                "selected_motion_mode_index must explicitly equal ORION mode-score argmax"
            )
        devices = (
            self.boxes_lidar,
            self.classes,
            self.scores,
            self.source_query_index,
            self.class_probabilities,
            self.selected_motion_occupancy,
            self.traffic_state_logits,
            self.all_trajectory_modes,
            self.trajectory_mode_scores,
            self.selected_motion_mode_index,
        )
        if any(value.device != self.centers_lidar.device for value in devices):
            raise ActualTargetExportError("all decoded tensors must share one device")
        # Flattened query×class top-k may emit the same source query multiple
        # times. The selected class/score can differ, but every query-level
        # tensor must remain identical across those decoded entries.
        for query_index in self.source_query_index.unique().tolist():
            rows = torch.nonzero(
                self.source_query_index == query_index, as_tuple=False
            ).flatten()
            if rows.numel() <= 1:
                continue
            reference = rows[0]
            for value, name in (
                (self.centers_lidar, "decoded centers"),
                (self.class_probabilities, "full class sigmoid"),
                (self.selected_motion_occupancy, "selected motion occupancy"),
                (self.traffic_state_logits, "traffic logits"),
                (self.all_trajectory_modes, "all trajectory modes"),
                (self.trajectory_mode_scores, "trajectory mode scores"),
            ):
                expected = value[reference].expand_as(value[rows])
                if not torch.allclose(value[rows], expected, atol=1e-6, rtol=1e-5):
                    raise ActualTargetExportError(
                        f"duplicate source query {query_index} has inconsistent {name}"
                    )
            if not torch.equal(
                self.selected_motion_mode_index[rows],
                self.selected_motion_mode_index[reference].expand_as(rows),
            ):
                raise ActualTargetExportError(
                    f"duplicate source query {query_index} has inconsistent selected mode"
                )

    @property
    def traffic_state_probabilities(self) -> torch.Tensor:
        if self.traffic_probability_transform == "sigmoid":
            return self.traffic_state_logits.sigmoid()
        return self.traffic_state_logits.softmax(dim=-1)

    @property
    def duplicate_source_queries_present(self) -> bool:
        return self.source_query_index.unique().numel() != self.source_query_index.numel()


@dataclass(frozen=True)
class PrivilegedGroundTruthFrameV1:
    """Privileged B2D labels used only by offline target construction."""

    centers_lidar: torch.Tensor
    boxes_lidar: torch.Tensor
    classes: torch.Tensor
    motion_occupancy: torch.Tensor
    motion_valid_mask: torch.Tensor
    traffic_states: torch.Tensor
    traffic_state_valid: torch.Tensor
    occupancy_rasterizer_id: str
    source_provenance: str = GT_PROVENANCE
    coordinate_frame: str = "lidar"
    schema_version: str = GT_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GT_INPUT_SCHEMA_VERSION:
            raise ActualTargetExportError(
                f"unsupported GT input schema {self.schema_version!r}"
            )
        if self.source_provenance != GT_PROVENANCE:
            raise ActualTargetExportError(
                "GT source must explicitly be privileged B2D boxes/future/state"
            )
        if self.coordinate_frame != "lidar":
            raise ActualTargetExportError("GT centers must be in lidar coordinates")
        _text(self.occupancy_rasterizer_id, "GT occupancy_rasterizer_id")
        _float_tensor(self.centers_lidar, "GT centers_lidar", ndim=2)
        if self.centers_lidar.shape[1] not in (2, 3):
            raise ActualTargetExportError("GT centers_lidar must have shape [G,2 or 3]")
        count = self.centers_lidar.shape[0]
        _float_tensor(self.boxes_lidar, "GT boxes_lidar", ndim=2)
        if self.boxes_lidar.shape[0] != count or self.boxes_lidar.shape[1] < 7:
            raise ActualTargetExportError(
                "GT boxes_lidar must have shape [G,D] with D>=7"
            )
        center_dims = self.centers_lidar.shape[1]
        if not torch.allclose(
            self.boxes_lidar[:, :center_dims],
            self.centers_lidar,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ActualTargetExportError(
                "GT boxes_lidar center fields must equal GT centers_lidar"
            )
        _integer_vector(self.classes, "GT classes", count)
        if torch.any(self.classes < 0):
            raise ActualTargetExportError("GT classes must be non-negative")
        _float_tensor(self.motion_occupancy, "GT motion_occupancy", unit_interval=True)
        if self.motion_occupancy.ndim < 4 or self.motion_occupancy.shape[0] != count:
            raise ActualTargetExportError(
                "GT motion_occupancy must have shape [G,T,H,W,...]"
            )
        _bool_tensor(
            self.motion_valid_mask,
            "motion_valid_mask",
            (count, self.motion_occupancy.shape[1]),
        )
        _integer_vector(self.traffic_states, "traffic_states", count)
        _bool_tensor(
            self.traffic_state_valid, "traffic_state_valid", (count,)
        )
        if bool(self.traffic_state_valid.any()) and torch.any(
            self.traffic_states[self.traffic_state_valid] < 0
        ):
            raise ActualTargetExportError(
                "valid traffic_states must be non-negative"
            )
        if bool(self.traffic_state_valid.any()) and torch.any(
            self.traffic_states[self.traffic_state_valid] > 2
        ):
            raise ActualTargetExportError(
                "ORION v1 valid traffic-state labels must lie in [0,2]"
            )
        devices = (
            self.boxes_lidar,
            self.classes,
            self.motion_occupancy,
            self.motion_valid_mask,
            self.traffic_states,
            self.traffic_state_valid,
        )
        if any(value.device != self.centers_lidar.device for value in devices):
            raise ActualTargetExportError("all GT tensors must share one device")


@dataclass(frozen=True)
class FrameChronologyV1:
    """Route chronology and temporal-memory audit identity for one branch."""

    route_id: str
    frame_idx: int
    sequence_index: int
    source_timestamp_s: float
    history_start_frame_idx: int
    branch_history_id: str
    paired_replay_id: str
    history_content: str
    previous_frame_idx: Optional[int]
    schema_version: str = CHRONOLOGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHRONOLOGY_SCHEMA_VERSION:
            raise ActualTargetExportError(
                f"unsupported chronology schema {self.schema_version!r}"
            )
        _text(self.route_id, "chronology route_id")
        _text(self.branch_history_id, "branch_history_id")
        _text(self.paired_replay_id, "paired_replay_id")
        if self.history_content not in ("clean", "observed", "shared"):
            raise ActualTargetExportError(
                "history_content must be clean, observed, or shared"
            )
        for name in ("frame_idx", "sequence_index", "history_start_frame_idx"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ActualTargetExportError(f"{name} must be a non-negative integer")
        if self.history_start_frame_idx > self.frame_idx:
            raise ActualTargetExportError("history cannot start after the current frame")
        if not math.isfinite(float(self.source_timestamp_s)) or self.source_timestamp_s < 0:
            raise ActualTargetExportError("source_timestamp_s must be finite and non-negative")
        if self.previous_frame_idx is not None:
            if (
                isinstance(self.previous_frame_idx, bool)
                or not isinstance(self.previous_frame_idx, int)
                or self.previous_frame_idx < self.history_start_frame_idx
                or self.previous_frame_idx >= self.frame_idx
            ):
                raise ActualTargetExportError(
                    "previous_frame_idx must be inside history and before frame_idx"
                )
        elif self.frame_idx != self.history_start_frame_idx:
            raise ActualTargetExportError(
                "non-initial frames must record previous_frame_idx"
            )


@dataclass(frozen=True)
class ObservationConditionV1:
    """Observed image condition; privileged labels are never model inputs."""

    corruption_family: str
    severity: float
    seed: Optional[int]
    event_window_frames: Optional[tuple[int, int]]
    active_at_frame: bool
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ActualTargetExportError(
                f"unsupported observation schema {self.schema_version!r}"
            )
        family = _text(self.corruption_family, "corruption_family")
        if not math.isfinite(float(self.severity)) or self.severity < 0:
            raise ActualTargetExportError("severity must be finite and non-negative")
        if family == "none":
            if self.severity != 0 or self.seed is not None or self.event_window_frames is not None or self.active_at_frame:
                raise ActualTargetExportError(
                    "clean observation must have severity=0, no seed/window, and inactive"
                )
            return
        if self.severity <= 0:
            raise ActualTargetExportError("a corruption condition must have positive severity")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ActualTargetExportError("a corruption condition requires a non-negative seed")
        if self.event_window_frames is None or len(self.event_window_frames) != 2:
            raise ActualTargetExportError("a corruption condition requires an event window")
        start, end = self.event_window_frames
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) or start < 0 or end < start:
            raise ActualTargetExportError("event_window_frames must be ordered non-negative integers")


@dataclass(frozen=True)
class FailureEventPolicyV1:
    """Explicit thresholds separating severity regression from event labels."""

    component_names: tuple[str, ...] = (
        "miss",
        "class",
        "localization",
        "motion_occupancy",
        "traffic_state",
        "false_positive",
    )
    component_thresholds: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    minimum_patch_support: float = 0.01
    calibration_policy_id: str = "mock-only-unfitted-thresholds"
    schema_version: str = FAILURE_EVENT_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FAILURE_EVENT_POLICY_VERSION:
            raise ActualTargetExportError(
                f"unsupported failure event policy {self.schema_version!r}"
            )
        if self.component_names != (
            "miss",
            "class",
            "localization",
            "motion_occupancy",
            "traffic_state",
            "false_positive",
        ):
            raise ActualTargetExportError(
                "component_names must use the canonical exporter order"
            )
        if len(self.component_thresholds) != len(self.component_names):
            raise ActualTargetExportError(
                "component_thresholds must follow every canonical component"
            )
        for value in self.component_thresholds:
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 1
            ):
                raise ActualTargetExportError("failure-event thresholds must lie in (0, 1]")
        if (
            isinstance(self.minimum_patch_support, bool)
            or not math.isfinite(float(self.minimum_patch_support))
            or not 0 < float(self.minimum_patch_support) <= 1
        ):
            raise ActualTargetExportError("minimum_patch_support must lie in (0, 1]")
        _text(self.calibration_policy_id, "calibration_policy_id")


@dataclass(frozen=True)
class ActualTargetBranchBundleV1:
    """Compact target maps and diagnostics for one clean/observed branch."""

    error_severity_target: torch.Tensor
    failure_event_target: torch.Tensor
    target_valid_mask: torch.Tensor
    component_errors: torch.Tensor
    component_error_names: tuple[str, ...]
    object_component_values: torch.Tensor
    object_component_valid: torch.Tensor
    gt_to_pred: torch.Tensor
    pred_to_gt: torch.Tensor
    matched_center_distance: torch.Tensor
    valid_pair_mask: torch.Tensor
    max_center_distance: torch.Tensor
    minimum_prediction_score: float
    match_policy_id: str
    gt_centers_lidar: torch.Tensor
    gt_boxes_lidar: torch.Tensor
    gt_classes: torch.Tensor
    decoded_centers_lidar: torch.Tensor
    decoded_boxes_lidar: torch.Tensor
    decoded_classes: torch.Tensor
    pairwise_bev_iou: torch.Tensor
    bev_iou_policy_id: str
    gt_projected_support: torch.Tensor
    pred_projected_support: torch.Tensor
    gt_motion_occupancy: torch.Tensor
    gt_motion_valid_mask: torch.Tensor
    selected_motion_occupancy: torch.Tensor
    gt_traffic_states: torch.Tensor
    gt_traffic_state_valid: torch.Tensor
    # This legacy-named field describes only the predicted selected-mode
    # rasterizer.  GT rasterization has independent provenance below.
    occupancy_rasterizer_id: str
    gt_occupancy_rasterizer_id: str
    source_query_index: torch.Tensor
    decoded_scores: torch.Tensor
    full_class_sigmoid: torch.Tensor
    traffic_state_logits: torch.Tensor
    traffic_probability_transform: str
    all_trajectory_modes: torch.Tensor
    trajectory_mode_scores: torch.Tensor
    selected_motion_mode_index: torch.Tensor
    duplicate_source_queries_present: bool
    decoder_layer: int
    decoder_flatten_policy: str
    decoder_topk: int
    with_light_state: bool
    motion_mode_policy: str
    failure_event_policy: FailureEventPolicyV1
    support_provenance: PatchSupportProvenanceV1
    target_provenance: TargetProvenanceV1
    chronology: FrameChronologyV1
    observation: ObservationConditionV1
    bev_occupancy_sidecar: Optional[BEVOccupancyErrorSidecarV1] = None
    schema_version: str = BRANCH_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BRANCH_BUNDLE_SCHEMA_VERSION:
            raise ActualTargetExportError(
                f"unsupported branch bundle schema {self.schema_version!r}"
            )
        _float_tensor(
            self.error_severity_target,
            "error_severity_target",
            ndim=2,
            unit_interval=True,
        )
        shape = tuple(self.error_severity_target.shape)
        _bool_tensor(
            self.failure_event_target, "failure_event_target", shape
        )
        _bool_tensor(self.target_valid_mask, "target_valid_mask", shape)
        _float_tensor(self.component_errors, "component_errors", ndim=3, unit_interval=True)
        if tuple(self.component_errors.shape[:2]) != shape:
            raise ActualTargetExportError("component_errors spatial shape must match severity")
        if self.component_errors.shape[-1] != len(self.component_error_names):
            raise ActualTargetExportError("component error names/count disagree")
        if len(self.component_error_names) != len(set(self.component_error_names)) or any(
            not str(name).strip() for name in self.component_error_names
        ):
            raise ActualTargetExportError("component_error_names must be unique and non-empty")
        if self.component_error_names != self.failure_event_policy.component_names:
            raise ActualTargetExportError(
                "component_error_names must match the failure-event policy"
            )
        _float_tensor(self.object_component_values, "object_component_values", ndim=2, unit_interval=True)
        if not isinstance(self.pred_to_gt, torch.Tensor) or self.pred_to_gt.ndim != 1 or self.pred_to_gt.dtype != torch.long:
            raise ActualTargetExportError("pred_to_gt must be a one-dimensional long tensor")
        if not isinstance(self.gt_to_pred, torch.Tensor) or self.gt_to_pred.ndim != 1 or self.gt_to_pred.dtype != torch.long:
            raise ActualTargetExportError("gt_to_pred must be a one-dimensional long tensor")
        gt_count = self.gt_to_pred.shape[0]
        pred_count = self.pred_to_gt.shape[0]
        if (
            not isinstance(self.matched_center_distance, torch.Tensor)
            or self.matched_center_distance.shape != (gt_count,)
            or not self.matched_center_distance.is_floating_point()
        ):
            raise ActualTargetExportError(
                "matched_center_distance must be floating point with shape [G]"
            )
        _bool_tensor(
            self.valid_pair_mask, "valid_pair_mask", (gt_count, pred_count)
        )
        _float_tensor(self.max_center_distance, "max_center_distance", ndim=1)
        if self.max_center_distance.shape != (gt_count,) or torch.any(
            self.max_center_distance <= 0
        ):
            raise ActualTargetExportError(
                "max_center_distance must be a positive vector with shape [G]"
            )
        if (
            isinstance(self.minimum_prediction_score, bool)
            or not math.isfinite(float(self.minimum_prediction_score))
            or not 0 <= float(self.minimum_prediction_score) <= 1
        ):
            raise ActualTargetExportError(
                "minimum_prediction_score must lie in [0, 1]"
            )
        if self.match_policy_id != MATCH_POLICY_ID:
            raise ActualTargetExportError(
                f"match_policy_id must be {MATCH_POLICY_ID!r}"
            )
        expected_object_component_shape = (
            gt_count + pred_count,
            len(self.component_error_names),
        )
        if self.object_component_values.shape != expected_object_component_shape:
            raise ActualTargetExportError(
                "object_component_values must have shape [G+N,6], including FP rows"
            )
        _bool_tensor(
            self.object_component_valid,
            "object_component_valid",
            expected_object_component_shape,
        )
        if torch.any(self.object_component_values[~self.object_component_valid] != 0):
            raise ActualTargetExportError(
                "invalid object-component entries must use zero placeholders"
            )
        _float_tensor(self.gt_centers_lidar, "gt_centers_lidar", ndim=2)
        _float_tensor(self.decoded_centers_lidar, "decoded_centers_lidar", ndim=2)
        if self.gt_centers_lidar.shape[0] != gt_count or self.decoded_centers_lidar.shape[0] != pred_count or self.gt_centers_lidar.shape[1:] != self.decoded_centers_lidar.shape[1:]:
            raise ActualTargetExportError("stored GT/decoded center shapes disagree with matches")
        _float_tensor(self.gt_boxes_lidar, "gt_boxes_lidar", ndim=2)
        _float_tensor(self.decoded_boxes_lidar, "decoded_boxes_lidar", ndim=2)
        if self.gt_boxes_lidar.shape[0] != gt_count or self.gt_boxes_lidar.shape[1] < 7:
            raise ActualTargetExportError("stored GT boxes must have shape [G,D], D>=7")
        if self.decoded_boxes_lidar.shape[0] != pred_count or self.decoded_boxes_lidar.shape[1] < 7:
            raise ActualTargetExportError(
                "stored decoded boxes must have shape [N,D], D>=7"
            )
        center_dims = self.gt_centers_lidar.shape[1]
        if not torch.allclose(
            self.gt_boxes_lidar[:, :center_dims],
            self.gt_centers_lidar,
            atol=1e-6,
            rtol=1e-6,
        ) or not torch.allclose(
            self.decoded_boxes_lidar[:, :center_dims],
            self.decoded_centers_lidar,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ActualTargetExportError(
                "stored box center fields must equal stored centers"
            )
        _integer_vector(self.gt_classes, "gt_classes", gt_count)
        _integer_vector(self.decoded_classes, "decoded_classes", pred_count)
        _float_tensor(self.decoded_scores, "decoded_scores", ndim=1, unit_interval=True)
        if self.decoded_scores.shape != (pred_count,):
            raise ActualTargetExportError("decoded_scores count must match pred_to_gt")
        match_audit_tensors = (
            self.gt_to_pred,
            self.pred_to_gt,
            self.matched_center_distance,
            self.valid_pair_mask,
            self.max_center_distance,
            self.gt_centers_lidar,
            self.decoded_centers_lidar,
            self.gt_classes,
            self.decoded_classes,
        )
        if any(value.device != self.decoded_scores.device for value in match_audit_tensors):
            raise ActualTargetExportError("stored match-audit tensors must share a device")
        pair_distance = torch.cdist(
            self.gt_centers_lidar[:, :2], self.decoded_centers_lidar[:, :2]
        )
        expected_valid_pairs = (
            (self.gt_classes[:, None] == self.decoded_classes[None, :])
            & (pair_distance <= self.max_center_distance[:, None])
            & (
                self.decoded_scores[None, :]
                >= float(self.minimum_prediction_score)
            )
        )
        if not torch.equal(self.valid_pair_mask, expected_valid_pairs):
            raise ActualTargetExportError(
                "valid_pair_mask is inconsistent with class/score/distance gates"
            )
        matched_gt = self.gt_to_pred >= 0
        if torch.any(self.gt_to_pred >= pred_count) or torch.any(self.pred_to_gt >= gt_count):
            raise ActualTargetExportError("stored match indices are out of range")
        if bool(matched_gt.any()):
            gt_index = torch.nonzero(matched_gt, as_tuple=False).flatten()
            pred_index = self.gt_to_pred[gt_index]
            if not bool(self.valid_pair_mask[gt_index, pred_index].all()):
                raise ActualTargetExportError("stored match uses an invalid gated pair")
            if not torch.equal(self.pred_to_gt[pred_index], gt_index):
                raise ActualTargetExportError("stored match maps are not reciprocal")
            if not torch.allclose(
                self.matched_center_distance[gt_index],
                pair_distance[gt_index, pred_index],
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ActualTargetExportError("stored matched distances are inconsistent")
        if bool((~matched_gt).any()) and not bool(
            torch.isposinf(self.matched_center_distance[~matched_gt]).all()
        ):
            raise ActualTargetExportError("unmatched GT distances must be +inf")
        matched_pred = self.pred_to_gt >= 0
        if bool(matched_pred.any()):
            pred_index = torch.nonzero(matched_pred, as_tuple=False).flatten()
            gt_index = self.pred_to_gt[pred_index]
            if not torch.equal(self.gt_to_pred[gt_index], pred_index):
                raise ActualTargetExportError("stored prediction matches are not reciprocal")
        _float_tensor(self.pairwise_bev_iou, "pairwise_bev_iou", ndim=2, unit_interval=True)
        if self.pairwise_bev_iou.shape != (gt_count, pred_count):
            raise ActualTargetExportError("pairwise_bev_iou shape disagrees with object counts")
        _text(self.bev_iou_policy_id, "bev_iou_policy_id")
        _float_tensor(self.gt_projected_support, "gt_projected_support", ndim=3, unit_interval=True)
        _float_tensor(self.pred_projected_support, "pred_projected_support", ndim=3, unit_interval=True)
        if self.gt_projected_support.shape != (*shape, gt_count) or self.pred_projected_support.shape != (*shape, pred_count):
            raise ActualTargetExportError("stored projected support shape disagrees with target/object counts")
        audit_tensors = (
            self.error_severity_target,
            self.failure_event_target,
            self.target_valid_mask,
            self.component_errors,
            self.object_component_values,
            self.object_component_valid,
            self.gt_projected_support,
            self.pred_projected_support,
        )
        if any(value.device != self.error_severity_target.device for value in audit_tensors):
            raise ActualTargetExportError(
                "target, object-component, and projected-support tensors must share a device"
            )
        all_support = torch.cat(
            (self.gt_projected_support, self.pred_projected_support), dim=-1
        )
        effective_components = torch.where(
            self.object_component_valid,
            self.object_component_values,
            torch.zeros_like(self.object_component_values),
        )
        expected_components = 1.0 - torch.prod(
            1.0
            - all_support.unsqueeze(-1)
            * effective_components.view(
                1, 1, gt_count + pred_count, len(self.component_error_names)
            ),
            dim=2,
        )
        expected_components = torch.where(
            self.target_valid_mask.unsqueeze(-1),
            expected_components,
            torch.zeros_like(expected_components),
        )
        if not torch.allclose(
            self.component_errors, expected_components, atol=1e-6, rtol=1e-6
        ):
            raise ActualTargetExportError(
                "component_errors are inconsistent with object values and projected support"
            )
        object_union = 1.0 - torch.prod(1.0 - effective_components, dim=-1)
        expected_severity = 1.0 - torch.prod(
            1.0 - all_support * object_union.view(1, 1, -1), dim=-1
        )
        expected_severity = torch.where(
            self.target_valid_mask,
            expected_severity,
            torch.zeros_like(expected_severity),
        )
        if not torch.allclose(
            self.error_severity_target, expected_severity, atol=1e-6, rtol=1e-6
        ):
            raise ActualTargetExportError(
                "error_severity_target is inconsistent with object values and projected support"
            )
        thresholds = self.object_component_values.new_tensor(
            self.failure_event_policy.component_thresholds
        )
        hard_object_events = self.object_component_valid & (
            self.object_component_values >= thresholds.view(1, -1)
        )
        expected_event = (
            (
                all_support.unsqueeze(-1)
                >= float(self.failure_event_policy.minimum_patch_support)
            )
            & hard_object_events.view(
                1, 1, gt_count + pred_count, len(self.component_error_names)
            )
        ).any(dim=2).any(dim=-1) & self.target_valid_mask
        if not torch.equal(self.failure_event_target, expected_event):
            raise ActualTargetExportError(
                "failure_event_target is inconsistent with object events and support gate"
            )
        _float_tensor(self.gt_motion_occupancy, "gt_motion_occupancy", unit_interval=True)
        _float_tensor(self.selected_motion_occupancy, "selected_motion_occupancy", unit_interval=True)
        if self.gt_motion_occupancy.ndim < 4 or self.gt_motion_occupancy.shape[0] != gt_count:
            raise ActualTargetExportError("stored GT occupancy must have shape [G,T,H,W,...]")
        if self.selected_motion_occupancy.shape != (pred_count, *self.gt_motion_occupancy.shape[1:]):
            raise ActualTargetExportError("stored selected occupancy must have shape [N,T,H,W,...]")
        _bool_tensor(
            self.gt_motion_valid_mask,
            "gt_motion_valid_mask",
            (gt_count, self.gt_motion_occupancy.shape[1]),
        )
        _integer_vector(self.gt_traffic_states, "gt_traffic_states", gt_count)
        _bool_tensor(
            self.gt_traffic_state_valid,
            "gt_traffic_state_valid",
            (gt_count,),
        )
        _text(
            self.occupancy_rasterizer_id,
            "predicted occupancy_rasterizer_id",
        )
        _text(
            self.gt_occupancy_rasterizer_id,
            "gt_occupancy_rasterizer_id",
        )
        _integer_vector(self.source_query_index, "source_query_index", pred_count)
        _float_tensor(self.decoded_scores, "decoded_scores", ndim=1, unit_interval=True)
        if self.decoded_scores.shape != (pred_count,):
            raise ActualTargetExportError("decoded_scores count must match pred_to_gt")
        _float_tensor(self.full_class_sigmoid, "full_class_sigmoid", ndim=2, unit_interval=True)
        if self.full_class_sigmoid.shape[0] != pred_count:
            raise ActualTargetExportError("full_class_sigmoid count must match pred_to_gt")
        _float_tensor(self.traffic_state_logits, "traffic_state_logits", ndim=2)
        if self.traffic_state_logits.shape != (pred_count, 4):
            raise ActualTargetExportError(
                "stored ORION v1 traffic_state_logits must have shape [N,4]"
            )
        if self.traffic_probability_transform != "sigmoid":
            raise ActualTargetExportError(
                "stored ORION v1 traffic probabilities must use sigmoid"
            )
        _float_tensor(self.all_trajectory_modes, "all_trajectory_modes", ndim=4)
        _float_tensor(self.trajectory_mode_scores, "trajectory_mode_scores", ndim=2)
        if self.all_trajectory_modes.shape[0] != pred_count or self.all_trajectory_modes.shape[-1] != 2:
            raise ActualTargetExportError("all_trajectory_modes must have shape [N,M,T,2]")
        if self.trajectory_mode_scores.shape != self.all_trajectory_modes.shape[:2]:
            raise ActualTargetExportError("trajectory mode tensors disagree")
        _integer_vector(
            self.selected_motion_mode_index,
            "selected_motion_mode_index",
            pred_count,
        )
        if pred_count:
            row = torch.arange(pred_count, device=self.decoded_scores.device)
            if torch.any(self.decoded_classes < 0) or int(self.decoded_classes.max()) >= self.full_class_sigmoid.shape[1]:
                raise ActualTargetExportError("stored decoded class ID is invalid")
            if not torch.allclose(
                self.decoded_scores,
                self.full_class_sigmoid[row, self.decoded_classes.long()],
                atol=1e-6,
                rtol=1e-5,
            ):
                raise ActualTargetExportError("stored decoded score/class parity failed")
            if not torch.equal(
                self.selected_motion_mode_index.long(),
                self.trajectory_mode_scores.argmax(dim=1),
            ):
                raise ActualTargetExportError("stored selected-mode parity failed")
        duplicate_actual = self.source_query_index.unique().numel() != pred_count
        if bool(self.duplicate_source_queries_present) != bool(duplicate_actual):
            raise ActualTargetExportError(
                "duplicate_source_queries_present disagrees with source_query_index"
            )
        for query_index in self.source_query_index.unique().tolist():
            rows = torch.nonzero(
                self.source_query_index == query_index, as_tuple=False
            ).flatten()
            if rows.numel() <= 1:
                continue
            reference = rows[0]
            for value, name in (
                (self.decoded_boxes_lidar, "decoded boxes"),
                (self.full_class_sigmoid, "full class sigmoid"),
                (self.selected_motion_occupancy, "selected motion occupancy"),
                (self.traffic_state_logits, "traffic logits"),
                (self.all_trajectory_modes, "all trajectory modes"),
                (self.trajectory_mode_scores, "trajectory mode scores"),
            ):
                if not torch.allclose(
                    value[rows],
                    value[reference].expand_as(value[rows]),
                    atol=1e-6,
                    rtol=1e-5,
                ):
                    raise ActualTargetExportError(
                        f"stored duplicate source query has inconsistent {name}"
                    )
        if isinstance(self.decoder_layer, bool) or not isinstance(self.decoder_layer, int) or self.decoder_layer < 0:
            raise ActualTargetExportError("stored decoder_layer is invalid")
        if self.decoder_flatten_policy != "sigmoid_query_class_topk":
            raise ActualTargetExportError("stored decoder flatten policy is incompatible")
        if isinstance(self.decoder_topk, bool) or not isinstance(self.decoder_topk, int) or self.decoder_topk <= 0:
            raise ActualTargetExportError("stored decoder_topk is invalid")
        if self.with_light_state is not True:
            raise ActualTargetExportError("stored bundle must assert with_light_state=True")
        if self.motion_mode_policy != MOTION_MODE_POLICY:
            raise ActualTargetExportError("stored motion target must use selected mode")
        invalid = ~self.target_valid_mask
        if bool(invalid.any()):
            if torch.any(self.error_severity_target[invalid] != 0):
                raise ActualTargetExportError("invalid severity cells must use zero placeholders")
            if torch.any(self.failure_event_target[invalid] != 0):
                raise ActualTargetExportError("invalid event cells must use zero placeholders")
            if torch.any(self.component_errors[invalid] != 0):
                raise ActualTargetExportError("invalid component cells must use zero placeholders")
        if self.target_provenance.camera_order != self.support_provenance.camera_order:
            raise ActualTargetExportError("branch camera order disagrees with support provenance")
        if self.target_provenance.image_transform_id != self.support_provenance.image_transform_id:
            raise ActualTargetExportError("branch image transform disagrees with support provenance")
        if self.target_provenance.route_id != self.chronology.route_id or self.target_provenance.frame_idx != self.chronology.frame_idx:
            raise ActualTargetExportError("target provenance and chronology frame identity disagree")
        if self.target_provenance.temporal_history_id != self.chronology.branch_history_id:
            raise ActualTargetExportError(
                "target temporal_history_id must equal branch_history_id"
            )
        if self.target_provenance.paired_history_protocol_id != self.chronology.paired_replay_id:
            raise ActualTargetExportError(
                "target paired_history_protocol_id must equal paired_replay_id"
            )
        branch = self.target_provenance.observation_branch
        if branch == "clean" and self.observation.corruption_family != "none":
            raise ActualTargetExportError("clean branch cannot declare a corruption")
        if branch == "clean" and self.chronology.history_content not in ("clean", "shared"):
            raise ActualTargetExportError("clean branch has incompatible history_content")
        if branch == "observed" and self.chronology.history_content not in ("observed", "shared"):
            raise ActualTargetExportError("observed branch has incompatible history_content")
        if self.observation.event_window_frames is not None:
            start, end = self.observation.event_window_frames
            expected_active = start <= self.chronology.frame_idx <= end
            if self.observation.active_at_frame != expected_active:
                raise ActualTargetExportError(
                    "active_at_frame disagrees with event_window_frames/frame_idx"
                )


@dataclass(frozen=True)
class PairedActualTargetBundleV1:
    """Comparable observed/clean branch targets for one recorded state."""

    bundle_id: str
    observed: ActualTargetBranchBundleV1
    clean: ActualTargetBranchBundleV1
    delta_error: torch.Tensor
    paired_valid_mask: torch.Tensor
    real_orion_hook_executed: bool = False
    patch_attribution_is_causal: bool = False
    schema_version: str = PAIRED_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_BUNDLE_SCHEMA_VERSION:
            raise ActualTargetExportError(
                f"unsupported paired bundle schema {self.schema_version!r}"
            )
        _text(self.bundle_id, "bundle_id")
        if self.observed.target_provenance.observation_branch != "observed" or self.clean.target_provenance.observation_branch != "clean":
            raise ActualTargetExportError("paired bundle requires observed and clean branches")
        if self.patch_attribution_is_causal:
            raise ActualTargetExportError("v1 projected patch attribution cannot be causal")
        shape = tuple(self.observed.error_severity_target.shape)
        _float_tensor(self.delta_error, "delta_error", ndim=2, unit_interval=True)
        if tuple(self.delta_error.shape) != shape or tuple(self.clean.error_severity_target.shape) != shape:
            raise ActualTargetExportError("paired branch and delta shapes must match")
        _bool_tensor(self.paired_valid_mask, "paired_valid_mask", shape)
        expected_valid = self.observed.target_valid_mask & self.clean.target_valid_mask
        if not torch.equal(self.paired_valid_mask, expected_valid):
            raise ActualTargetExportError("paired_valid_mask must be the branch-mask intersection")
        expected_delta = torch.where(
            expected_valid,
            torch.relu(self.observed.error_severity_target - self.clean.error_severity_target),
            torch.zeros_like(self.delta_error),
        )
        if not torch.allclose(self.delta_error, expected_delta):
            raise ActualTargetExportError("delta_error must equal relu(E_obs-E_clean)")
        _validate_branch_comparability(self.observed, self.clean)


def _validate_branch_comparability(
    observed: ActualTargetBranchBundleV1,
    clean: ActualTargetBranchBundleV1,
) -> None:
    target_fields = (
        "base_checkpoint_sha256",
        "inference_config_sha256",
        "git_revision",
        "route_id",
        "town",
        "frame_idx",
        "paired_history_protocol_id",
        "class_mapping_id",
        "decoder_policy_id",
        "camera_order",
        "image_transform_id",
        "target_version",
        "target_provenance",
    )
    disagreements = [
        name
        for name in target_fields
        if getattr(observed.target_provenance, name)
        != getattr(clean.target_provenance, name)
    ]
    chronology_fields = (
        "route_id",
        "frame_idx",
        "sequence_index",
        "source_timestamp_s",
        "history_start_frame_idx",
        "paired_replay_id",
        "previous_frame_idx",
    )
    disagreements.extend(
        f"chronology.{name}"
        for name in chronology_fields
        if getattr(observed.chronology, name) != getattr(clean.chronology, name)
    )
    if observed.support_provenance != clean.support_provenance:
        disagreements.append("support_provenance")
    if not torch.equal(observed.gt_centers_lidar, clean.gt_centers_lidar):
        disagreements.append("gt_centers_lidar")
    if not torch.equal(observed.gt_boxes_lidar, clean.gt_boxes_lidar):
        disagreements.append("gt_boxes_lidar")
    if not torch.equal(observed.gt_classes, clean.gt_classes):
        disagreements.append("gt_classes")
    if not torch.equal(observed.gt_projected_support, clean.gt_projected_support):
        disagreements.append("gt_projected_support")
    if observed.failure_event_policy != clean.failure_event_policy:
        disagreements.append("failure_event_policy")
    if observed.bev_iou_policy_id != clean.bev_iou_policy_id:
        disagreements.append("bev_iou_policy_id")
    if observed.occupancy_rasterizer_id != clean.occupancy_rasterizer_id:
        disagreements.append("predicted_occupancy_rasterizer_id")
    if (
        observed.gt_occupancy_rasterizer_id
        != clean.gt_occupancy_rasterizer_id
    ):
        disagreements.append("gt_occupancy_rasterizer_id")
    if observed.component_error_names != clean.component_error_names:
        disagreements.append("component_error_names")
    histories = (observed.chronology.history_content, clean.chronology.history_content)
    if histories not in (("observed", "clean"), ("shared", "shared")):
        disagreements.append("chronology.history_content")
    if disagreements:
        raise ActualTargetExportError(
            "clean/observed branches are not comparable: " + ", ".join(disagreements)
        )


def build_actual_target_branch(
    decoded: DecodedORIONFrameV1,
    ground_truth: PrivilegedGroundTruthFrameV1,
    pairwise_bev_iou: torch.Tensor,
    gt_projected_support: torch.Tensor,
    pred_projected_support: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    *,
    support_provenance: PatchSupportProvenanceV1,
    target_provenance: TargetProvenanceV1,
    chronology: FrameChronologyV1,
    observation: ObservationConditionV1,
    failure_event_policy: FailureEventPolicyV1,
    bev_iou_policy_id: str,
    max_center_distance: float | torch.Tensor = 4.0,
    minimum_prediction_score: float = 0.0,
    bev_occupancy_sidecar: Optional[BEVOccupancyErrorSidecarV1] = None,
) -> ActualTargetBranchBundleV1:
    """Build one branch from decoded outputs and precomputed support proxies."""

    if decoded.decoder_policy_id != target_provenance.decoder_policy_id:
        raise ActualTargetExportError("decoded and target decoder_policy_id disagree")
    if decoded.class_mapping_id != target_provenance.class_mapping_id:
        raise ActualTargetExportError("decoded and target class_mapping_id disagree")
    # GT future occupancy and decoded selected-mode occupancy intentionally use
    # different algorithms.  Their IDs are retained independently; exact
    # time/grid shape below establishes whether the tensors are comparable.
    normalized_bev_iou_policy_id = _text(
        bev_iou_policy_id, "bev_iou_policy_id"
    )
    if decoded.centers_lidar.shape[1] != ground_truth.centers_lidar.shape[1]:
        raise ActualTargetExportError("decoded/GT center coordinate dimensions disagree")
    if decoded.selected_motion_occupancy.shape[1:] != ground_truth.motion_occupancy.shape[1:]:
        raise ActualTargetExportError("decoded/GT occupancy time/spatial shapes disagree")
    _float_tensor(pairwise_bev_iou, "pairwise_bev_iou", ndim=2, unit_interval=True)
    gt_count = ground_truth.centers_lidar.shape[0]
    pred_count = decoded.centers_lidar.shape[0]
    if pairwise_bev_iou.shape != (gt_count, pred_count):
        raise ActualTargetExportError(
            f"pairwise_bev_iou must have shape [{gt_count}, {pred_count}]"
        )
    _float_tensor(gt_projected_support, "gt_projected_support", ndim=3, unit_interval=True)
    _float_tensor(pred_projected_support, "pred_projected_support", ndim=3, unit_interval=True)
    views = len(support_provenance.camera_order)
    patches = support_provenance.patch_count
    if gt_projected_support.shape != (views, patches, gt_count):
        raise ActualTargetExportError(
            f"gt_projected_support must have shape [{views}, {patches}, {gt_count}]"
        )
    if pred_projected_support.shape != (views, patches, pred_count):
        raise ActualTargetExportError(
            f"pred_projected_support must have shape [{views}, {patches}, {pred_count}]"
        )
    _bool_tensor(patch_valid_mask, "patch_valid_mask", (views, patches))
    device = decoded.centers_lidar.device
    tensors = (
        ground_truth.centers_lidar,
        pairwise_bev_iou,
        gt_projected_support,
        pred_projected_support,
        patch_valid_mask,
    )
    if any(value.device != device for value in tensors):
        raise ActualTargetExportError("decoded, GT, IoU, support, and mask must share a device")

    try:
        match = class_aware_distance_gated_match(
            ground_truth.centers_lidar,
            ground_truth.classes,
            decoded.centers_lidar,
            decoded.classes,
            decoded.scores,
            coordinate_frame="lidar",
            max_center_distance=max_center_distance,
            minimum_prediction_score=minimum_prediction_score,
        )
        components = compute_object_failure_components(
            match,
            ground_truth.classes,
            decoded.class_probabilities,
            decoded.scores,
            pairwise_bev_iou=pairwise_bev_iou,
            gt_motion_occupancy=ground_truth.motion_occupancy,
            pred_motion_occupancy=decoded.selected_motion_occupancy,
            motion_valid_mask=ground_truth.motion_valid_mask,
            motion_mode_policy=decoded.motion_mode_policy,
            gt_traffic_states=ground_truth.traffic_states,
            pred_traffic_state_probabilities=decoded.traffic_state_probabilities,
            traffic_state_valid=ground_truth.traffic_state_valid,
        )
    except ObjectFailureTargetError as error:
        raise ActualTargetExportError(str(error)) from error

    all_support = torch.cat((gt_projected_support, pred_projected_support), dim=-1)
    all_error = torch.cat((components.soft_union, components.false_positive_error))
    all_valid = torch.cat((components.soft_union_valid, components.false_positive_valid))
    try:
        severity = aggregate_projected_visible_support(
            all_support,
            all_error,
            all_valid,
            patch_valid_mask,
            support_provenance=support_provenance,
            target_provenance=target_provenance,
        )
    except ObjectFailureTargetError as error:
        raise ActualTargetExportError(str(error)) from error

    component_names = components.component_names + ("false_positive",)
    object_component_values = torch.zeros(
        (gt_count + pred_count, len(component_names)),
        device=device,
        dtype=components.values.dtype,
    )
    object_component_valid = torch.zeros(
        (gt_count + pred_count, len(component_names)),
        device=device,
        dtype=torch.bool,
    )
    object_component_values[:gt_count, : len(components.component_names)] = (
        components.values
    )
    object_component_valid[:gt_count, : len(components.component_names)] = (
        components.valid
    )
    object_component_values[gt_count:, -1] = components.false_positive_error
    object_component_valid[gt_count:, -1] = components.false_positive_valid
    patch_components = []
    for index in range(len(component_names)):
        if index < len(components.component_names):
            object_values = torch.cat(
                (
                    components.values[:, index],
                    torch.zeros_like(components.false_positive_error),
                )
            )
            object_valid = torch.cat(
                (
                    components.valid[:, index],
                    torch.zeros_like(components.false_positive_valid),
                )
            )
        else:
            object_values = torch.cat(
                (
                    torch.zeros_like(components.soft_union),
                    components.false_positive_error,
                )
            )
            object_valid = torch.cat(
                (
                    torch.zeros_like(components.soft_union_valid),
                    components.false_positive_valid,
                )
            )
        projected = aggregate_projected_visible_support(
            all_support,
            object_values,
            object_valid,
            patch_valid_mask,
            support_provenance=support_provenance,
            target_provenance=target_provenance,
        )
        patch_components.append(projected.error)
    component_errors = torch.stack(patch_components, dim=-1)

    if component_names != failure_event_policy.component_names:
        raise ActualTargetExportError(
            "failure-event policy component order disagrees with exporter"
        )
    component_thresholds = component_errors.new_tensor(
        failure_event_policy.component_thresholds
    )
    hard_object_component_event = object_component_valid & (
        object_component_values >= component_thresholds.view(1, -1)
    )
    projected_hard_event = (
        all_support.unsqueeze(-1)
        >= float(failure_event_policy.minimum_patch_support)
    ) & hard_object_component_event.view(
        1, 1, gt_count + pred_count, len(component_names)
    )
    failure_event = projected_hard_event.any(dim=2).any(dim=-1) & patch_valid_mask

    return ActualTargetBranchBundleV1(
        error_severity_target=severity.error.detach().cpu().float(),
        failure_event_target=failure_event.detach().cpu(),
        target_valid_mask=patch_valid_mask.detach().cpu(),
        component_errors=component_errors.detach().cpu().float(),
        component_error_names=component_names,
        object_component_values=object_component_values.detach().cpu().float(),
        object_component_valid=object_component_valid.detach().cpu(),
        gt_to_pred=components.match.gt_to_pred.detach().cpu(),
        pred_to_gt=components.match.pred_to_gt.detach().cpu(),
        matched_center_distance=components.match.matched_center_distance.detach().cpu().float(),
        valid_pair_mask=components.match.valid_pair_mask.detach().cpu(),
        max_center_distance=components.match.max_center_distance.detach().cpu().float(),
        minimum_prediction_score=components.match.minimum_prediction_score,
        match_policy_id=MATCH_POLICY_ID,
        gt_centers_lidar=ground_truth.centers_lidar.detach().cpu().float(),
        gt_boxes_lidar=ground_truth.boxes_lidar.detach().cpu().float(),
        gt_classes=ground_truth.classes.detach().cpu(),
        decoded_centers_lidar=decoded.centers_lidar.detach().cpu().float(),
        decoded_boxes_lidar=decoded.boxes_lidar.detach().cpu().float(),
        decoded_classes=decoded.classes.detach().cpu(),
        pairwise_bev_iou=pairwise_bev_iou.detach().cpu().float(),
        bev_iou_policy_id=normalized_bev_iou_policy_id,
        gt_projected_support=gt_projected_support.detach().cpu().float(),
        pred_projected_support=pred_projected_support.detach().cpu().float(),
        gt_motion_occupancy=ground_truth.motion_occupancy.detach().cpu().float(),
        gt_motion_valid_mask=ground_truth.motion_valid_mask.detach().cpu(),
        selected_motion_occupancy=decoded.selected_motion_occupancy.detach().cpu().float(),
        gt_traffic_states=ground_truth.traffic_states.detach().cpu(),
        gt_traffic_state_valid=ground_truth.traffic_state_valid.detach().cpu(),
        occupancy_rasterizer_id=decoded.occupancy_rasterizer_id,
        gt_occupancy_rasterizer_id=ground_truth.occupancy_rasterizer_id,
        source_query_index=decoded.source_query_index.detach().cpu(),
        decoded_scores=decoded.scores.detach().cpu().float(),
        full_class_sigmoid=decoded.class_probabilities.detach().cpu().float(),
        traffic_state_logits=decoded.traffic_state_logits.detach().cpu().float(),
        traffic_probability_transform=decoded.traffic_probability_transform,
        all_trajectory_modes=decoded.all_trajectory_modes.detach().cpu().float(),
        trajectory_mode_scores=decoded.trajectory_mode_scores.detach().cpu().float(),
        selected_motion_mode_index=decoded.selected_motion_mode_index.detach().cpu(),
        duplicate_source_queries_present=decoded.duplicate_source_queries_present,
        decoder_layer=decoded.decoder_layer,
        decoder_flatten_policy=decoded.decoder_flatten_policy,
        decoder_topk=decoded.decoder_topk,
        with_light_state=decoded.with_light_state,
        motion_mode_policy=decoded.motion_mode_policy,
        failure_event_policy=failure_event_policy,
        support_provenance=support_provenance,
        target_provenance=target_provenance,
        chronology=chronology,
        observation=observation,
        bev_occupancy_sidecar=_cpu_bev_sidecar(bev_occupancy_sidecar),
    )


def pair_actual_target_branches(
    observed: ActualTargetBranchBundleV1,
    clean: ActualTargetBranchBundleV1,
    *,
    bundle_id: str,
    real_orion_hook_executed: bool = False,
) -> PairedActualTargetBundleV1:
    """Pair comparable branches and compute corruption-attributable DeltaE."""

    _validate_branch_comparability(observed, clean)
    if real_orion_hook_executed and (
        "mock" in observed.failure_event_policy.calibration_policy_id.lower()
        or "unfitted" in observed.failure_event_policy.calibration_policy_id.lower()
    ):
        raise ActualTargetExportError(
            "a real-hook bundle cannot use mock/unfitted failure-event thresholds"
        )
    paired_valid = observed.target_valid_mask & clean.target_valid_mask
    delta = torch.where(
        paired_valid,
        torch.relu(observed.error_severity_target - clean.error_severity_target),
        torch.zeros_like(observed.error_severity_target),
    )
    return PairedActualTargetBundleV1(
        bundle_id=bundle_id,
        observed=observed,
        clean=clean,
        delta_error=delta,
        paired_valid_mask=paired_valid,
        real_orion_hook_executed=bool(real_orion_hook_executed),
        patch_attribution_is_causal=False,
    )


def bridge_actual_target_bundle_to_v2_record(
    bundle: PairedActualTargetBundleV1,
    observed_patch_features: torch.Tensor,
    clean_patch_features: torch.Tensor,
    *,
    record_id: str,
    pair_id: str,
    corruption_mask: Optional[torch.Tensor] = None,
    ensemble_teacher_variance: Optional[torch.Tensor] = None,
) -> PairedSpatialFeatureRecord:
    """Attach a target bundle to features using the Stage-1 v2 contract."""

    _float_tensor(observed_patch_features, "observed_patch_features", ndim=3)
    _float_tensor(clean_patch_features, "clean_patch_features", ndim=3)
    if observed_patch_features.shape != clean_patch_features.shape:
        raise ActualTargetExportError("observed and clean feature shapes must match")
    if tuple(observed_patch_features.shape[:2]) != tuple(bundle.observed.error_severity_target.shape):
        raise ActualTargetExportError("feature [V,P] shape must match target bundle")
    if observed_patch_features.shape[-1] <= 0:
        raise ActualTargetExportError("feature dimension must be positive")
    if corruption_mask is not None:
        _float_tensor(corruption_mask, "corruption_mask", ndim=2, unit_interval=True)
        if corruption_mask.shape != bundle.observed.error_severity_target.shape:
            raise ActualTargetExportError("corruption_mask shape must match target")
    if ensemble_teacher_variance is not None:
        _float_tensor(ensemble_teacher_variance, "ensemble_teacher_variance", ndim=2)
        if ensemble_teacher_variance.shape != bundle.observed.error_severity_target.shape or torch.any(ensemble_teacher_variance < 0):
            raise ActualTargetExportError(
                "ensemble_teacher_variance must be non-negative and match target shape"
            )
    observed_provenance = bundle.observed.target_provenance
    metadata = {
        "actual_target_bundle": {
            "schema_version": bundle.schema_version,
            "branch_schema_version": bundle.observed.schema_version,
            "bundle_id": bundle.bundle_id,
            "target_version": OBJECT_FAILURE_TARGET_VERSION,
            "target_provenance": ACTUAL_TARGET_PROVENANCE,
            "patch_attribution": bundle.observed.support_provenance.attribution,
            "patch_attribution_is_causal": False,
            "real_orion_hook_executed": bundle.real_orion_hook_executed,
            "delta_error_retained_in_source_bundle": True,
            "failure_event_policy": _event_policy_payload(
                bundle.observed.failure_event_policy
            ),
            "predicted_occupancy_rasterizer_id": (
                bundle.observed.occupancy_rasterizer_id
            ),
            "gt_occupancy_rasterizer_id": (
                bundle.observed.gt_occupancy_rasterizer_id
            ),
            "bev_iou_policy_id": bundle.observed.bev_iou_policy_id,
            "match_policy_id": bundle.observed.match_policy_id,
            "minimum_prediction_score": bundle.observed.minimum_prediction_score,
            "max_center_distance_retained_per_gt": True,
            "full_gt_and_decoded_boxes_retained": True,
            "motion_reproduction_tensors_retained": {
                "gt_motion_occupancy": True,
                "gt_motion_valid_mask": True,
                "selected_motion_occupancy": True,
            },
        },
        "source_identity": {
            "route_id": observed_provenance.route_id,
            "town": observed_provenance.town,
            "frame_idx": observed_provenance.frame_idx,
            "paired_replay_id": bundle.observed.chronology.paired_replay_id,
            "observed_branch_history_id": bundle.observed.chronology.branch_history_id,
            "clean_branch_history_id": bundle.clean.chronology.branch_history_id,
            "chronology_schema_version": bundle.observed.chronology.schema_version,
        },
        "claim_boundary": {
            "actual_frozen_orion_task_error": True,
            "projected_patch_attribution_is_causal": False,
            "supports_llm_understanding_claim": False,
            "real_orion_hook_completed": bundle.real_orion_hook_executed,
        },
    }
    return PairedSpatialFeatureRecord(
        record_id=_text(record_id, "record_id"),
        pair_id=_text(pair_id, "pair_id"),
        route_id=observed_provenance.route_id,
        town=observed_provenance.town,
        severity=float(bundle.observed.observation.severity),
        observed_patch_features=observed_patch_features.detach().cpu().float(),
        clean_patch_features=clean_patch_features.detach().cpu().float(),
        error_severity_target=bundle.observed.error_severity_target,
        failure_event_target=bundle.observed.failure_event_target,
        target_valid_mask=bundle.observed.target_valid_mask,
        clean_error_severity_target=bundle.clean.error_severity_target,
        clean_failure_event_target=bundle.clean.failure_event_target,
        clean_target_valid_mask=bundle.clean.target_valid_mask,
        component_errors=bundle.observed.component_errors,
        clean_component_errors=bundle.clean.component_errors,
        component_error_names=bundle.observed.component_error_names,
        component_error_axis=-1,
        corruption_mask=(
            corruption_mask.detach().cpu().float()
            if corruption_mask is not None
            else None
        ),
        ensemble_teacher_variance=(
            ensemble_teacher_variance.detach().cpu().float()
            if ensemble_teacher_variance is not None
            else None
        ),
        metadata=metadata,
    )


def _cpu_bev_sidecar(
    sidecar: Optional[BEVOccupancyErrorSidecarV1],
) -> Optional[BEVOccupancyErrorSidecarV1]:
    if sidecar is None:
        return None
    return BEVOccupancyErrorSidecarV1(
        gt_occupancy=sidecar.gt_occupancy.detach().cpu().float(),
        predicted_occupancy=sidecar.predicted_occupancy.detach().cpu().float(),
        absolute_error=sidecar.absolute_error.detach().cpu().float(),
        valid_mask=sidecar.valid_mask.detach().cpu(),
        gt_provenance=sidecar.gt_provenance,
        prediction_provenance=sidecar.prediction_provenance,
        coordinate_frame=sidecar.coordinate_frame,
        bev_bounds_xyxy=sidecar.bev_bounds_xyxy,
        resolution_m=sidecar.resolution_m,
        schema_version=sidecar.schema_version,
    )


def _support_provenance_payload(value: PatchSupportProvenanceV1) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "camera_order": list(value.camera_order),
        "image_hw": list(value.image_hw),
        "patch_hw": list(value.patch_hw),
        "image_transform_id": value.image_transform_id,
        "coordinate_frame": value.coordinate_frame,
        "projection_matrix_kind": value.projection_matrix_kind,
        "attribution": value.attribution,
    }


def _target_provenance_payload(value: TargetProvenanceV1) -> dict[str, Any]:
    return {
        "target_version": value.target_version,
        "target_provenance": value.target_provenance,
        "base_checkpoint_sha256": value.base_checkpoint_sha256,
        "inference_config_sha256": value.inference_config_sha256,
        "git_revision": value.git_revision,
        "route_id": value.route_id,
        "town": value.town,
        "frame_idx": value.frame_idx,
        "observation_branch": value.observation_branch,
        "temporal_history_id": value.temporal_history_id,
        "paired_history_protocol_id": value.paired_history_protocol_id,
        "class_mapping_id": value.class_mapping_id,
        "decoder_policy_id": value.decoder_policy_id,
        "camera_order": list(value.camera_order),
        "image_transform_id": value.image_transform_id,
    }


def _chronology_payload(value: FrameChronologyV1) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "route_id": value.route_id,
        "frame_idx": value.frame_idx,
        "sequence_index": value.sequence_index,
        "source_timestamp_s": value.source_timestamp_s,
        "history_start_frame_idx": value.history_start_frame_idx,
        "branch_history_id": value.branch_history_id,
        "paired_replay_id": value.paired_replay_id,
        "history_content": value.history_content,
        "previous_frame_idx": value.previous_frame_idx,
    }


def _observation_payload(value: ObservationConditionV1) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "corruption_family": value.corruption_family,
        "severity": value.severity,
        "seed": value.seed,
        "event_window_frames": (
            list(value.event_window_frames)
            if value.event_window_frames is not None
            else None
        ),
        "active_at_frame": value.active_at_frame,
    }


def _event_policy_payload(value: FailureEventPolicyV1) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "component_names": list(value.component_names),
        "component_thresholds": list(value.component_thresholds),
        "minimum_patch_support": value.minimum_patch_support,
        "calibration_policy_id": value.calibration_policy_id,
    }


def _sidecar_payload(value: Optional[BEVOccupancyErrorSidecarV1]) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    return {
        "schema_version": value.schema_version,
        "gt_occupancy": value.gt_occupancy,
        "predicted_occupancy": value.predicted_occupancy,
        "absolute_error": value.absolute_error,
        "valid_mask": value.valid_mask,
        "gt_provenance": value.gt_provenance,
        "prediction_provenance": value.prediction_provenance,
        "coordinate_frame": value.coordinate_frame,
        "bev_bounds_xyxy": list(value.bev_bounds_xyxy),
        "resolution_m": value.resolution_m,
    }


def _branch_payload(value: ActualTargetBranchBundleV1) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "error_severity_target": value.error_severity_target,
        "failure_event_target": value.failure_event_target,
        "target_valid_mask": value.target_valid_mask,
        "component_errors": value.component_errors,
        "component_error_names": list(value.component_error_names),
        "object_component_values": value.object_component_values,
        "object_component_valid": value.object_component_valid,
        "gt_to_pred": value.gt_to_pred,
        "pred_to_gt": value.pred_to_gt,
        "matched_center_distance": value.matched_center_distance,
        "valid_pair_mask": value.valid_pair_mask,
        "max_center_distance": value.max_center_distance,
        "minimum_prediction_score": value.minimum_prediction_score,
        "match_policy_id": value.match_policy_id,
        "gt_centers_lidar": value.gt_centers_lidar,
        "gt_boxes_lidar": value.gt_boxes_lidar,
        "gt_classes": value.gt_classes,
        "decoded_centers_lidar": value.decoded_centers_lidar,
        "decoded_boxes_lidar": value.decoded_boxes_lidar,
        "decoded_classes": value.decoded_classes,
        "pairwise_bev_iou": value.pairwise_bev_iou,
        "bev_iou_policy_id": value.bev_iou_policy_id,
        "gt_projected_support": value.gt_projected_support,
        "pred_projected_support": value.pred_projected_support,
        "gt_motion_occupancy": value.gt_motion_occupancy,
        "gt_motion_valid_mask": value.gt_motion_valid_mask,
        "selected_motion_occupancy": value.selected_motion_occupancy,
        "gt_traffic_states": value.gt_traffic_states,
        "gt_traffic_state_valid": value.gt_traffic_state_valid,
        "occupancy_rasterizer_id": value.occupancy_rasterizer_id,
        "gt_occupancy_rasterizer_id": value.gt_occupancy_rasterizer_id,
        "source_query_index": value.source_query_index,
        "decoded_scores": value.decoded_scores,
        "full_class_sigmoid": value.full_class_sigmoid,
        "traffic_state_logits": value.traffic_state_logits,
        "traffic_probability_transform": value.traffic_probability_transform,
        "all_trajectory_modes": value.all_trajectory_modes,
        "trajectory_mode_scores": value.trajectory_mode_scores,
        "selected_motion_mode_index": value.selected_motion_mode_index,
        "duplicate_source_queries_present": value.duplicate_source_queries_present,
        "decoder_layer": value.decoder_layer,
        "decoder_flatten_policy": value.decoder_flatten_policy,
        "decoder_topk": value.decoder_topk,
        "with_light_state": value.with_light_state,
        "motion_mode_policy": value.motion_mode_policy,
        "failure_event_policy": _event_policy_payload(value.failure_event_policy),
        "support_provenance": _support_provenance_payload(value.support_provenance),
        "target_provenance": _target_provenance_payload(value.target_provenance),
        "chronology": _chronology_payload(value.chronology),
        "observation": _observation_payload(value.observation),
        "bev_occupancy_sidecar": _sidecar_payload(value.bev_occupancy_sidecar),
    }


def _sidecar_from_payload(payload: Optional[Mapping[str, Any]]) -> Optional[BEVOccupancyErrorSidecarV1]:
    if payload is None:
        return None
    if payload.get("schema_version") != BEV_OCCUPANCY_SIDECAR_VERSION:
        raise ActualTargetExportError("incompatible BEV sidecar schema")
    return BEVOccupancyErrorSidecarV1(
        schema_version=str(payload["schema_version"]),
        gt_occupancy=payload["gt_occupancy"],
        predicted_occupancy=payload["predicted_occupancy"],
        absolute_error=payload["absolute_error"],
        valid_mask=payload["valid_mask"],
        gt_provenance=str(payload["gt_provenance"]),
        prediction_provenance=str(payload["prediction_provenance"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        bev_bounds_xyxy=tuple(payload["bev_bounds_xyxy"]),
        resolution_m=float(payload["resolution_m"]),
    )


def _branch_from_payload(payload: Mapping[str, Any]) -> ActualTargetBranchBundleV1:
    support_raw = payload["support_provenance"]
    target_raw = payload["target_provenance"]
    chronology_raw = payload["chronology"]
    observation_raw = payload["observation"]
    event_raw = payload["failure_event_policy"]
    return ActualTargetBranchBundleV1(
        schema_version=str(payload.get("schema_version", "")),
        error_severity_target=payload["error_severity_target"],
        failure_event_target=payload["failure_event_target"],
        target_valid_mask=payload["target_valid_mask"],
        component_errors=payload["component_errors"],
        component_error_names=tuple(payload["component_error_names"]),
        object_component_values=payload["object_component_values"],
        object_component_valid=payload["object_component_valid"],
        gt_to_pred=payload["gt_to_pred"],
        pred_to_gt=payload["pred_to_gt"],
        matched_center_distance=payload["matched_center_distance"],
        valid_pair_mask=payload["valid_pair_mask"],
        max_center_distance=payload["max_center_distance"],
        minimum_prediction_score=float(payload["minimum_prediction_score"]),
        match_policy_id=str(payload["match_policy_id"]),
        gt_centers_lidar=payload["gt_centers_lidar"],
        gt_boxes_lidar=payload["gt_boxes_lidar"],
        gt_classes=payload["gt_classes"],
        decoded_centers_lidar=payload["decoded_centers_lidar"],
        decoded_boxes_lidar=payload["decoded_boxes_lidar"],
        decoded_classes=payload["decoded_classes"],
        pairwise_bev_iou=payload["pairwise_bev_iou"],
        bev_iou_policy_id=str(payload["bev_iou_policy_id"]),
        gt_projected_support=payload["gt_projected_support"],
        pred_projected_support=payload["pred_projected_support"],
        gt_motion_occupancy=payload["gt_motion_occupancy"],
        gt_motion_valid_mask=payload["gt_motion_valid_mask"],
        selected_motion_occupancy=payload["selected_motion_occupancy"],
        gt_traffic_states=payload["gt_traffic_states"],
        gt_traffic_state_valid=payload["gt_traffic_state_valid"],
        occupancy_rasterizer_id=str(payload["occupancy_rasterizer_id"]),
        gt_occupancy_rasterizer_id=str(payload["gt_occupancy_rasterizer_id"]),
        source_query_index=payload["source_query_index"],
        decoded_scores=payload["decoded_scores"],
        full_class_sigmoid=payload["full_class_sigmoid"],
        traffic_state_logits=payload["traffic_state_logits"],
        traffic_probability_transform=str(payload["traffic_probability_transform"]),
        all_trajectory_modes=payload["all_trajectory_modes"],
        trajectory_mode_scores=payload["trajectory_mode_scores"],
        selected_motion_mode_index=payload["selected_motion_mode_index"],
        duplicate_source_queries_present=bool(
            payload["duplicate_source_queries_present"]
        ),
        decoder_layer=int(payload["decoder_layer"]),
        decoder_flatten_policy=str(payload["decoder_flatten_policy"]),
        decoder_topk=int(payload["decoder_topk"]),
        with_light_state=bool(payload["with_light_state"]),
        motion_mode_policy=str(payload["motion_mode_policy"]),
        failure_event_policy=FailureEventPolicyV1(
            **{
                **dict(event_raw),
                "component_names": tuple(event_raw["component_names"]),
                "component_thresholds": tuple(event_raw["component_thresholds"]),
            }
        ),
        support_provenance=PatchSupportProvenanceV1(
            **{
                **dict(support_raw),
                "camera_order": tuple(support_raw["camera_order"]),
                "image_hw": tuple(support_raw["image_hw"]),
                "patch_hw": tuple(support_raw["patch_hw"]),
            }
        ),
        target_provenance=TargetProvenanceV1(
            **{
                **dict(target_raw),
                "camera_order": tuple(target_raw["camera_order"]),
            }
        ),
        chronology=FrameChronologyV1(**dict(chronology_raw)),
        observation=ObservationConditionV1(
            **{
                **dict(observation_raw),
                "event_window_frames": (
                    tuple(observation_raw["event_window_frames"])
                    if observation_raw["event_window_frames"] is not None
                    else None
                ),
            }
        ),
        bev_occupancy_sidecar=_sidecar_from_payload(
            payload.get("bev_occupancy_sidecar")
        ),
    )


def save_paired_actual_target_bundle(
    path: Path | str,
    bundle: PairedActualTargetBundleV1,
) -> None:
    """Save tensors and primitive metadata; refuse implicit overwrites."""

    output = Path(path)
    if output.exists():
        raise ActualTargetExportError(f"refusing to overwrite existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": bundle.schema_version,
            "bundle_id": bundle.bundle_id,
            "observed": _branch_payload(bundle.observed),
            "clean": _branch_payload(bundle.clean),
            "delta_error": bundle.delta_error,
            "paired_valid_mask": bundle.paired_valid_mask,
            "real_orion_hook_executed": bundle.real_orion_hook_executed,
            "patch_attribution_is_causal": bundle.patch_attribution_is_causal,
        },
        output,
    )


def load_paired_actual_target_bundle(
    path: Path | str,
) -> PairedActualTargetBundleV1:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return PairedActualTargetBundleV1(
        schema_version=str(payload.get("schema_version", "")),
        bundle_id=str(payload["bundle_id"]),
        observed=_branch_from_payload(payload["observed"]),
        clean=_branch_from_payload(payload["clean"]),
        delta_error=payload["delta_error"],
        paired_valid_mask=payload["paired_valid_mask"],
        real_orion_hook_executed=bool(payload.get("real_orion_hook_executed", False)),
        patch_attribution_is_causal=bool(
            payload.get("patch_attribution_is_causal", False)
        ),
    )


def build_cpu_mock_actual_target_bundle(
    feature_dim: int = 8,
) -> tuple[PairedActualTargetBundleV1, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic CPU fixture; it does not execute a real model."""

    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ActualTargetExportError("feature_dim must be a positive integer")
    camera_order = ("CAM_FRONT", "CAM_LEFT")
    support_provenance = PatchSupportProvenanceV1(
        camera_order=camera_order,
        image_hw=(80, 120),
        patch_hw=(2, 3),
        image_transform_id="cpu-mock-post-augmentation-v1",
    )
    protocol_id = "cpu-mock-dual-chronological-replay/v1"

    def provenance(branch: str) -> TargetProvenanceV1:
        return TargetProvenanceV1(
            base_checkpoint_sha256="a" * 64,
            inference_config_sha256="b" * 64,
            git_revision="cpu-mock-no-real-hook",
            route_id="mock_route_001",
            town="Town01",
            frame_idx=42,
            observation_branch=branch,
            temporal_history_id=f"cpu-mock-{branch}-memory-content-frame-42",
            paired_history_protocol_id=protocol_id,
            class_mapping_id="cpu-mock-b2d-class-map-v1",
            decoder_policy_id="custom-nms-free-sigmoid-topk300-mock-v1",
            camera_order=camera_order,
            image_transform_id=support_provenance.image_transform_id,
        )

    def chronology(branch: str) -> FrameChronologyV1:
        return FrameChronologyV1(
            route_id="mock_route_001",
            frame_idx=42,
            sequence_index=42,
            source_timestamp_s=4.2,
            history_start_frame_idx=0,
            branch_history_id=f"cpu-mock-{branch}-memory-content-frame-42",
            paired_replay_id=protocol_id,
            history_content=branch,
            previous_frame_idx=41,
        )

    gt = PrivilegedGroundTruthFrameV1(
        centers_lidar=torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        boxes_lidar=torch.tensor(
            [
                [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            ]
        ),
        classes=torch.tensor([0, 1]),
        motion_occupancy=torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
            ]
        ),
        motion_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        traffic_states=torch.tensor([-1, 1]),
        traffic_state_valid=torch.tensor([False, True]),
        occupancy_rasterizer_id="cpu-mock-planningmetric-compatible-v1",
    )
    gt_support = torch.zeros(2, 6, 2)
    gt_support[0, 1, 0] = 1.0
    # Small projected support must not erase a hard object-level miss event.
    gt_support[0, 4, 1] = 0.10
    gt_support[1, 0, 0] = 0.5
    patch_valid = torch.ones(2, 6, dtype=torch.bool)
    patch_valid[1, 5] = False

    clean_modes = torch.zeros(2, 2, 2, 2)
    clean_modes[:, 1, :, 0] = 0.25
    clean_mode_scores = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    clean_decoded = DecodedORIONFrameV1(
        centers_lidar=gt.centers_lidar.clone(),
        boxes_lidar=gt.boxes_lidar.clone(),
        classes=torch.tensor([0, 1]),
        scores=torch.tensor([0.9, 0.9]),
        source_query_index=torch.tensor([7, 11]),
        class_probabilities=torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]),
        selected_motion_occupancy=gt.motion_occupancy.clone(),
        traffic_state_logits=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [-2.0, 2.0, -2.0, 2.0]]
        ),
        all_trajectory_modes=clean_modes,
        trajectory_mode_scores=clean_mode_scores,
        selected_motion_mode_index=clean_mode_scores.argmax(dim=1),
        occupancy_rasterizer_id="cpu-mock-planningmetric-compatible-v1",
        decoder_layer=5,
        decoder_policy_id="custom-nms-free-sigmoid-topk300-mock-v1",
        class_mapping_id="cpu-mock-b2d-class-map-v1",
        with_light_state=True,
    )
    clean_pred_support = gt_support.clone()
    clean_iou = torch.tensor([[0.9, 0.0], [0.0, 0.9]])

    observed_modes = torch.zeros(3, 2, 2, 2)
    observed_modes[:, 1, :, 0] = 0.5
    observed_mode_scores = torch.tensor(
        [[2.0, 1.0], [0.2, 1.2], [0.2, 1.2]]
    )
    repeated_query_probabilities = torch.tensor([0.1, 0.1, 0.8])
    observed_decoded = DecodedORIONFrameV1(
        centers_lidar=torch.tensor([[1.0, 0.0], [10.0, 0.0], [10.0, 0.0]]),
        boxes_lidar=torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                [10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            ]
        ),
        classes=torch.tensor([0, 2, 1]),
        scores=torch.tensor([0.6, 0.8, 0.1]),
        source_query_index=torch.tensor([7, 11, 11]),
        class_probabilities=torch.stack(
            (torch.tensor([0.6, 0.2, 0.1]), repeated_query_probabilities, repeated_query_probabilities)
        ),
        selected_motion_occupancy=torch.zeros(3, 2, 2, 2),
        traffic_state_logits=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0, 1.0],
            ]
        ),
        all_trajectory_modes=observed_modes,
        trajectory_mode_scores=observed_mode_scores,
        selected_motion_mode_index=observed_mode_scores.argmax(dim=1),
        occupancy_rasterizer_id="cpu-mock-planningmetric-compatible-v1",
        decoder_layer=5,
        decoder_policy_id="custom-nms-free-sigmoid-topk300-mock-v1",
        class_mapping_id="cpu-mock-b2d-class-map-v1",
        with_light_state=True,
    )
    observed_pred_support = torch.zeros(2, 6, 3)
    observed_pred_support[0, 1, 0] = 1.0
    # Query 11/class 2 is a high-confidence FP at a separate small support;
    # its low-score duplicate class 1 remains below the decode match threshold.
    observed_pred_support[0, 3, 1] = 0.10
    observed_pred_support[0, 4, 2] = 0.10
    observed_iou = torch.zeros(2, 3)
    observed_iou[0, 0] = 0.6
    observed_iou[1, 2] = 0.9

    gt_bev = gt.motion_occupancy.amax(dim=0)
    clean_bev = clean_decoded.selected_motion_occupancy.amax(dim=0)
    observed_bev = observed_decoded.selected_motion_occupancy.amax(dim=0)
    sidecar_kwargs = {
        "valid_mask": torch.ones_like(gt_bev, dtype=torch.bool),
        "gt_provenance": "CPU mock of PlanningMetric-rasterized B2D boxes",
        "prediction_provenance": "CPU mock rasterized selected ORION mode",
        "coordinate_frame": "ego_bev",
        "bev_bounds_xyxy": (-1.0, -1.0, 1.0, 1.0),
        "resolution_m": 1.0,
    }
    policy = FailureEventPolicyV1()
    clean_bundle = build_actual_target_branch(
        clean_decoded,
        gt,
        clean_iou,
        gt_support,
        clean_pred_support,
        patch_valid,
        support_provenance=support_provenance,
        target_provenance=provenance("clean"),
        chronology=chronology("clean"),
        observation=ObservationConditionV1(
            corruption_family="none",
            severity=0.0,
            seed=None,
            event_window_frames=None,
            active_at_frame=False,
        ),
        failure_event_policy=policy,
        bev_iou_policy_id="cpu-mock-pairwise-bev-iou/v1",
        minimum_prediction_score=0.5,
        bev_occupancy_sidecar=make_bev_occupancy_error_sidecar(
            gt_bev, clean_bev, **sidecar_kwargs
        ),
    )
    observed_bundle = build_actual_target_branch(
        observed_decoded,
        gt,
        observed_iou,
        gt_support,
        observed_pred_support,
        patch_valid,
        support_provenance=support_provenance,
        target_provenance=provenance("observed"),
        chronology=chronology("observed"),
        observation=ObservationConditionV1(
            corruption_family="local_blur",
            severity=2.0,
            seed=7,
            event_window_frames=(40, 50),
            active_at_frame=True,
        ),
        failure_event_policy=policy,
        bev_iou_policy_id="cpu-mock-pairwise-bev-iou/v1",
        minimum_prediction_score=0.5,
        bev_occupancy_sidecar=make_bev_occupancy_error_sidecar(
            gt_bev, observed_bev, **sidecar_kwargs
        ),
    )
    paired = pair_actual_target_branches(
        observed_bundle,
        clean_bundle,
        bundle_id="cpu-mock-route-001-frame-42-local-blur",
        real_orion_hook_executed=False,
    )
    generator = torch.Generator().manual_seed(7)
    clean_features = torch.randn(2, 6, feature_dim, generator=generator)
    observed_features = clean_features.clone()
    observed_features[0, 3:5] += 0.5
    corruption_mask = torch.zeros(2, 6)
    corruption_mask[0, 3:5] = 1.0
    return paired, observed_features, clean_features, corruption_mask


__all__ = [
    "BRANCH_BUNDLE_SCHEMA_VERSION",
    "CHRONOLOGY_SCHEMA_VERSION",
    "DECODED_INPUT_SCHEMA_VERSION",
    "FAILURE_EVENT_POLICY_VERSION",
    "GT_INPUT_SCHEMA_VERSION",
    "MATCH_POLICY_ID",
    "OBSERVATION_SCHEMA_VERSION",
    "PAIRED_BUNDLE_SCHEMA_VERSION",
    "ActualTargetBranchBundleV1",
    "ActualTargetExportError",
    "DecodedORIONFrameV1",
    "FailureEventPolicyV1",
    "FrameChronologyV1",
    "ObservationConditionV1",
    "PairedActualTargetBundleV1",
    "PrivilegedGroundTruthFrameV1",
    "bridge_actual_target_bundle_to_v2_record",
    "build_cpu_mock_actual_target_bundle",
    "build_actual_target_branch",
    "load_paired_actual_target_bundle",
    "pair_actual_target_branches",
    "save_paired_actual_target_bundle",
]
