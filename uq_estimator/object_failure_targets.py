"""Pure utilities for task-grounded Stage-1 object-failure targets.

This module deliberately does not import ORION, MMCV, CARLA, or a dataset.
It turns already-decoded predictions and privileged ground truth into bounded,
auditable error components and aggregates *projected visible object support*
onto a camera patch grid.  The projection support is an attribution proxy: it
does not establish that a pixel or patch caused the object failure.

The caller remains responsible for decoding ORION outputs, constructing
post-augmentation projected supports, and using the repository's audited BEV
box/occupancy geometry.  Ambiguous coordinate systems or provenance are
rejected instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Sequence

import torch


OBJECT_FAILURE_TARGET_VERSION = "orion_projected_object_failure/v1"
PATCH_SUPPORT_SCHEMA_VERSION = "orion.projected-visible-support/v1"
BEV_OCCUPANCY_SIDECAR_VERSION = "orion.bev-occupancy-error/v1"
ACTUAL_TARGET_PROVENANCE = "actual_frozen_orion_task_error"
PATCH_ATTRIBUTION_CLAIM = "projected_visible_object_support_proxy"
POST_AUGMENTATION_PROJECTION = "post_augmentation_lidar2img"
MOTION_MODE_POLICY = "orion_selected_mode"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ObjectFailureTargetError(ValueError):
    """Raised when target geometry, tensors, or provenance are ambiguous."""


def _require_text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ObjectFailureTargetError(f"{name} must be non-empty")
    return result


def _require_sha256(value: str, name: str) -> str:
    raw = str(value).strip()
    result = raw.lower()
    if not _SHA256_RE.fullmatch(result):
        raise ObjectFailureTargetError(
            f"{name} must be a lowercase 64-character SHA-256 hex digest"
        )
    if raw != result:
        raise ObjectFailureTargetError(f"{name} must use lowercase hex")
    return result


def _require_floating_tensor(
    value: torch.Tensor,
    name: str,
    *,
    ndim: Optional[int] = None,
    unit_interval: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ObjectFailureTargetError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ObjectFailureTargetError(f"{name} must have {ndim} dimensions")
    if not value.is_floating_point():
        raise ObjectFailureTargetError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ObjectFailureTargetError(f"{name} must contain only finite values")
    if unit_interval and (torch.any(value < 0) or torch.any(value > 1)):
        raise ObjectFailureTargetError(f"{name} must lie in [0, 1]")


def _require_integer_vector(value: torch.Tensor, name: str, length: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ObjectFailureTargetError(f"{name} must have shape [{length}]")
    if value.shape[0] != length:
        raise ObjectFailureTargetError(f"{name} must have shape [{length}]")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ObjectFailureTargetError(f"{name} must use an integer dtype")
    if torch.any(value < 0):
        raise ObjectFailureTargetError(f"{name} must contain non-negative IDs")


def _require_bool_tensor(
    value: torch.Tensor, name: str, shape: Sequence[int]
) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ObjectFailureTargetError(
            f"{name} must have shape {tuple(shape)}, got "
            f"{tuple(value.shape) if isinstance(value, torch.Tensor) else type(value)!r}"
        )
    if value.dtype != torch.bool:
        raise ObjectFailureTargetError(f"{name} must have boolean dtype")


@dataclass(frozen=True)
class PatchSupportProvenanceV1:
    """Geometry contract for precomputed projected object supports.

    ``image_transform_id`` must identify the exact resize/crop/flip/rotation
    applied to both RGB and calibration.  Raw-camera projections must not be
    paired with processed EVAViT patches.
    """

    camera_order: tuple[str, ...]
    image_hw: tuple[int, int]
    patch_hw: tuple[int, int]
    image_transform_id: str
    coordinate_frame: str = "lidar"
    projection_matrix_kind: str = POST_AUGMENTATION_PROJECTION
    attribution: str = PATCH_ATTRIBUTION_CLAIM
    schema_version: str = PATCH_SUPPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PATCH_SUPPORT_SCHEMA_VERSION:
            raise ObjectFailureTargetError(
                f"unsupported patch support schema {self.schema_version!r}"
            )
        if self.coordinate_frame != "lidar":
            raise ObjectFailureTargetError(
                "coordinate_frame must be explicitly 'lidar' for v1"
            )
        if self.projection_matrix_kind != POST_AUGMENTATION_PROJECTION:
            raise ObjectFailureTargetError(
                "v1 requires post-augmentation lidar2img projection"
            )
        if self.attribution != PATCH_ATTRIBUTION_CLAIM:
            raise ObjectFailureTargetError(
                "patch attribution must remain the projected-support proxy claim"
            )
        if not self.camera_order or any(not str(name).strip() for name in self.camera_order):
            raise ObjectFailureTargetError("camera_order must contain non-empty names")
        if len(set(self.camera_order)) != len(self.camera_order):
            raise ObjectFailureTargetError("camera_order must not contain duplicates")
        for name, shape in (("image_hw", self.image_hw), ("patch_hw", self.patch_hw)):
            if len(shape) != 2 or any(isinstance(x, bool) or int(x) <= 0 for x in shape):
                raise ObjectFailureTargetError(f"{name} must contain two positive integers")
            if any(int(x) != x for x in shape):
                raise ObjectFailureTargetError(f"{name} must contain integer values")
        _require_text(self.image_transform_id, "image_transform_id")

    @property
    def patch_count(self) -> int:
        return int(self.patch_hw[0]) * int(self.patch_hw[1])


@dataclass(frozen=True)
class TargetProvenanceV1:
    """Frozen-model and sample identity required by an actual target."""

    base_checkpoint_sha256: str
    inference_config_sha256: str
    git_revision: str
    route_id: str
    town: str
    frame_idx: int
    observation_branch: str
    # Branch-specific identity/hash of the actual temporal memory content.
    temporal_history_id: str
    # Shared replay/reset/schedule-prefix protocol used to establish pairing.
    paired_history_protocol_id: str
    class_mapping_id: str
    decoder_policy_id: str
    camera_order: tuple[str, ...]
    image_transform_id: str
    target_version: str = OBJECT_FAILURE_TARGET_VERSION
    target_provenance: str = ACTUAL_TARGET_PROVENANCE

    def __post_init__(self) -> None:
        if self.target_version != OBJECT_FAILURE_TARGET_VERSION:
            raise ObjectFailureTargetError(
                f"unsupported target version {self.target_version!r}"
            )
        if self.target_provenance != ACTUAL_TARGET_PROVENANCE:
            raise ObjectFailureTargetError(
                "target_provenance must identify actual frozen-ORION task error"
            )
        _require_sha256(self.base_checkpoint_sha256, "base_checkpoint_sha256")
        _require_sha256(self.inference_config_sha256, "inference_config_sha256")
        for name in (
            "git_revision",
            "route_id",
            "town",
            "temporal_history_id",
            "paired_history_protocol_id",
            "class_mapping_id",
            "decoder_policy_id",
            "image_transform_id",
        ):
            _require_text(getattr(self, name), name)
        if isinstance(self.frame_idx, bool) or not isinstance(self.frame_idx, int):
            raise ObjectFailureTargetError("frame_idx must be an integer")
        if self.frame_idx < 0:
            raise ObjectFailureTargetError("frame_idx must be non-negative")
        if self.observation_branch not in ("observed", "clean"):
            raise ObjectFailureTargetError(
                "observation_branch must be exactly 'observed' or 'clean'"
            )
        if not self.camera_order or any(not str(name).strip() for name in self.camera_order):
            raise ObjectFailureTargetError("camera_order must contain non-empty names")
        if len(set(self.camera_order)) != len(self.camera_order):
            raise ObjectFailureTargetError("camera_order must not contain duplicates")


@dataclass(frozen=True)
class ClassAwareMatch:
    """One-to-one, class-aware matches after score and distance gating."""

    gt_to_pred: torch.Tensor
    pred_to_gt: torch.Tensor
    matched_center_distance: torch.Tensor
    valid_pair_mask: torch.Tensor
    coordinate_frame: str
    max_center_distance: torch.Tensor
    minimum_prediction_score: float

    def __post_init__(self) -> None:
        if self.coordinate_frame != "lidar":
            raise ObjectFailureTargetError("match coordinate_frame must be 'lidar'")
        if not isinstance(self.gt_to_pred, torch.Tensor) or self.gt_to_pred.ndim != 1:
            raise ObjectFailureTargetError("gt_to_pred must be a one-dimensional tensor")
        if self.gt_to_pred.dtype != torch.long:
            raise ObjectFailureTargetError("gt_to_pred must use torch.long")
        if not isinstance(self.pred_to_gt, torch.Tensor) or self.pred_to_gt.ndim != 1:
            raise ObjectFailureTargetError("pred_to_gt must be a one-dimensional tensor")
        if self.pred_to_gt.dtype != torch.long:
            raise ObjectFailureTargetError("pred_to_gt must use torch.long")
        gt_count, pred_count = self.gt_to_pred.shape[0], self.pred_to_gt.shape[0]
        _require_bool_tensor(
            self.valid_pair_mask, "valid_pair_mask", (gt_count, pred_count)
        )
        _require_floating_tensor(
            self.max_center_distance, "max_center_distance", ndim=1
        )
        if self.max_center_distance.shape != (gt_count,) or torch.any(
            self.max_center_distance <= 0
        ):
            raise ObjectFailureTargetError(
                "max_center_distance must be a positive vector with one value per GT"
            )
        if not isinstance(self.matched_center_distance, torch.Tensor) or self.matched_center_distance.shape != (gt_count,):
            raise ObjectFailureTargetError(
                "matched_center_distance must have one value per GT"
            )
        if not self.matched_center_distance.is_floating_point():
            raise ObjectFailureTargetError("matched_center_distance must be floating point")
        if not 0.0 <= float(self.minimum_prediction_score) <= 1.0:
            raise ObjectFailureTargetError(
                "minimum_prediction_score must lie in [0, 1]"
            )
        tensors = (
            self.pred_to_gt,
            self.matched_center_distance,
            self.valid_pair_mask,
            self.max_center_distance,
        )
        if any(tensor.device != self.gt_to_pred.device for tensor in tensors):
            raise ObjectFailureTargetError("all match tensors must share a device")
        matched = self.gt_to_pred >= 0
        if torch.any(self.gt_to_pred >= pred_count) or torch.any(self.pred_to_gt >= gt_count):
            raise ObjectFailureTargetError("match indices are out of range")
        if bool(matched.any()):
            gt_indices = torch.nonzero(matched, as_tuple=False).flatten()
            pred_indices = self.gt_to_pred[gt_indices]
            if not bool(self.valid_pair_mask[gt_indices, pred_indices].all()):
                raise ObjectFailureTargetError("a match uses an invalid gated pair")
            if not bool(torch.isfinite(self.matched_center_distance[gt_indices]).all()):
                raise ObjectFailureTargetError("matched distances must be finite")
            if torch.any(
                self.matched_center_distance[gt_indices]
                > self.max_center_distance[gt_indices]
            ):
                raise ObjectFailureTargetError("a matched distance exceeds its gate")
            if not torch.equal(self.pred_to_gt[pred_indices], gt_indices):
                raise ObjectFailureTargetError("GT/prediction match maps are not reciprocal")
            if torch.unique(pred_indices).numel() != pred_indices.numel():
                raise ObjectFailureTargetError("prediction matches must be one-to-one")
        if bool((~matched).any()) and not bool(
            torch.isposinf(self.matched_center_distance[~matched]).all()
        ):
            raise ObjectFailureTargetError("unmatched GT distances must be +inf")
        pred_matched = self.pred_to_gt >= 0
        if bool(pred_matched.any()):
            pred_indices = torch.nonzero(pred_matched, as_tuple=False).flatten()
            gt_indices = self.pred_to_gt[pred_indices]
            if not torch.equal(self.gt_to_pred[gt_indices], pred_indices):
                raise ObjectFailureTargetError("prediction/GT match maps are not reciprocal")

    @property
    def matched_gt(self) -> torch.Tensor:
        return self.gt_to_pred >= 0

    @property
    def unmatched_prediction(self) -> torch.Tensor:
        return self.pred_to_gt < 0


@dataclass(frozen=True)
class ObjectFailureComponents:
    """Auditable bounded component errors for every GT object."""

    component_names: tuple[str, ...]
    values: torch.Tensor
    valid: torch.Tensor
    soft_union: torch.Tensor
    soft_union_valid: torch.Tensor
    false_positive_error: torch.Tensor
    false_positive_valid: torch.Tensor
    match: ClassAwareMatch
    motion_mode_policy: Optional[str]


@dataclass(frozen=True)
class ProjectedObjectFailureTargetV1:
    """Actual object failure aggregated on camera patches by noisy-OR."""

    error: torch.Tensor
    valid_mask: torch.Tensor
    support: torch.Tensor
    object_error: torch.Tensor
    object_valid: torch.Tensor
    support_provenance: PatchSupportProvenanceV1
    target_provenance: TargetProvenanceV1
    attribution_is_causal: bool = False


@dataclass(frozen=True)
class PairedFailureTargetsV1:
    """Separate observed, clean, and corruption-attributable target maps."""

    observed_error: torch.Tensor
    clean_error: torch.Tensor
    delta_error: torch.Tensor
    valid_mask: torch.Tensor
    observed_valid_mask: torch.Tensor
    clean_valid_mask: torch.Tensor
    target_version: str = OBJECT_FAILURE_TARGET_VERSION


@dataclass(frozen=True)
class BEVOccupancyErrorSidecarV1:
    """Optional BEV audit sidecar; never a flattened camera-patch label."""

    gt_occupancy: torch.Tensor
    predicted_occupancy: torch.Tensor
    absolute_error: torch.Tensor
    valid_mask: torch.Tensor
    gt_provenance: str
    prediction_provenance: str
    coordinate_frame: str
    bev_bounds_xyxy: tuple[float, float, float, float]
    resolution_m: float
    schema_version: str = BEV_OCCUPANCY_SIDECAR_VERSION


def class_aware_distance_gated_match(
    gt_centers: torch.Tensor,
    gt_classes: torch.Tensor,
    pred_centers: torch.Tensor,
    pred_classes: torch.Tensor,
    pred_scores: torch.Tensor,
    *,
    coordinate_frame: str,
    max_center_distance: float | torch.Tensor = 4.0,
    minimum_prediction_score: float = 0.0,
) -> ClassAwareMatch:
    """Match decoded objects only across valid class/score/distance edges.

    The linear assignment is performed *after* invalid edges are excluded and
    every result is gated again.  Consequently, a Hungarian assignment to a
    wrong-class, low-score, or distant query can never turn a GT miss into a
    success.  The padded cost prioritizes maximum valid cardinality, then
    minimum total center distance.
    """

    if coordinate_frame != "lidar":
        raise ObjectFailureTargetError(
            "coordinate_frame must be explicitly 'lidar' for v1 matching"
        )
    _require_floating_tensor(gt_centers, "gt_centers", ndim=2)
    _require_floating_tensor(pred_centers, "pred_centers", ndim=2)
    if gt_centers.shape[1] not in (2, 3) or pred_centers.shape[1] != gt_centers.shape[1]:
        raise ObjectFailureTargetError(
            "gt_centers and pred_centers must share shape [objects, 2 or 3]"
        )
    if gt_centers.device != pred_centers.device:
        raise ObjectFailureTargetError("GT and prediction tensors must share a device")
    gt_count, pred_count = gt_centers.shape[0], pred_centers.shape[0]
    _require_integer_vector(gt_classes, "gt_classes", gt_count)
    _require_integer_vector(pred_classes, "pred_classes", pred_count)
    _require_floating_tensor(pred_scores, "pred_scores", ndim=1, unit_interval=True)
    if pred_scores.shape != (pred_count,):
        raise ObjectFailureTargetError(f"pred_scores must have shape [{pred_count}]")
    for tensor, name in (
        (gt_classes, "gt_classes"),
        (pred_classes, "pred_classes"),
        (pred_scores, "pred_scores"),
    ):
        if tensor.device != gt_centers.device:
            raise ObjectFailureTargetError(f"{name} must share the centers' device")
    if not 0.0 <= float(minimum_prediction_score) <= 1.0:
        raise ObjectFailureTargetError("minimum_prediction_score must lie in [0, 1]")

    if isinstance(max_center_distance, torch.Tensor):
        _require_floating_tensor(max_center_distance, "max_center_distance")
        if max_center_distance.ndim == 0:
            distance_gate = max_center_distance.expand(gt_count)
        elif max_center_distance.shape == (gt_count,):
            distance_gate = max_center_distance
        else:
            raise ObjectFailureTargetError(
                f"max_center_distance must be scalar or shape [{gt_count}]"
            )
        distance_gate = distance_gate.to(device=gt_centers.device, dtype=gt_centers.dtype)
    else:
        distance_gate = torch.full(
            (gt_count,),
            float(max_center_distance),
            device=gt_centers.device,
            dtype=gt_centers.dtype,
        )
    if torch.any(distance_gate <= 0):
        raise ObjectFailureTargetError("max_center_distance must be positive")

    distance = torch.cdist(gt_centers[:, :2], pred_centers[:, :2])
    valid_pairs = (
        (gt_classes[:, None] == pred_classes[None, :])
        & (distance <= distance_gate[:, None])
        & (pred_scores[None, :] >= float(minimum_prediction_score))
    )
    gt_to_pred = torch.full(
        (gt_count,), -1, device=gt_centers.device, dtype=torch.long
    )
    pred_to_gt = torch.full(
        (pred_count,), -1, device=gt_centers.device, dtype=torch.long
    )
    matched_distance = torch.full(
        (gt_count,), float("inf"), device=gt_centers.device, dtype=gt_centers.dtype
    )

    if gt_count and pred_count and bool(valid_pairs.any()):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:  # pragma: no cover - normal env includes scipy
            raise RuntimeError(
                "scipy is required for exact gated one-to-one matching"
            ) from error

        # Each GT gets a private dummy column.  The dummy penalty is larger
        # than any possible sum of valid distances, so the assignment first
        # maximizes the number of valid matches and only then minimizes range.
        max_gate = float(distance_gate.max().item())
        unmatched_cost = (min(gt_count, pred_count) + 1.0) * (max_gate + 1.0)
        cost = torch.full(
            (gt_count, pred_count + gt_count),
            2.0 * unmatched_cost,
            dtype=torch.float64,
        )
        cost[:, :pred_count] = torch.where(
            valid_pairs.detach().cpu(),
            distance.detach().cpu().to(torch.float64),
            torch.full((gt_count, pred_count), 2.0 * unmatched_cost, dtype=torch.float64),
        )
        for gt_index in range(gt_count):
            cost[gt_index, pred_count + gt_index] = unmatched_cost
        rows, columns = linear_sum_assignment(cost.numpy())
        for gt_index, pred_index in zip(rows.tolist(), columns.tolist()):
            if pred_index >= pred_count:
                continue
            # Fail closed even if the assignment implementation or padding is
            # changed later: an invalid edge is never accepted post hoc.
            if not bool(valid_pairs[gt_index, pred_index]):
                continue
            gt_to_pred[gt_index] = pred_index
            pred_to_gt[pred_index] = gt_index
            matched_distance[gt_index] = distance[gt_index, pred_index]

    return ClassAwareMatch(
        gt_to_pred=gt_to_pred,
        pred_to_gt=pred_to_gt,
        matched_center_distance=matched_distance,
        valid_pair_mask=valid_pairs,
        coordinate_frame=coordinate_frame,
        max_center_distance=distance_gate,
        minimum_prediction_score=float(minimum_prediction_score),
    )


def bounded_soft_union(
    component_values: torch.Tensor,
    component_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine valid bounded error channels as ``1 - product(1 - e_k)``."""

    _require_floating_tensor(
        component_values, "component_values", unit_interval=True
    )
    _require_bool_tensor(component_valid, "component_valid", component_values.shape)
    if component_values.ndim < 1 or component_values.shape[-1] == 0:
        raise ObjectFailureTargetError(
            "component_values must have a non-empty component dimension"
        )
    masked = torch.where(component_valid, component_values, torch.zeros_like(component_values))
    union = 1.0 - torch.prod(1.0 - masked, dim=-1)
    return union.clamp(0.0, 1.0), component_valid.any(dim=-1)


def _soft_iou_error(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    valid_time: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean temporal soft-IoU error for one matched object."""

    spatial_dims = tuple(range(1, ground_truth.ndim))
    intersection = (ground_truth * prediction).sum(dim=spatial_dims)
    union = (ground_truth + prediction - ground_truth * prediction).sum(dim=spatial_dims)
    iou = torch.where(union > eps, intersection / union.clamp_min(eps), torch.ones_like(union))
    return (1.0 - iou[valid_time]).mean().clamp(0.0, 1.0)


def compute_object_failure_components(
    match: ClassAwareMatch,
    gt_classes: torch.Tensor,
    pred_class_probabilities: torch.Tensor,
    pred_scores: torch.Tensor,
    *,
    pairwise_bev_iou: Optional[torch.Tensor] = None,
    gt_motion_occupancy: Optional[torch.Tensor] = None,
    pred_motion_occupancy: Optional[torch.Tensor] = None,
    motion_valid_mask: Optional[torch.Tensor] = None,
    motion_mode_policy: Optional[str] = None,
    gt_traffic_states: Optional[torch.Tensor] = None,
    pred_traffic_state_probabilities: Optional[torch.Tensor] = None,
    traffic_state_valid: Optional[torch.Tensor] = None,
) -> ObjectFailureComponents:
    """Compute miss/class/localization/motion/state errors separately.

    Motion occupancy must already use ORION's selected mode.  Oracle-best-mode
    occupancy is intentionally rejected as a primary target.
    """

    gt_count = match.gt_to_pred.shape[0]
    pred_count = match.pred_to_gt.shape[0]
    _require_integer_vector(gt_classes, "gt_classes", gt_count)
    _require_floating_tensor(
        pred_class_probabilities,
        "pred_class_probabilities",
        ndim=2,
        unit_interval=True,
    )
    if pred_class_probabilities.shape[0] != pred_count:
        raise ObjectFailureTargetError(
            "pred_class_probabilities first dimension must equal prediction count"
        )
    _require_floating_tensor(pred_scores, "pred_scores", ndim=1, unit_interval=True)
    if pred_scores.shape != (pred_count,):
        raise ObjectFailureTargetError(f"pred_scores must have shape [{pred_count}]")
    if gt_classes.device != match.gt_to_pred.device:
        raise ObjectFailureTargetError("gt_classes and match must share a device")
    if pred_class_probabilities.device != match.gt_to_pred.device or pred_scores.device != match.gt_to_pred.device:
        raise ObjectFailureTargetError("prediction tensors and match must share a device")
    if gt_count and int(gt_classes.max().item()) >= pred_class_probabilities.shape[1]:
        raise ObjectFailureTargetError(
            "gt_classes contains an ID outside pred_class_probabilities"
        )

    names = ("miss", "class", "localization", "motion_occupancy", "traffic_state")
    values = pred_class_probabilities.new_zeros((gt_count, len(names)))
    valid = torch.zeros((gt_count, len(names)), dtype=torch.bool, device=values.device)
    matched = match.matched_gt
    values[:, 0] = (~matched).to(values.dtype)
    valid[:, 0] = True

    if bool(matched.any()):
        gt_indices = torch.nonzero(matched, as_tuple=False).flatten()
        pred_indices = match.gt_to_pred[gt_indices]
        values[gt_indices, 1] = 1.0 - pred_class_probabilities[
            pred_indices, gt_classes[gt_indices].long()
        ]
        valid[gt_indices, 1] = True

    if pairwise_bev_iou is not None:
        _require_floating_tensor(
            pairwise_bev_iou, "pairwise_bev_iou", ndim=2, unit_interval=True
        )
        if pairwise_bev_iou.shape != (gt_count, pred_count):
            raise ObjectFailureTargetError(
                f"pairwise_bev_iou must have shape [{gt_count}, {pred_count}]"
            )
        if pairwise_bev_iou.device != values.device:
            raise ObjectFailureTargetError("pairwise_bev_iou and match must share a device")
        if bool(matched.any()):
            gt_indices = torch.nonzero(matched, as_tuple=False).flatten()
            pred_indices = match.gt_to_pred[gt_indices]
            normalized_distance = (
                match.matched_center_distance[gt_indices]
                / match.max_center_distance[gt_indices]
            ).clamp(0.0, 1.0)
            iou_error = 1.0 - pairwise_bev_iou[gt_indices, pred_indices]
            values[gt_indices, 2] = torch.maximum(normalized_distance, iou_error)
            valid[gt_indices, 2] = True

    motion_inputs = (
        gt_motion_occupancy,
        pred_motion_occupancy,
        motion_valid_mask,
        motion_mode_policy,
    )
    if any(value is not None for value in motion_inputs):
        if any(value is None for value in motion_inputs):
            raise ObjectFailureTargetError(
                "all motion occupancy tensors, valid mask, and mode policy are required together"
            )
        if motion_mode_policy != MOTION_MODE_POLICY:
            raise ObjectFailureTargetError(
                "primary motion occupancy must use ORION's selected mode"
            )
        assert gt_motion_occupancy is not None
        assert pred_motion_occupancy is not None
        assert motion_valid_mask is not None
        _require_floating_tensor(
            gt_motion_occupancy, "gt_motion_occupancy", unit_interval=True
        )
        _require_floating_tensor(
            pred_motion_occupancy, "pred_motion_occupancy", unit_interval=True
        )
        if gt_motion_occupancy.ndim < 3:
            raise ObjectFailureTargetError(
                "motion occupancy must have shape [objects, time, spatial...]"
            )
        if gt_motion_occupancy.shape[0] != gt_count or pred_motion_occupancy.shape[0] != pred_count:
            raise ObjectFailureTargetError(
                "motion occupancy object counts must match GT/prediction counts"
            )
        if gt_motion_occupancy.shape[1:] != pred_motion_occupancy.shape[1:]:
            raise ObjectFailureTargetError(
                "GT and predicted motion occupancy must share time/spatial shape"
            )
        _require_bool_tensor(
            motion_valid_mask,
            "motion_valid_mask",
            (gt_count, gt_motion_occupancy.shape[1]),
        )
        if gt_motion_occupancy.device != values.device or pred_motion_occupancy.device != values.device or motion_valid_mask.device != values.device:
            raise ObjectFailureTargetError("motion occupancy tensors and match must share a device")
        for gt_index in range(gt_count):
            temporal_valid = motion_valid_mask[gt_index]
            if not bool(temporal_valid.any()):
                continue
            valid[gt_index, 3] = True
            pred_index = int(match.gt_to_pred[gt_index].item())
            if pred_index < 0:
                values[gt_index, 3] = 1.0
            else:
                values[gt_index, 3] = _soft_iou_error(
                    gt_motion_occupancy[gt_index],
                    pred_motion_occupancy[pred_index],
                    temporal_valid,
                )

    traffic_inputs = (
        gt_traffic_states,
        pred_traffic_state_probabilities,
        traffic_state_valid,
    )
    if any(value is not None for value in traffic_inputs):
        if any(value is None for value in traffic_inputs):
            raise ObjectFailureTargetError(
                "traffic state labels, probabilities, and valid mask are required together"
            )
        assert gt_traffic_states is not None
        assert pred_traffic_state_probabilities is not None
        assert traffic_state_valid is not None
        if (
            not isinstance(gt_traffic_states, torch.Tensor)
            or gt_traffic_states.ndim != 1
            or gt_traffic_states.shape[0] != gt_count
            or gt_traffic_states.dtype == torch.bool
            or gt_traffic_states.is_floating_point()
            or gt_traffic_states.is_complex()
        ):
            raise ObjectFailureTargetError(
                f"gt_traffic_states must be an integer tensor with shape [{gt_count}]"
            )
        _require_floating_tensor(
            pred_traffic_state_probabilities,
            "pred_traffic_state_probabilities",
            ndim=2,
            unit_interval=True,
        )
        if pred_traffic_state_probabilities.shape[0] != pred_count:
            raise ObjectFailureTargetError(
                "traffic-state probability count must match predictions"
            )
        _require_bool_tensor(
            traffic_state_valid, "traffic_state_valid", (gt_count,)
        )
        if gt_count and bool(traffic_state_valid.any()):
            if torch.any(gt_traffic_states[traffic_state_valid] < 0):
                raise ObjectFailureTargetError(
                    "valid GT traffic states must be non-negative"
                )
            max_state = int(gt_traffic_states[traffic_state_valid].max().item())
            if max_state >= pred_traffic_state_probabilities.shape[1]:
                raise ObjectFailureTargetError(
                    "valid GT traffic state exceeds the probability dimension"
                )
        if gt_traffic_states.device != values.device or pred_traffic_state_probabilities.device != values.device or traffic_state_valid.device != values.device:
            raise ObjectFailureTargetError("traffic-state tensors and match must share a device")
        for gt_index in torch.nonzero(traffic_state_valid, as_tuple=False).flatten().tolist():
            valid[gt_index, 4] = True
            pred_index = int(match.gt_to_pred[gt_index].item())
            if pred_index < 0:
                values[gt_index, 4] = 1.0
            else:
                state = int(gt_traffic_states[gt_index].item())
                values[gt_index, 4] = 1.0 - pred_traffic_state_probabilities[
                    pred_index, state
                ]

    union, union_valid = bounded_soft_union(values, valid)
    fp_valid = match.unmatched_prediction & (
        pred_scores >= match.minimum_prediction_score
    )
    fp_error = torch.where(fp_valid, pred_scores, torch.zeros_like(pred_scores))
    return ObjectFailureComponents(
        component_names=names,
        values=values.clamp(0.0, 1.0),
        valid=valid,
        soft_union=union,
        soft_union_valid=union_valid,
        false_positive_error=fp_error,
        false_positive_valid=fp_valid,
        match=match,
        motion_mode_policy=motion_mode_policy,
    )


def aggregate_projected_visible_support(
    support: torch.Tensor,
    object_error: torch.Tensor,
    object_valid: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    *,
    support_provenance: PatchSupportProvenanceV1,
    target_provenance: TargetProvenanceV1,
) -> ProjectedObjectFailureTargetV1:
    """Noisy-OR aggregate ``[V,P,J]`` visible supports into ``[V,P]``.

    Invalid patches are returned as numeric zero only as a storage placeholder;
    callers must use ``valid_mask`` in every loss and metric.
    """

    _require_floating_tensor(support, "support", ndim=3, unit_interval=True)
    _require_floating_tensor(object_error, "object_error", ndim=1, unit_interval=True)
    views, patches, objects = support.shape
    if object_error.shape != (objects,):
        raise ObjectFailureTargetError(f"object_error must have shape [{objects}]")
    _require_bool_tensor(object_valid, "object_valid", (objects,))
    _require_bool_tensor(patch_valid_mask, "patch_valid_mask", (views, patches))
    if support_provenance.camera_order != target_provenance.camera_order:
        raise ObjectFailureTargetError(
            "support and target provenance camera_order values disagree"
        )
    if support_provenance.image_transform_id != target_provenance.image_transform_id:
        raise ObjectFailureTargetError(
            "support and target provenance image_transform_id values disagree"
        )
    if len(support_provenance.camera_order) != views:
        raise ObjectFailureTargetError(
            "support view count does not match provenance camera_order"
        )
    if support_provenance.patch_count != patches:
        raise ObjectFailureTargetError(
            "support patch count does not match provenance patch_hw"
        )
    for tensor, name in (
        (object_error, "object_error"),
        (object_valid, "object_valid"),
        (patch_valid_mask, "patch_valid_mask"),
    ):
        if tensor.device != support.device:
            raise ObjectFailureTargetError(f"{name} and support must share a device")

    effective_error = torch.where(
        object_valid, object_error, torch.zeros_like(object_error)
    )
    factors = 1.0 - support * effective_error.view(1, 1, objects)
    error = 1.0 - factors.prod(dim=-1)
    error = torch.where(patch_valid_mask, error, torch.zeros_like(error))
    return ProjectedObjectFailureTargetV1(
        error=error.clamp(0.0, 1.0),
        valid_mask=patch_valid_mask,
        support=support,
        object_error=object_error,
        object_valid=object_valid,
        support_provenance=support_provenance,
        target_provenance=target_provenance,
        attribution_is_causal=False,
    )


def make_paired_failure_targets(
    observed: ProjectedObjectFailureTargetV1,
    clean: ProjectedObjectFailureTargetV1,
) -> PairedFailureTargetsV1:
    """Keep ``E_obs``, ``E_clean``, and ``relu(E_obs-E_clean)`` distinct."""

    if observed.target_provenance.observation_branch != "observed":
        raise ObjectFailureTargetError("observed target must declare branch='observed'")
    if clean.target_provenance.observation_branch != "clean":
        raise ObjectFailureTargetError("clean target must declare branch='clean'")
    if observed.error.shape != clean.error.shape:
        raise ObjectFailureTargetError("observed and clean target shapes must match")
    observed_identity = observed.target_provenance
    clean_identity = clean.target_provenance
    comparable_fields = (
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
        for name in comparable_fields
        if getattr(observed_identity, name) != getattr(clean_identity, name)
    ]
    if disagreements:
        raise ObjectFailureTargetError(
            "paired target provenance differs in: " + ", ".join(disagreements)
        )
    valid = observed.valid_mask & clean.valid_mask
    observed_error = torch.where(
        observed.valid_mask, observed.error, torch.zeros_like(observed.error)
    )
    clean_error = torch.where(
        clean.valid_mask, clean.error, torch.zeros_like(clean.error)
    )
    delta = torch.where(
        valid,
        torch.relu(observed.error - clean.error),
        torch.zeros_like(observed.error),
    )
    return PairedFailureTargetsV1(
        observed_error=observed_error,
        clean_error=clean_error,
        delta_error=delta,
        valid_mask=valid,
        observed_valid_mask=observed.valid_mask,
        clean_valid_mask=clean.valid_mask,
    )


def make_bev_occupancy_error_sidecar(
    gt_occupancy: torch.Tensor,
    predicted_occupancy: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gt_provenance: str,
    prediction_provenance: str,
    coordinate_frame: str,
    bev_bounds_xyxy: tuple[float, float, float, float],
    resolution_m: float,
) -> BEVOccupancyErrorSidecarV1:
    """Create an explicit BEV occupancy audit sidecar from matched grids."""

    _require_floating_tensor(gt_occupancy, "gt_occupancy", unit_interval=True)
    _require_floating_tensor(
        predicted_occupancy, "predicted_occupancy", unit_interval=True
    )
    if gt_occupancy.ndim not in (3, 4):
        raise ObjectFailureTargetError(
            "occupancy must have shape [T,H,W] or [T,H,W,C]"
        )
    if predicted_occupancy.shape != gt_occupancy.shape:
        raise ObjectFailureTargetError("GT and predicted occupancy shapes must match")
    _require_bool_tensor(valid_mask, "valid_mask", gt_occupancy.shape)
    if gt_occupancy.device != predicted_occupancy.device or gt_occupancy.device != valid_mask.device:
        raise ObjectFailureTargetError("occupancy tensors and valid_mask must share a device")
    if coordinate_frame != "ego_bev":
        raise ObjectFailureTargetError(
            "BEV sidecar coordinate_frame must be explicitly 'ego_bev'"
        )
    if len(bev_bounds_xyxy) != 4 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in bev_bounds_xyxy
    ):
        raise ObjectFailureTargetError("bev_bounds_xyxy must contain four numbers")
    x_min, y_min, x_max, y_max = (float(value) for value in bev_bounds_xyxy)
    if not all(torch.isfinite(torch.tensor([x_min, y_min, x_max, y_max]))):
        raise ObjectFailureTargetError("bev_bounds_xyxy must be finite")
    if not x_min < x_max or not y_min < y_max:
        raise ObjectFailureTargetError("BEV bounds must have positive area")
    if (
        isinstance(resolution_m, bool)
        or not bool(torch.isfinite(torch.tensor(float(resolution_m))))
        or float(resolution_m) <= 0
    ):
        raise ObjectFailureTargetError("resolution_m must be finite and positive")
    gt_source = _require_text(gt_provenance, "gt_provenance")
    pred_source = _require_text(prediction_provenance, "prediction_provenance")
    absolute_error = torch.where(
        valid_mask,
        torch.abs(predicted_occupancy - gt_occupancy),
        torch.zeros_like(gt_occupancy),
    )
    return BEVOccupancyErrorSidecarV1(
        gt_occupancy=gt_occupancy,
        predicted_occupancy=predicted_occupancy,
        absolute_error=absolute_error,
        valid_mask=valid_mask,
        gt_provenance=gt_source,
        prediction_provenance=pred_source,
        coordinate_frame=coordinate_frame,
        bev_bounds_xyxy=(x_min, y_min, x_max, y_max),
        resolution_m=float(resolution_m),
    )


__all__ = [
    "ACTUAL_TARGET_PROVENANCE",
    "BEV_OCCUPANCY_SIDECAR_VERSION",
    "MOTION_MODE_POLICY",
    "OBJECT_FAILURE_TARGET_VERSION",
    "PATCH_ATTRIBUTION_CLAIM",
    "PATCH_SUPPORT_SCHEMA_VERSION",
    "POST_AUGMENTATION_PROJECTION",
    "BEVOccupancyErrorSidecarV1",
    "ClassAwareMatch",
    "ObjectFailureComponents",
    "ObjectFailureTargetError",
    "PairedFailureTargetsV1",
    "PatchSupportProvenanceV1",
    "ProjectedObjectFailureTargetV1",
    "TargetProvenanceV1",
    "aggregate_projected_visible_support",
    "bounded_soft_union",
    "class_aware_distance_gated_match",
    "compute_object_failure_components",
    "make_bev_occupancy_error_sidecar",
    "make_paired_failure_targets",
]
