"""Fail-closed bridge from projected object errors to Stage-1 record v2.

This module freezes the first actual-target failure-event semantics without
importing ORION, MMCV, CARLA, or any GPU dependency. Continuous error severity
and a discrete failure event are deliberately constructed by different rules:

* severity is the existing noisy-OR of per-object soft-union task errors;
* event first thresholds object-level components, then projects the hard
  failed-object event through a separate minimum-visible-support gate.

The event policy therefore never reuses the severity map as a probability or
silently learns a threshold from held-out data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Optional

import torch

from uq_estimator.object_failure_targets import (
    ACTUAL_TARGET_PROVENANCE,
    MOTION_MODE_POLICY,
    OBJECT_FAILURE_TARGET_VERSION,
    ObjectFailureComponents,
    ProjectedObjectFailureTargetV1,
    bounded_soft_union,
)
from uq_estimator.spatial_training import (
    PAIRED_RECORD_SCHEMA_VERSION,
    TARGET_ACTUAL_FAILURE,
    TARGET_CONTRACT_SCHEMA_VERSION,
    PairedSpatialFeatureRecord,
)


FAILURE_EVENT_POLICY_SCHEMA_VERSION = (
    "orion.projected-object-failure-event-policy/v1"
)
ACTUAL_TARGET_BRIDGE_SCHEMA_VERSION = "orion.actual-target-record-bridge/v1"
COMPONENT_PATCH_AGGREGATION = "noisy_or_projected_visible_support_per_component"
EVENT_PATCH_PROJECTION = "any_failed_object_with_minimum_visible_patch_support"
EVENT_AGGREGATION = "threshold_object_components_then_project_support_then_any"
CANONICAL_COMPONENT_NAMES = (
    "miss",
    "class",
    "localization",
    "motion_occupancy",
    "traffic_state",
)
# These object-component thresholds are frozen for policy v1. Any threshold change
# requires a new policy version rather than mutating provenance under the same ID.
FROZEN_COMPONENT_THRESHOLDS_V1 = (0.50, 0.50, 0.50, 0.50, 0.50)
FROZEN_MIN_PATCH_SUPPORT_V1 = 0.01


class ActualTargetBridgeError(ValueError):
    """Raised when actual-target provenance or tensor semantics are unsafe."""


@dataclass(frozen=True)
class FailureEventPolicyV1:
    """Immutable component-level definition of a binary patch failure event."""

    component_names: tuple[str, ...] = CANONICAL_COMPONENT_NAMES
    component_thresholds: tuple[float, ...] = FROZEN_COMPONENT_THRESHOLDS_V1
    comparison: str = "greater_than_or_equal"
    continuous_component_patch_aggregation: str = COMPONENT_PATCH_AGGREGATION
    event_patch_projection: str = EVENT_PATCH_PROJECTION
    minimum_patch_support: float = FROZEN_MIN_PATCH_SUPPORT_V1
    support_comparison: str = "greater_than_or_equal"
    event_aggregation: str = EVENT_AGGREGATION
    output_semantics: str = "binary_patch_failure_event_not_probability"
    invalid_patch_policy: str = "masked_not_negative"
    motion_mode_policy: str = MOTION_MODE_POLICY
    missing_traffic_state_policy: str = "component_invalid_not_negative"
    schema_version: str = FAILURE_EVENT_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FAILURE_EVENT_POLICY_SCHEMA_VERSION:
            raise ActualTargetBridgeError(
                f"unsupported failure-event policy {self.schema_version!r}"
            )
        if self.component_names != CANONICAL_COMPONENT_NAMES:
            raise ActualTargetBridgeError(
                "policy v1 component_names are frozen; create a new policy version"
            )
        if self.component_thresholds != FROZEN_COMPONENT_THRESHOLDS_V1:
            raise ActualTargetBridgeError(
                "policy v1 component thresholds are frozen; create a new policy version"
            )
        if self.comparison != "greater_than_or_equal":
            raise ActualTargetBridgeError("policy v1 comparison is frozen")
        if self.continuous_component_patch_aggregation != COMPONENT_PATCH_AGGREGATION:
            raise ActualTargetBridgeError(
                "policy v1 continuous component patch aggregation is frozen"
            )
        if self.event_patch_projection != EVENT_PATCH_PROJECTION:
            raise ActualTargetBridgeError("policy v1 event patch projection is frozen")
        if self.minimum_patch_support != FROZEN_MIN_PATCH_SUPPORT_V1:
            raise ActualTargetBridgeError(
                "policy v1 minimum patch support is frozen; create a new policy version"
            )
        if self.support_comparison != "greater_than_or_equal":
            raise ActualTargetBridgeError("policy v1 support comparison is frozen")
        if self.event_aggregation != EVENT_AGGREGATION:
            raise ActualTargetBridgeError("policy v1 event aggregation is frozen")
        if self.output_semantics != "binary_patch_failure_event_not_probability":
            raise ActualTargetBridgeError("policy v1 output semantics are frozen")
        if self.invalid_patch_policy != "masked_not_negative":
            raise ActualTargetBridgeError("policy v1 invalid-patch semantics are frozen")
        if self.motion_mode_policy != MOTION_MODE_POLICY:
            raise ActualTargetBridgeError(
                "policy v1 motion target must use ORION's selected mode"
            )
        if self.missing_traffic_state_policy != "component_invalid_not_negative":
            raise ActualTargetBridgeError(
                "policy v1 missing traffic-state GT must remain invalid"
            )
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
            for value in self.component_thresholds
        ):
            raise ActualTargetBridgeError(
                "component thresholds must be finite and lie in (0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActualTargetBridgeResultV1:
    """Record plus reproducible component/event maps used to create it."""

    record: PairedSpatialFeatureRecord
    observed_component_errors: torch.Tensor
    clean_component_errors: torch.Tensor
    observed_component_events: torch.Tensor
    clean_component_events: torch.Tensor
    policy: FailureEventPolicyV1
    schema_version: str = ACTUAL_TARGET_BRIDGE_SCHEMA_VERSION


def _require_float_tensor(
    value: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    unit_interval: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ActualTargetBridgeError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ActualTargetBridgeError(f"{name} must be finite floating point")
    if unit_interval and (torch.any(value < 0) or torch.any(value > 1)):
        raise ActualTargetBridgeError(f"{name} must lie in [0, 1]")


def _require_bool_tensor(
    value: torch.Tensor, name: str, shape: tuple[int, ...]
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != torch.bool
    ):
        raise ActualTargetBridgeError(f"{name} must be boolean with shape {shape}")


def _validate_projected_target(
    target: ProjectedObjectFailureTargetV1,
    components: ObjectFailureComponents,
    policy: FailureEventPolicyV1,
    branch: str,
) -> None:
    if target.target_provenance.observation_branch != branch:
        raise ActualTargetBridgeError(
            f"{branch} target must declare observation_branch={branch!r}"
        )
    if target.target_provenance.target_provenance != ACTUAL_TARGET_PROVENANCE:
        raise ActualTargetBridgeError("source target is not actual frozen-ORION error")
    if target.target_provenance.target_version != OBJECT_FAILURE_TARGET_VERSION:
        raise ActualTargetBridgeError("source object-target version is incompatible")
    if target.attribution_is_causal is not False:
        raise ActualTargetBridgeError(
            "projected patch attribution must remain explicitly non-causal"
        )

    if not isinstance(target.support, torch.Tensor) or target.support.ndim != 3:
        raise ActualTargetBridgeError("projected support must have shape [V,P,J]")
    views, patches, objects = target.support.shape
    _require_float_tensor(
        target.support, "projected support", (views, patches, objects), unit_interval=True
    )
    _require_float_tensor(target.error, "projected error", (views, patches), unit_interval=True)
    _require_bool_tensor(target.valid_mask, "projected valid_mask", (views, patches))
    _require_float_tensor(target.object_error, "object_error", (objects,), unit_interval=True)
    _require_bool_tensor(target.object_valid, "object_valid", (objects,))
    tensors = (
        target.error,
        target.valid_mask,
        target.object_error,
        target.object_valid,
    )
    if any(value.device != target.support.device for value in tensors):
        raise ActualTargetBridgeError("all projected-target tensors must share a device")
    if len(target.support_provenance.camera_order) != views:
        raise ActualTargetBridgeError("support camera count differs from provenance")
    if target.support_provenance.patch_count != patches:
        raise ActualTargetBridgeError("support patch count differs from provenance")
    if target.support_provenance.camera_order != target.target_provenance.camera_order:
        raise ActualTargetBridgeError("support and target camera orders disagree")
    if target.support_provenance.image_transform_id != target.target_provenance.image_transform_id:
        raise ActualTargetBridgeError("support and target image transforms disagree")
    if bool((target.error[~target.valid_mask] != 0).any()):
        raise ActualTargetBridgeError("invalid projected cells must use zero placeholders")

    if components.component_names != policy.component_names:
        raise ActualTargetBridgeError(
            "object component names/order differ from the frozen event policy"
        )
    component_count = len(policy.component_names)
    _require_float_tensor(
        components.values,
        "object component values",
        (objects, component_count),
        unit_interval=True,
    )
    _require_bool_tensor(
        components.valid, "object component valid", (objects, component_count)
    )
    _require_float_tensor(
        components.soft_union, "object soft_union", (objects,), unit_interval=True
    )
    _require_bool_tensor(
        components.soft_union_valid, "object soft_union_valid", (objects,)
    )
    component_tensors = (
        components.valid,
        components.soft_union,
        components.soft_union_valid,
    )
    if any(value.device != components.values.device for value in component_tensors):
        raise ActualTargetBridgeError("all object-component tensors must share a device")
    if components.values.device != target.support.device:
        raise ActualTargetBridgeError("components and projected target must share a device")
    prediction_count = components.match.pred_to_gt.shape[0]
    _require_float_tensor(
        components.false_positive_error,
        "false_positive_error",
        (prediction_count,),
        unit_interval=True,
    )
    _require_bool_tensor(
        components.false_positive_valid,
        "false_positive_valid",
        (prediction_count,),
    )
    if (
        components.false_positive_error.device != components.values.device
        or components.false_positive_valid.device != components.values.device
    ):
        raise ActualTargetBridgeError(
            "false-positive and object-component tensors must share a device"
        )
    if components.match.gt_to_pred.shape[0] != objects:
        raise ActualTargetBridgeError(
            "object-component rows must follow the matched GT object order"
        )
    if bool((components.values[~components.valid] != 0).any()):
        raise ActualTargetBridgeError(
            "invalid object-component cells must use zero placeholders, not negative labels"
        )
    motion_index = policy.component_names.index("motion_occupancy")
    if bool(components.valid[:, motion_index].any()) and (
        components.motion_mode_policy != policy.motion_mode_policy
    ):
        raise ActualTargetBridgeError(
            "valid motion component does not use ORION's selected-mode policy"
        )
    traffic_index = policy.component_names.index("traffic_state")
    if bool((~components.valid[:, traffic_index]).any()) and bool(
        (components.values[~components.valid[:, traffic_index], traffic_index] != 0).any()
    ):
        raise ActualTargetBridgeError(
            "missing traffic-state GT must be invalid, never encoded as a negative event"
        )
    recomputed_union, recomputed_valid = bounded_soft_union(
        components.values, components.valid
    )
    if not torch.equal(recomputed_valid, components.soft_union_valid):
        raise ActualTargetBridgeError("component soft_union_valid is inconsistent")
    if not torch.allclose(recomputed_union, components.soft_union, atol=1e-6, rtol=1e-6):
        raise ActualTargetBridgeError("component soft_union is inconsistent")
    if not torch.equal(target.object_valid, components.soft_union_valid):
        raise ActualTargetBridgeError("projected object_valid differs from components")
    if not torch.allclose(target.object_error, components.soft_union, atol=1e-6, rtol=1e-6):
        raise ActualTargetBridgeError("projected object_error differs from components")

    effective_object_error = torch.where(
        target.object_valid, target.object_error, torch.zeros_like(target.object_error)
    )
    reconstructed = 1.0 - torch.prod(
        1.0 - target.support * effective_object_error.view(1, 1, objects), dim=-1
    )
    reconstructed = torch.where(
        target.valid_mask, reconstructed, torch.zeros_like(reconstructed)
    )
    if not torch.allclose(reconstructed, target.error, atol=1e-6, rtol=1e-6):
        raise ActualTargetBridgeError("projected severity is inconsistent with support")

    # ObjectFailureComponents v1 has no projected support for unmatched false
    # positives. Silently dropping them would make the target incomplete.
    if bool(components.false_positive_valid.any()):
        raise ActualTargetBridgeError(
            "unprojected false-positive errors are present; bridge v1 refuses to drop them"
        )


def project_component_errors_to_patches(
    target: ProjectedObjectFailureTargetV1,
    components: ObjectFailureComponents,
    policy: Optional[FailureEventPolicyV1] = None,
) -> torch.Tensor:
    """Return continuous component errors shaped ``[V,P,K]``."""

    policy = policy or FailureEventPolicyV1()
    branch = target.target_provenance.observation_branch
    _validate_projected_target(target, components, policy, branch)
    effective = torch.where(
        components.valid, components.values, torch.zeros_like(components.values)
    )
    factors = 1.0 - target.support.unsqueeze(-1) * effective.view(
        1, 1, *effective.shape
    )
    component_errors = 1.0 - factors.prod(dim=2)
    component_errors = torch.where(
        target.valid_mask.unsqueeze(-1),
        component_errors,
        torch.zeros_like(component_errors),
    )
    return component_errors.clamp(0.0, 1.0)


def object_component_failure_events(
    components: ObjectFailureComponents,
    policy: Optional[FailureEventPolicyV1] = None,
) -> torch.Tensor:
    """Threshold object-level components before any spatial projection."""

    policy = policy or FailureEventPolicyV1()
    if not isinstance(components.values, torch.Tensor) or components.values.ndim != 2:
        raise ActualTargetBridgeError("object component values must have shape [J,K]")
    objects, component_count = components.values.shape
    _require_float_tensor(
        components.values,
        "object component values",
        (objects, component_count),
        unit_interval=True,
    )
    if component_count != len(policy.component_names):
        raise ActualTargetBridgeError("component axis differs from failure-event policy")
    _require_bool_tensor(
        components.valid, "object component valid", (objects, component_count)
    )
    if components.valid.device != components.values.device:
        raise ActualTargetBridgeError(
            "object component values and valid mask must share a device"
        )
    thresholds = components.values.new_tensor(policy.component_thresholds)
    return (components.values >= thresholds.view(1, -1)) & components.valid


def project_object_component_events_to_patches(
    support: torch.Tensor,
    object_component_events: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    policy: Optional[FailureEventPolicyV1] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project hard object events with a separate visibility-support gate."""

    policy = policy or FailureEventPolicyV1()
    if not isinstance(support, torch.Tensor) or support.ndim != 3:
        raise ActualTargetBridgeError("support must have shape [V,P,J]")
    views, patches, objects = support.shape
    _require_float_tensor(
        support, "support", (views, patches, objects), unit_interval=True
    )
    _require_bool_tensor(
        object_component_events,
        "object_component_events",
        (objects, len(policy.component_names)),
    )
    _require_bool_tensor(patch_valid_mask, "patch_valid_mask", (views, patches))
    if (
        object_component_events.device != support.device
        or patch_valid_mask.device != support.device
    ):
        raise ActualTargetBridgeError("event projection tensors must share a device")
    visible = support >= float(policy.minimum_patch_support)
    per_component = (
        visible.unsqueeze(-1) & object_component_events.view(
            1, 1, objects, len(policy.component_names)
        )
    ).any(dim=2)
    per_component = per_component & patch_valid_mask.unsqueeze(-1)
    return per_component, per_component.any(dim=-1) & patch_valid_mask


def bridge_projected_object_targets_to_record_v2(
    *,
    record_id: str,
    pair_id: str,
    severity: float,
    observed_patch_features: torch.Tensor,
    clean_patch_features: torch.Tensor,
    observed_target: ProjectedObjectFailureTargetV1,
    clean_target: ProjectedObjectFailureTargetV1,
    observed_components: ObjectFailureComponents,
    clean_components: ObjectFailureComponents,
    pairing_protocol_id: str,
    corruption_schedule_id: str,
    corruption_mask: Optional[torch.Tensor] = None,
    ensemble_teacher_variance: Optional[torch.Tensor] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    policy: Optional[FailureEventPolicyV1] = None,
) -> ActualTargetBridgeResultV1:
    """Build a v2 actual-target record with independent observed/clean labels."""

    policy = policy or FailureEventPolicyV1()
    if (
        isinstance(severity, bool)
        or not math.isfinite(float(severity))
        or float(severity) < 0
    ):
        raise ActualTargetBridgeError("severity must be finite and non-negative")
    _validate_projected_target(observed_target, observed_components, policy, "observed")
    _validate_projected_target(clean_target, clean_components, policy, "clean")
    if not str(pairing_protocol_id).strip() or not str(corruption_schedule_id).strip():
        raise ActualTargetBridgeError(
            "pairing_protocol_id and corruption_schedule_id must be non-empty"
        )
    observed_identity = observed_target.target_provenance
    clean_identity = clean_target.target_provenance
    normalized_pairing_protocol = str(pairing_protocol_id).strip()
    if (
        observed_identity.paired_history_protocol_id != normalized_pairing_protocol
        or clean_identity.paired_history_protocol_id != normalized_pairing_protocol
    ):
        raise ActualTargetBridgeError(
            "pairing_protocol_id must equal both branches' paired_history_protocol_id"
        )
    # Branch histories may legitimately differ when clean/corrupt streams are
    # replayed chronologically. They are preserved separately below and are not
    # falsely asserted equal. All state/geometry/config fields that make the
    # frame a valid pair remain strict.
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
        raise ActualTargetBridgeError(
            "paired target provenance differs in: " + ", ".join(disagreements)
        )
    if not torch.equal(observed_target.support, clean_target.support):
        raise ActualTargetBridgeError(
            "observed and clean projected supports differ; object ordering/geometry is ambiguous"
        )

    expected_spatial = observed_target.error.shape
    for value, name in (
        (observed_patch_features, "observed_patch_features"),
        (clean_patch_features, "clean_patch_features"),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ActualTargetBridgeError(f"{name} must have shape [V,P,D]")
        if value.shape[:-1] != expected_spatial:
            raise ActualTargetBridgeError(
                f"{name} view/patch axes differ from the projected target"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ActualTargetBridgeError(f"{name} must be finite floating point")
    if observed_patch_features.shape != clean_patch_features.shape:
        raise ActualTargetBridgeError("observed and clean feature shapes differ")

    observed_component_errors = project_component_errors_to_patches(
        observed_target, observed_components, policy
    )
    clean_component_errors = project_component_errors_to_patches(
        clean_target, clean_components, policy
    )
    observed_object_events = object_component_failure_events(
        observed_components, policy
    )
    clean_object_events = object_component_failure_events(clean_components, policy)
    observed_component_events, observed_event = project_object_component_events_to_patches(
        observed_target.support,
        observed_object_events,
        observed_target.valid_mask,
        policy,
    )
    clean_component_events, clean_event = project_object_component_events_to_patches(
        clean_target.support,
        clean_object_events,
        clean_target.valid_mask,
        policy,
    )

    reserved_metadata = "actual_target_bridge"
    record_metadata = dict(metadata or {})
    if reserved_metadata in record_metadata:
        raise ActualTargetBridgeError(
            f"metadata key {reserved_metadata!r} is reserved by the bridge"
        )
    source = observed_target.target_provenance
    joint_valid = observed_target.valid_mask & clean_target.valid_mask
    delta_error = torch.where(
        joint_valid,
        torch.relu(observed_target.error - clean_target.error),
        torch.zeros_like(observed_target.error),
    )
    record_metadata[reserved_metadata] = {
        "schema_version": ACTUAL_TARGET_BRIDGE_SCHEMA_VERSION,
        "paired_record_schema_version": PAIRED_RECORD_SCHEMA_VERSION,
        "target_contract_schema_version": TARGET_CONTRACT_SCHEMA_VERSION,
        "record_target_provenance": TARGET_ACTUAL_FAILURE,
        "source_target_provenance": ACTUAL_TARGET_PROVENANCE,
        "source_target_version": OBJECT_FAILURE_TARGET_VERSION,
        "failure_event_policy": policy.to_dict(),
        "severity_semantics": "projected_noisy_or_of_object_soft_union_error",
        "event_semantics": EVENT_AGGREGATION,
        "event_minimum_patch_support": policy.minimum_patch_support,
        "severity_is_event_probability": False,
        "observed_clean_measured_independently": True,
        "pairing_protocol_id": normalized_pairing_protocol,
        "corruption_schedule_id": str(corruption_schedule_id).strip(),
        "temporal_histories": {
            "observed": observed_identity.temporal_history_id,
            "clean": clean_identity.temporal_history_id,
            "required_equal": False,
        },
        "invalid_patch_policy": "masked_not_negative",
        "component_error_axis": -1,
        "component_error_names": list(policy.component_names),
        "delta_error_role": "audit_and_ranking_only_not_primary_target",
        "delta_error_mean_on_joint_valid": (
            float(delta_error[joint_valid].mean())
            if bool(joint_valid.any())
            else None
        ),
        "source_identity": {
            "base_checkpoint_sha256": source.base_checkpoint_sha256,
            "inference_config_sha256": source.inference_config_sha256,
            "git_revision": source.git_revision,
            "route_id": source.route_id,
            "town": source.town,
            "frame_idx": source.frame_idx,
            "temporal_history_id": source.temporal_history_id,
            "paired_history_protocol_id": source.paired_history_protocol_id,
            "class_mapping_id": source.class_mapping_id,
            "decoder_policy_id": source.decoder_policy_id,
            "camera_order": list(source.camera_order),
            "image_transform_id": source.image_transform_id,
        },
        "known_limitations": [
            "patch attribution is projected support, not pixel causality",
            "bridge v1 rejects unprojected false-positive object errors",
            "object identity/order is checked through equal projected support but actor IDs are not carried",
            "fixed 0.50 component thresholds require calibration-split sensitivity reporting",
            "visible support below the independent 0.01 event-projection gate remains unlabeled",
        ],
    }

    def cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu()

    record = PairedSpatialFeatureRecord(
        record_id=record_id,
        pair_id=pair_id,
        route_id=source.route_id,
        town=source.town,
        severity=float(severity),
        observed_patch_features=cpu(observed_patch_features).float(),
        clean_patch_features=cpu(clean_patch_features).float(),
        error_severity_target=cpu(observed_target.error).float(),
        failure_event_target=cpu(observed_event),
        target_valid_mask=cpu(observed_target.valid_mask),
        clean_error_severity_target=cpu(clean_target.error).float(),
        clean_failure_event_target=cpu(clean_event),
        clean_target_valid_mask=cpu(clean_target.valid_mask),
        component_errors=cpu(observed_component_errors).float(),
        clean_component_errors=cpu(clean_component_errors).float(),
        component_error_names=policy.component_names,
        component_error_axis=-1,
        corruption_mask=(cpu(corruption_mask).float() if corruption_mask is not None else None),
        ensemble_teacher_variance=(
            cpu(ensemble_teacher_variance).float()
            if ensemble_teacher_variance is not None
            else None
        ),
        metadata=record_metadata,
    )
    return ActualTargetBridgeResultV1(
        record=record,
        observed_component_errors=cpu(observed_component_errors),
        clean_component_errors=cpu(clean_component_errors),
        observed_component_events=cpu(observed_component_events),
        clean_component_events=cpu(clean_component_events),
        policy=policy,
    )


__all__ = [
    "ACTUAL_TARGET_BRIDGE_SCHEMA_VERSION",
    "CANONICAL_COMPONENT_NAMES",
    "COMPONENT_PATCH_AGGREGATION",
    "EVENT_AGGREGATION",
    "EVENT_PATCH_PROJECTION",
    "FAILURE_EVENT_POLICY_SCHEMA_VERSION",
    "FROZEN_COMPONENT_THRESHOLDS_V1",
    "FROZEN_MIN_PATCH_SUPPORT_V1",
    "ActualTargetBridgeError",
    "ActualTargetBridgeResultV1",
    "FailureEventPolicyV1",
    "bridge_projected_object_targets_to_record_v2",
    "object_component_failure_events",
    "project_object_component_events_to_patches",
    "project_component_errors_to_patches",
]
