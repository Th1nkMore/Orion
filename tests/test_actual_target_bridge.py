"""CPU tests for the projected-object to paired-record v2 bridge."""

from dataclasses import replace

import pytest
import torch

from uq_estimator.actual_target_bridge import (
    ACTUAL_TARGET_BRIDGE_SCHEMA_VERSION,
    FAILURE_EVENT_POLICY_SCHEMA_VERSION,
    ActualTargetBridgeError,
    FailureEventPolicyV1,
    bridge_projected_object_targets_to_record_v2,
)
from uq_estimator.object_failure_targets import (
    MOTION_MODE_POLICY,
    PatchSupportProvenanceV1,
    TargetProvenanceV1,
    aggregate_projected_visible_support,
    class_aware_distance_gated_match,
    compute_object_failure_components,
)
from uq_estimator.spatial_training import (
    TARGET_ACTUAL_FAILURE,
    load_paired_feature_records,
    save_paired_feature_records,
)


def _support_provenance():
    return PatchSupportProvenanceV1(
        camera_order=("CAM_FRONT",),
        image_hw=(80, 120),
        patch_hw=(1, 3),
        image_transform_id="deterministic-eval-v1",
    )


def _target_provenance(branch: str, history: str):
    return TargetProvenanceV1(
        base_checkpoint_sha256="a" * 64,
        inference_config_sha256="b" * 64,
        git_revision="git-deadbeef",
        route_id="route_146",
        town="Town01",
        frame_idx=42,
        observation_branch=branch,
        temporal_history_id=history,
        paired_history_protocol_id="chronological-dual-replay/v1",
        class_mapping_id="b2d-orion-classes-v1",
        decoder_policy_id="last-layer-thresholds-calibration-v1",
        camera_order=("CAM_FRONT",),
        image_transform_id="deterministic-eval-v1",
    )


def _components(correct_probabilities, diagonal_iou):
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    classes = torch.tensor([0, 1])
    match = class_aware_distance_gated_match(
        centers,
        classes,
        centers.clone(),
        classes.clone(),
        torch.ones(2),
        coordinate_frame="lidar",
        max_center_distance=4.0,
        minimum_prediction_score=0.5,
    )
    probabilities = torch.tensor(
        [
            [correct_probabilities[0], 1.0 - correct_probabilities[0]],
            [1.0 - correct_probabilities[1], correct_probabilities[1]],
        ]
    )
    iou = torch.zeros(2, 2)
    iou[0, 0] = diagonal_iou[0]
    iou[1, 1] = diagonal_iou[1]
    # Include selected-mode motion so the bridge can audit its semantics. The
    # identical occupancies yield zero valid motion error.
    occupancy = torch.ones(2, 1, 1, 1)
    return compute_object_failure_components(
        match,
        classes,
        probabilities,
        torch.ones(2),
        pairwise_bev_iou=iou,
        gt_motion_occupancy=occupancy,
        pred_motion_occupancy=occupancy.clone(),
        motion_valid_mask=torch.ones(2, 1, dtype=torch.bool),
        motion_mode_policy=MOTION_MODE_POLICY,
        # Only object 1 has traffic-state GT. Object 0 must remain invalid,
        # not be converted into a negative event.
        gt_traffic_states=torch.tensor([0, 1]),
        pred_traffic_state_probabilities=torch.tensor([[0.5, 0.5], [0.0, 1.0]]),
        traffic_state_valid=torch.tensor([False, True]),
    )


def _case():
    observed_components = _components((0.6, 0.4), (0.6, 0.9))
    clean_components = _components((0.9, 0.9), (0.9, 0.9))
    support = torch.zeros(1, 3, 2)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0
    observed_valid = torch.tensor([[True, True, False]])
    clean_valid = torch.tensor([[False, True, True]])
    observed = aggregate_projected_visible_support(
        support,
        observed_components.soft_union,
        observed_components.soft_union_valid,
        observed_valid,
        support_provenance=_support_provenance(),
        target_provenance=_target_provenance("observed", "observed-history-chain"),
    )
    clean = aggregate_projected_visible_support(
        support,
        clean_components.soft_union,
        clean_components.soft_union_valid,
        clean_valid,
        support_provenance=_support_provenance(),
        target_provenance=_target_provenance("clean", "clean-history-chain"),
    )
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    return observed, clean, observed_components, clean_components, features


def _bridge(**overrides):
    observed, clean, observed_components, clean_components, features = _case()
    arguments = {
        "record_id": "route_146/frame_42/local_blur_2",
        "pair_id": "route_146/frame_42/local_blur",
        "severity": 2.0,
        "observed_patch_features": features + 0.1,
        "clean_patch_features": features,
        "observed_target": observed,
        "clean_target": clean,
        "observed_components": observed_components,
        "clean_components": clean_components,
        "pairing_protocol_id": "chronological-dual-replay/v1",
        "corruption_schedule_id": "local-blur-window-seed-7/v1",
        "corruption_mask": torch.tensor([[1.0, 1.0, 0.0]]),
    }
    arguments.update(overrides)
    return bridge_projected_object_targets_to_record_v2(**arguments)


def test_bridge_separates_continuous_severity_from_component_level_event():
    result = _bridge()
    record = result.record
    # Patch 0 has two moderate components: soft-union severity is 0.64, but no
    # individual component crosses the frozen 0.5 event threshold.
    assert record.error_severity_target[0, 0].item() == pytest.approx(0.64)
    assert record.failure_event_target[0, 0].item() is False
    # Patch 1 has the same 0.64 total severity, but its class error is 0.6 and
    # therefore is a component-defined event.
    assert record.error_severity_target[0, 1].item() == pytest.approx(0.64)
    assert record.failure_event_target[0, 1].item() is True
    assert not torch.equal(
        record.error_severity_target, record.failure_event_target.float()
    )
    assert record.target_provenance == TARGET_ACTUAL_FAILURE


def test_bridge_preserves_independent_clean_targets_masks_and_components():
    result = _bridge()
    record = result.record
    assert record.target_valid_mask.tolist() == [[True, True, False]]
    assert record.clean_target_valid_mask.tolist() == [[False, True, True]]
    assert record.clean_error_severity_target[0, 1].item() == pytest.approx(0.19)
    assert not record.clean_failure_event_target.any()
    assert record.component_error_names == (
        "miss",
        "class",
        "localization",
        "motion_occupancy",
        "traffic_state",
    )
    assert record.component_error_axis == -1
    assert record.component_errors.shape == (1, 3, 5)
    assert record.clean_component_errors.shape == (1, 3, 5)
    # Object 0 has no traffic-state GT: its component is invalid/zero, not a
    # negative label supplied to the event policy.
    assert result.observed_component_errors[0, 0, 4].item() == 0.0


def test_policy_and_branch_histories_are_auditable_and_round_trip(tmp_path):
    result = _bridge()
    bridge_metadata = result.record.metadata["actual_target_bridge"]
    assert bridge_metadata["schema_version"] == ACTUAL_TARGET_BRIDGE_SCHEMA_VERSION
    assert (
        bridge_metadata["failure_event_policy"]["schema_version"]
        == FAILURE_EVENT_POLICY_SCHEMA_VERSION
    )
    assert bridge_metadata["failure_event_policy"]["component_thresholds"] == (
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    )
    assert bridge_metadata["failure_event_policy"]["minimum_patch_support"] == 0.01
    assert bridge_metadata["failure_event_policy"]["event_aggregation"] == (
        "threshold_object_components_then_project_support_then_any"
    )
    assert bridge_metadata["failure_event_policy"]["motion_mode_policy"] == MOTION_MODE_POLICY
    assert bridge_metadata["failure_event_policy"]["missing_traffic_state_policy"] == (
        "component_invalid_not_negative"
    )
    assert bridge_metadata["temporal_histories"] == {
        "observed": "observed-history-chain",
        "clean": "clean-history-chain",
        "required_equal": False,
    }
    assert bridge_metadata["pairing_protocol_id"] == "chronological-dual-replay/v1"
    assert bridge_metadata["corruption_schedule_id"] == "local-blur-window-seed-7/v1"

    path = tmp_path / "actual-record.pt"
    save_paired_feature_records(path, [result.record])
    loaded = load_paired_feature_records(path)[0]
    assert loaded.metadata["actual_target_bridge"] == bridge_metadata
    torch.testing.assert_close(loaded.component_errors, result.record.component_errors)


def test_policy_is_frozen_and_cannot_change_thresholds_under_v1():
    with pytest.raises(ActualTargetBridgeError, match="thresholds are frozen"):
        FailureEventPolicyV1(component_thresholds=(0.25, 0.5, 0.5, 0.5, 0.5))
    with pytest.raises(ActualTargetBridgeError, match="selected mode"):
        FailureEventPolicyV1(motion_mode_policy="oracle_best_mode")
    with pytest.raises(ActualTargetBridgeError, match="minimum patch support is frozen"):
        FailureEventPolicyV1(minimum_patch_support=0.10)

    _, _, observed_components, _, _ = _case()
    with pytest.raises(ActualTargetBridgeError, match="selected-mode"):
        _bridge(
            observed_components=replace(
                observed_components, motion_mode_policy="oracle_best_mode"
            )
        )

    invalid_values = observed_components.values.clone()
    invalid_values[0, 4] = 0.8  # traffic GT is missing/invalid for object 0
    with pytest.raises(ActualTargetBridgeError, match="invalid object-component"):
        _bridge(
            observed_components=replace(
                observed_components, values=invalid_values
            )
        )


def test_small_support_missed_pedestrian_remains_event_positive():
    gt_center = torch.tensor([[0.0, 0.0]])
    gt_class = torch.tensor([0])
    observed_match = class_aware_distance_gated_match(
        gt_center,
        gt_class,
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        coordinate_frame="lidar",
    )
    observed_components = compute_object_failure_components(
        observed_match,
        gt_class,
        torch.empty(0, 1),
        torch.empty(0),
    )
    clean_match = class_aware_distance_gated_match(
        gt_center,
        gt_class,
        gt_center.clone(),
        gt_class.clone(),
        torch.ones(1),
        coordinate_frame="lidar",
    )
    clean_components = compute_object_failure_components(
        clean_match,
        gt_class,
        torch.ones(1, 1),
        torch.ones(1),
        pairwise_bev_iou=torch.ones(1, 1),
    )
    support = torch.tensor([[[0.10]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    support_provenance = PatchSupportProvenanceV1(
        camera_order=("CAM_FRONT",),
        image_hw=(80, 120),
        patch_hw=(1, 1),
        image_transform_id="deterministic-eval-v1",
    )
    observed = aggregate_projected_visible_support(
        support,
        observed_components.soft_union,
        observed_components.soft_union_valid,
        valid,
        support_provenance=support_provenance,
        target_provenance=_target_provenance("observed", "observed-history-chain"),
    )
    clean = aggregate_projected_visible_support(
        support,
        clean_components.soft_union,
        clean_components.soft_union_valid,
        valid,
        support_provenance=support_provenance,
        target_provenance=_target_provenance("clean", "clean-history-chain"),
    )
    result = bridge_projected_object_targets_to_record_v2(
        record_id="small-pedestrian",
        pair_id="small-pedestrian-pair",
        severity=1.0,
        observed_patch_features=torch.ones(1, 1, 2),
        clean_patch_features=torch.ones(1, 1, 2),
        observed_target=observed,
        clean_target=clean,
        observed_components=observed_components,
        clean_components=clean_components,
        pairing_protocol_id="chronological-dual-replay/v1",
        corruption_schedule_id="small-pedestrian-window/v1",
    )
    # Fractional severity remains 0.1, but the object-level miss is hard true
    # and support 0.1 passes the independent 0.01 visibility gate.
    assert result.record.error_severity_target.item() == pytest.approx(0.10)
    assert result.record.failure_event_target.item() is True
    assert result.observed_component_events[0, 0, 0].item() is True


def test_bridge_fails_closed_on_tampering_geometry_and_unprojected_fp():
    observed, clean, observed_components, clean_components, features = _case()
    tampered_error = observed.error.clone()
    tampered_error[0, 0] += 0.01
    tampered = replace(observed, error=tampered_error)
    with pytest.raises(ActualTargetBridgeError, match="severity is inconsistent"):
        _bridge(observed_target=tampered)

    changed_support = clean.support.clone()
    changed_support[0, 2, 0] = 0.1
    changed_clean = aggregate_projected_visible_support(
        changed_support,
        clean_components.soft_union,
        clean_components.soft_union_valid,
        clean.valid_mask,
        support_provenance=clean.support_provenance,
        target_provenance=clean.target_provenance,
    )
    with pytest.raises(ActualTargetBridgeError, match="supports differ"):
        _bridge(clean_target=changed_clean)

    fp_components = replace(
        observed_components,
        false_positive_valid=torch.tensor([True, False]),
        false_positive_error=torch.tensor([0.9, 0.0]),
    )
    with pytest.raises(ActualTargetBridgeError, match="false-positive"):
        _bridge(observed_components=fp_components)

    with pytest.raises(ActualTargetBridgeError, match="pairing_protocol_id"):
        _bridge(pairing_protocol_id="")


def test_bridge_allows_distinct_histories_but_rejects_frame_or_config_mismatch():
    result = _bridge()
    assert result.record.metadata["actual_target_bridge"]["temporal_histories"][
        "required_equal"
    ] is False

    _, clean, _, _, _ = _case()
    wrong_frame = replace(
        clean,
        target_provenance=replace(clean.target_provenance, frame_idx=43),
    )
    with pytest.raises(ActualTargetBridgeError, match="frame_idx"):
        _bridge(clean_target=wrong_frame)

    wrong_config = replace(
        clean,
        target_provenance=replace(
            clean.target_provenance, inference_config_sha256="c" * 64
        ),
    )
    with pytest.raises(ActualTargetBridgeError, match="inference_config_sha256"):
        _bridge(clean_target=wrong_config)
