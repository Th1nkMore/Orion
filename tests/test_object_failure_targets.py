from dataclasses import replace

import pytest
import torch

from uq_estimator.object_failure_targets import (
    MOTION_MODE_POLICY,
    ObjectFailureTargetError,
    PatchSupportProvenanceV1,
    TargetProvenanceV1,
    aggregate_projected_visible_support,
    bounded_soft_union,
    class_aware_distance_gated_match,
    compute_object_failure_components,
    make_bev_occupancy_error_sidecar,
    make_paired_failure_targets,
)


CAMERAS = ("CAM_FRONT", "CAM_LEFT")


def _support_provenance() -> PatchSupportProvenanceV1:
    return PatchSupportProvenanceV1(
        camera_order=CAMERAS,
        image_hw=(80, 120),
        patch_hw=(2, 3),
        image_transform_id="deterministic-eval-transform-v1",
    )


def _target_provenance(branch: str) -> TargetProvenanceV1:
    return TargetProvenanceV1(
        base_checkpoint_sha256="a" * 64,
        inference_config_sha256="b" * 64,
        git_revision="git-deadbeef",
        route_id="route_146",
        town="Town01",
        frame_idx=42,
        observation_branch=branch,
        temporal_history_id=f"route_146-{branch}-history-content",
        paired_history_protocol_id="route_146-replay-from-frame-zero-v1",
        class_mapping_id="b2d-orion-classes-v1",
        decoder_policy_id="last-layer-thresholds-calibration-split-v1",
        camera_order=CAMERAS,
        image_transform_id="deterministic-eval-transform-v1",
    )


def _simple_match(max_distance: float = 4.0):
    gt_centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    pred_centers = torch.tensor([[1.0, 0.0], [11.0, 0.0], [40.0, 0.0]])
    return class_aware_distance_gated_match(
        gt_centers,
        torch.tensor([0, 1, 2]),
        pred_centers,
        torch.tensor([0, 1, 2]),
        torch.tensor([0.9, 0.8, 0.95]),
        coordinate_frame="lidar",
        max_center_distance=max_distance,
        minimum_prediction_score=0.5,
    )


def test_matching_rejects_wrong_class_far_and_low_score_forced_assignments():
    gt = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    pred = torch.tensor([[0.1, 0.0], [20.5, 0.0], [10.1, 0.0]])
    result = class_aware_distance_gated_match(
        gt,
        torch.tensor([0, 1, 2]),
        pred,
        torch.tensor([9, 2, 1]),
        torch.tensor([0.99, 0.95, 0.1]),
        coordinate_frame="lidar",
        max_center_distance=4.0,
        minimum_prediction_score=0.5,
    )
    assert result.gt_to_pred.tolist() == [-1, -1, 1]
    assert result.pred_to_gt.tolist() == [-1, 2, -1]
    assert not result.valid_pair_mask[0, 0]  # close but wrong class
    assert not result.valid_pair_mask[1, 2]  # correct class but below score gate


def test_matching_maximizes_valid_cardinality_before_distance():
    # GT0 can use either prediction; GT1 can only use pred0. A nearest-first
    # greedy matcher would consume pred0 for GT0 and leave GT1 unmatched.
    gt = torch.tensor([[0.0, 0.0], [1.9, 0.0]])
    pred = torch.tensor([[0.1, 0.0], [-1.0, 0.0]])
    result = class_aware_distance_gated_match(
        gt,
        torch.tensor([0, 0]),
        pred,
        torch.tensor([0, 0]),
        torch.ones(2),
        coordinate_frame="lidar",
        max_center_distance=torch.tensor([2.0, 2.0]),
    )
    assert result.matched_gt.all()
    assert result.gt_to_pred.tolist() == [1, 0]


def test_matching_empty_sets_and_distance_sensitivity():
    empty = class_aware_distance_gated_match(
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([0.8]),
        coordinate_frame="lidar",
    )
    assert empty.gt_to_pred.numel() == 0
    assert empty.unmatched_prediction.tolist() == [True]

    assert _simple_match(4.0).gt_to_pred.tolist() == [0, 1, -1]
    assert _simple_match(0.5).gt_to_pred.tolist() == [-1, -1, -1]


@pytest.mark.parametrize(
    "kwargs,pattern",
    (
        ({"coordinate_frame": "camera"}, "lidar"),
        ({"max_center_distance": 0.0}, "positive"),
        ({"minimum_prediction_score": 1.1}, r"\[0, 1\]"),
    ),
)
def test_matching_fails_closed_on_geometry_or_thresholds(kwargs, pattern):
    base = dict(
        gt_centers=torch.tensor([[0.0, 0.0]]),
        gt_classes=torch.tensor([0]),
        pred_centers=torch.tensor([[0.0, 0.0]]),
        pred_classes=torch.tensor([0]),
        pred_scores=torch.tensor([1.0]),
        coordinate_frame="lidar",
    )
    base.update(kwargs)
    with pytest.raises(ObjectFailureTargetError, match=pattern):
        class_aware_distance_gated_match(**base)


def test_components_remain_separate_bounded_and_soft_union_is_auditable():
    match = _simple_match()
    class_probabilities = torch.tensor(
        [
            [0.75, 0.10, 0.15],
            [0.10, 0.60, 0.30],
            [0.05, 0.05, 0.90],
        ]
    )
    pairwise_iou = torch.zeros(3, 3)
    pairwise_iou[0, 0] = 0.5
    pairwise_iou[1, 1] = 0.9

    gt_occ = torch.zeros(3, 2, 2, 2)
    pred_occ = torch.zeros(3, 2, 2, 2)
    gt_occ[0, :, 0, 0] = 1.0
    pred_occ[0, :, 0, 0] = 0.5
    gt_occ[1, :, 1, 1] = 1.0
    pred_occ[1, :, 1, 1] = 1.0
    gt_occ[2, :, 0, 1] = 1.0

    result = compute_object_failure_components(
        match,
        torch.tensor([0, 1, 2]),
        class_probabilities,
        torch.tensor([0.9, 0.8, 0.95]),
        pairwise_bev_iou=pairwise_iou,
        gt_motion_occupancy=gt_occ,
        pred_motion_occupancy=pred_occ,
        motion_valid_mask=torch.ones(3, 2, dtype=torch.bool),
        motion_mode_policy=MOTION_MODE_POLICY,
        gt_traffic_states=torch.tensor([0, 1, 0]),
        pred_traffic_state_probabilities=torch.tensor(
            [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]]
        ),
        traffic_state_valid=torch.tensor([False, True, False]),
    )

    assert result.component_names == (
        "miss",
        "class",
        "localization",
        "motion_occupancy",
        "traffic_state",
    )
    assert result.values.shape == (3, 5)
    assert torch.all((result.values >= 0) & (result.values <= 1))
    assert result.values[:, 0].tolist() == [0.0, 0.0, 1.0]
    assert result.values[0, 1].item() == pytest.approx(0.25)
    assert result.values[0, 2].item() == pytest.approx(0.5)
    assert result.values[0, 3].item() == pytest.approx(0.5)
    assert result.values[1, 4].item() == pytest.approx(0.1)
    assert result.soft_union[0].item() == pytest.approx(0.8125)
    assert result.soft_union[2].item() == 1.0
    assert result.false_positive_valid.tolist() == [False, False, True]
    assert result.false_positive_error.tolist() == pytest.approx([0.0, 0.0, 0.95])


def test_missing_object_gets_motion_and_state_error_one_when_labels_are_valid():
    match = class_aware_distance_gated_match(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([0]),
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        coordinate_frame="lidar",
    )
    result = compute_object_failure_components(
        match,
        torch.tensor([0]),
        torch.empty(0, 2),
        torch.empty(0),
        gt_motion_occupancy=torch.ones(1, 1, 2, 2),
        pred_motion_occupancy=torch.empty(0, 1, 2, 2),
        motion_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        motion_mode_policy=MOTION_MODE_POLICY,
        gt_traffic_states=torch.tensor([1]),
        pred_traffic_state_probabilities=torch.empty(0, 2),
        traffic_state_valid=torch.tensor([True]),
    )
    assert result.values[0, [0, 3, 4]].tolist() == [1.0, 1.0, 1.0]
    assert not result.valid[0, 1]
    assert not result.valid[0, 2]


def test_primary_motion_target_rejects_oracle_best_mode():
    match = _simple_match()
    with pytest.raises(ObjectFailureTargetError, match="selected mode"):
        compute_object_failure_components(
            match,
            torch.tensor([0, 1, 2]),
            torch.eye(3),
            torch.ones(3),
            gt_motion_occupancy=torch.zeros(3, 1, 2, 2),
            pred_motion_occupancy=torch.zeros(3, 1, 2, 2),
            motion_valid_mask=torch.ones(3, 1, dtype=torch.bool),
            motion_mode_policy="oracle_best_mode",
        )


def test_soft_union_ignores_invalid_components_without_zero_labeling_them():
    values = torch.tensor([[0.2, 0.5], [0.9, 0.8]])
    valid = torch.tensor([[True, True], [False, False]])
    union, union_valid = bounded_soft_union(values, valid)
    assert union.tolist() == pytest.approx([0.6, 0.0])
    assert union_valid.tolist() == [True, False]


def test_noisy_or_projected_support_and_explicit_patch_validity():
    support = torch.zeros(2, 6, 2)
    support[0, 0, :] = torch.tensor([1.0, 1.0])
    support[0, 1, 0] = 0.5
    support[1, 5, 1] = 1.0
    patch_valid = torch.ones(2, 6, dtype=torch.bool)
    patch_valid[1, 5] = False
    target = aggregate_projected_visible_support(
        support,
        torch.tensor([0.5, 0.4]),
        torch.tensor([True, True]),
        patch_valid,
        support_provenance=_support_provenance(),
        target_provenance=_target_provenance("observed"),
    )
    assert target.error[0, 0].item() == pytest.approx(0.7)
    assert target.error[0, 1].item() == pytest.approx(0.25)
    assert target.error[1, 5].item() == 0.0
    assert not target.valid_mask[1, 5]
    assert target.attribution_is_causal is False


def test_projected_support_refuses_raw_calibration_or_camera_mismatch():
    with pytest.raises(ObjectFailureTargetError, match="post-augmentation"):
        PatchSupportProvenanceV1(
            camera_order=CAMERAS,
            image_hw=(80, 120),
            patch_hw=(2, 3),
            image_transform_id="x",
            projection_matrix_kind="raw_lidar2img",
        )

    with pytest.raises(ObjectFailureTargetError, match="camera_order"):
        aggregate_projected_visible_support(
            torch.zeros(2, 6, 1),
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            torch.ones(2, 6, dtype=torch.bool),
            support_provenance=_support_provenance(),
            target_provenance=replace(
                _target_provenance("observed"),
                camera_order=("CAM_FRONT", "CAM_RIGHT"),
            ),
        )


def test_paired_targets_keep_clean_error_and_delta_separate():
    support = torch.ones(2, 6, 1)
    valid = torch.ones(2, 6, dtype=torch.bool)
    object_valid = torch.ones(1, dtype=torch.bool)
    observed = aggregate_projected_visible_support(
        support,
        torch.tensor([0.8]),
        object_valid,
        valid,
        support_provenance=_support_provenance(),
        target_provenance=_target_provenance("observed"),
    )
    clean = aggregate_projected_visible_support(
        support,
        torch.tensor([0.3]),
        object_valid,
        valid,
        support_provenance=_support_provenance(),
        target_provenance=_target_provenance("clean"),
    )
    paired = make_paired_failure_targets(observed, clean)
    assert torch.allclose(paired.observed_error, torch.full((2, 6), 0.8))
    assert torch.allclose(paired.clean_error, torch.full((2, 6), 0.3))
    assert torch.allclose(paired.delta_error, torch.full((2, 6), 0.5))


def test_paired_targets_allow_branch_specific_history_but_reject_protocol_mismatch():
    support = torch.ones(2, 6, 1)
    valid = torch.ones(2, 6, dtype=torch.bool)
    observed = aggregate_projected_visible_support(
        support,
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        valid,
        support_provenance=_support_provenance(),
        target_provenance=_target_provenance("observed"),
    )
    clean = aggregate_projected_visible_support(
        support,
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        valid,
        support_provenance=_support_provenance(),
        target_provenance=replace(
            _target_provenance("clean"),
            paired_history_protocol_id="reset-at-event-incompatible-protocol",
        ),
    )
    with pytest.raises(ObjectFailureTargetError, match="paired_history_protocol_id"):
        make_paired_failure_targets(observed, clean)


def test_bev_sidecar_preserves_bev_tensor_and_masks_invalid_cells():
    gt = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    pred = torch.tensor([[[0.5, 0.2], [0.0, 0.0]]])
    valid = torch.tensor([[[True, True], [False, True]]])
    sidecar = make_bev_occupancy_error_sidecar(
        gt,
        pred,
        valid,
        gt_provenance="PlanningMetric rasterized B2D current/future boxes",
        prediction_provenance="rasterized frozen-ORION selected-mode boxes/motion",
        coordinate_frame="ego_bev",
        bev_bounds_xyxy=(-50.0, -50.0, 50.0, 50.0),
        resolution_m=0.5,
    )
    assert sidecar.absolute_error.shape == (1, 2, 2)
    assert torch.allclose(
        sidecar.absolute_error,
        torch.tensor([[[0.5, 0.2], [0.0, 1.0]]]),
    )
    assert sidecar.resolution_m == 0.5


def test_provenance_requires_real_hashes_and_known_branch():
    with pytest.raises(ObjectFailureTargetError, match="SHA-256"):
        replace(_target_provenance("clean"), base_checkpoint_sha256="unknown")
    with pytest.raises(ObjectFailureTargetError, match="observed.*clean"):
        replace(_target_provenance("clean"), observation_branch="corrupt")
